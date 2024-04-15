import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from copy import deepcopy

import sys
sys.path.append('../../')
from models.shared import CosineWarmupScheduler, Encoder, Decoder, visualize_ae_reconstruction, SimpleEncoder, SimpleDecoder
import wandb


class Autoencoder(pl.LightningModule):
    """ Simple Autoencoder network """

    def __init__(self, num_latents,
                       c_in=3,
                       c_hid=64,
                       lr=1e-3,
                       warmup=500, 
                       max_iters=100000,
                       img_width=64,
                       noise_level=0.05,
                       regularizer_weight=1e-4,
                       action_size=-1,
                       mi_reg_weight=0.0,
                       latent_mi_reg_weight=0.0,
                       whole_episode_contrastive=False,
                       triplet_contrastive=False,
                       reconstructive=True,
                       similarity='cosine',
                       use_infonce_loss=False,
                       use_scoring_network=False,
                       num_negative_samples=10,
                       infonce_loss_weight=1.0,
                       negative_sampling_mode='temporal_distance',
                       temporal_bias_strength=1.0,
                       **kwargs):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latents in the bottleneck.
        c_in : int
               Number of input channels (3 for RGB)
        c_hid : int
                Hidden dimensionality to use in the network
        lr : float
             Learning rate to use for training.
        warmup : int
                 Number of learning rate warmup steps
        max_iters : int
                    Number of max. training iterations. Needed for 
                    cosine annealing of the learning rate.
        img_width : int
                    Width of the input image (assumed to be equal to height)
        noise_level : float
                      Standard deviation of the added noise to the latents.
        """
        super().__init__()
        self.reconstructive = reconstructive
        self.contrastive = (whole_episode_contrastive or triplet_contrastive or use_infonce_loss)
        self.similarity = CosineSimilarity() if similarity == 'cosine' else EuclideanSimilarity()
        self.negative_sampling_mode = negative_sampling_mode
        self.temporal_bias_strength = temporal_bias_strength
        assert not (whole_episode_contrastive and triplet_contrastive), 'Cannot use both whole episode and triplet contrastive loss'
        self.save_hyperparameters()

        if self.hparams.img_width == 32:
            EncoderClass = SimpleEncoder
            DecoderClass = SimpleDecoder
        else:
            EncoderClass = Encoder
            DecoderClass = Decoder
        use_coordconv = kwargs.get('use_coordconv', False)
        self.encoder = EncoderClass(num_latents=self.hparams.num_latents,
                                    c_hid=self.hparams.c_hid,
                                    c_in=self.hparams.c_in,
                                    width=self.hparams.img_width,
                                    act_fn=nn.SiLU,
                                    residual=True,
                                    num_blocks=2,
                                    variational=False,
                                    use_coordconv=use_coordconv)
        if self.reconstructive:
            self.decoder = DecoderClass(num_latents=self.hparams.num_latents + max(0, self.hparams.action_size),
                                        c_hid=self.hparams.c_hid,
                                        c_out=self.hparams.c_in,
                                        width=self.hparams.img_width,
                                        num_blocks=2,
                                        act_fn=nn.SiLU,
                                        use_coordconv=use_coordconv)
        if self.hparams.action_size > 0 and self.hparams.mi_reg_weight > 0:
            self.action_mi_estimator = nn.Sequential(
                nn.Linear(self.hparams.action_size + self.hparams.num_latents, self.hparams.c_hid),
                nn.SiLU(),
                nn.Linear(self.hparams.c_hid, self.hparams.c_hid),
                nn.SiLU(),
                nn.Linear(self.hparams.c_hid, 1)
            )
            self.action_mi_estimator_copy = deepcopy(self.action_mi_estimator)
            for p in self.action_mi_estimator_copy.parameters():
                p.requires_grad = False
        else:
            self.action_mi_estimator = None
            self.action_mi_estimator_copy = None
        
        if self.hparams.latent_mi_reg_weight > 0:
            self.latent_nce_loss = nn.BCEWithLogitsLoss()
            self.latent_nce_classifier = nn.Sequential(
                nn.Linear(self.hparams.num_latents * 2, self.hparams.c_hid),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hparams.c_hid, self.hparams.c_hid),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hparams.c_hid, 1)
            )
        if use_infonce_loss:
            self.context_network = nn.GRU(self.hparams.num_latents, self.hparams.c_hid_ctx, batch_first=True)
            gru_c_hid = self.context_network.hidden_size
            if use_scoring_network:
                self.scoring_network = nn.Sequential(
                    nn.Linear(self.hparams.num_latents + self.hparams.c_hid_ctx, self.hparams.c_hid),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(self.hparams.c_hid, self.hparams.c_hid),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(self.hparams.c_hid, 1)
                )
            else:
                if self.hparams.num_cpc_steps == 1:
                    self.W_1 = nn.Linear(self.hparams.c_hid_ctx, self.hparams.num_latents)
                else:
                    raise NotImplementedError('Only 1-step CPC is implemented')


    def forward(self, x, actions=None, return_z=False, return_identical=False):
        z = self.encoder(x)
        # Adding noise to latent encodings preventing potential latent space collapse
        z_samp = z + torch.randn_like(z) * self.hparams.noise_level
        if actions is not None and self.hparams.action_size > 0:
            z_samp = torch.cat([z_samp, actions], dim=-1)
        if self.reconstructive:
            x_rec = self.decoder(z_samp)
            if return_z:
                return x_rec, z
            else:
                return x_rec
        else:
            if not return_identical:
                return z_samp
            else:
                # Find which images x in the batch are identical based on x, not z
                raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train', weighted_loss=(0.2, 0.8)):
        # Trained by standard MSE loss if reconstructive
        if self.hparams.whole_episode_contrastive:
            (imgs, frame_positions), actions = batch, None
            self.batch_nums, self.episode_len = imgs.shape[0], imgs.shape[1]
            # Images are of shape (batch, episode_len, C, H, W) 
            imgs = imgs.view(-1, *imgs.shape[2:])
            # Adjust frame positions to match the image view
            frame_positions = frame_positions.view(-1, *frame_positions.shape[2:])
        elif self.hparams.triplet_contrastive:
            # batch is [B, *A_img_shape], [B, *P_img_shape], [B, *N_img_shape]
            (anchor_frames, anchor_positions), (positive_frame, positive_position), (negative_frame, negative_position) = batch
            # reshape to [3B, img_shape]
            imgs = torch.cat([anchor_frames, positive_frame, negative_frame], dim=0).squeeze()
            frame_positions = torch.cat([anchor_positions, positive_position, negative_position], dim=0).squeeze()
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                imgs, actions = batch
        else:
            imgs, actions = batch, None
        
        if self.reconstructive:
            x_rec, z = self.forward(imgs, actions=actions, return_z=True)
            # Check if all channels are close to 0 (black pixels)
            is_black = torch.all(imgs <= 1e-6, dim=1, keepdim=True)

            weights = torch.where(is_black, weighted_loss[0] * torch.ones_like(imgs), weighted_loss[1] * torch.ones_like(imgs))

            loss_rec = torch.mean(weights * (x_rec - imgs) ** 2)
            self.log(f'{mode}_loss_rec', loss_rec)
            with torch.no_grad():
                self.log(f'{mode}_loss_rec_mse', F.mse_loss(x_rec, imgs))
                self.log(f'{mode}_loss_rec_abs', torch.abs(x_rec - imgs).mean())
            
                noncompressed_rec = F.mse_loss(x_rec, imgs, reduction='none')
                self.log(f'{mode}_loss_rec_max', noncompressed_rec.max())
                self.log(f'{mode}_loss_rec_smaller_01', (noncompressed_rec < 0.1).float().mean())
                self.log(f'{mode}_loss_rec_smaller_001', (noncompressed_rec < 0.01).float().mean())
                self.log(f'{mode}_loss_rec_smaller_0001', (noncompressed_rec < 0.001).float().mean())
        else:
            loss_rec = 0.0
            z = self.forward(imgs, actions=None)
        if self.hparams.whole_episode_contrastive:
            if self.hparams.use_infonce_loss:
                z = z.view(self.batch_nums, self.episode_len, -1)
                c = self.context_network(z)[0]
                loss_infonce = self._get_infonce_loss(z, c, self.hparams.num_negative_samples)
                self.log(f'{mode}_loss_infonce', loss_infonce)
                loss_contrastive = loss_infonce * self.hparams.infonce_loss_weight
            else:
                z = z.view(self.batch_nums, self.episode_len, -1)
                frame_positions = frame_positions.view(self.batch_nums, self.episode_len, -1)
                loss_contrastive = self._get_contrastive_loss(z, frame_positions)
                self.log(f'{mode}_loss_contrastive', loss_contrastive)
        elif self.hparams.triplet_contrastive:
            # z is of shape [3B, num_latents]
            z_anc, z_pos, z_neg = z.chunk(3, dim=0)
            loss_contrastive = self._get_triplet_loss(z_anc, z_pos, z_neg, self.similarity, self.hparams.margin)

        loss_reg = (z ** 2).mean()
        self.log(f'{mode}_loss_reg', loss_reg)

        if self.hparams.cov_reg_weight > 0:
            loss_cov_reg = self.covariance_regularizer(z)
            self.log(f'{mode}_loss_cov_reg', loss_cov_reg)
            self.log(f'{mode}_loss_reg_weighted', loss_reg * self.hparams.regularizer_weight)
        else:
            loss_cov_reg = 0.0

        loss = loss_rec * self.reconstructive \
            + loss_reg * self.hparams.regularizer_weight \
            + loss_cov_reg * self.hparams.cov_reg_weight \
            + loss_contrastive * self.contrastive

        if self.action_mi_estimator is not None and mode == 'train':
            # Mutual information regularization
            loss_mi_reg_model, loss_mi_reg_latents = self._get_mi_reg_loss(z, actions)
            loss = loss + (loss_mi_reg_model + loss_mi_reg_latents) * self.hparams.mi_reg_weight
            self.log(f'{mode}_loss_mi_reg_model', loss_mi_reg_model)
            self.log(f'{mode}_loss_mi_reg_latents', loss_mi_reg_latents)
            self.log(f'{mode}_loss_mi_reg_latents_weighted', loss_mi_reg_latents * self.hparams.mi_reg_weight)

        if self.hparams.latent_mi_reg_weight > 0:
            nce_loss = self._get_nce_loss(z)
            self.log(f'{mode}_nce_loss', nce_loss)
            self.log(f'{mode}_nce_loss_weighted', nce_loss * self.hparams.latent_mi_reg_weight)
            loss += nce_loss * self.hparams.latent_mi_reg_weight
        
        if self.hparams.latents_pwhsic_reg_weight > 0:
            hsic_module = RbfHSIC(self.estimate_sigma(z), algorithm='biased')
            hsic_loss = self.pairwise_hsic_loss(z, hsic_module)
            self.log(f'{mode}_hsic_loss', hsic_loss)
            self.log(f'{mode}_hsic_loss_weighted', hsic_loss * self.hparams.latents_pwhsic_reg_weight)
            loss += hsic_loss * self.hparams.latents_pwhsic_reg_weight
        
        return loss

    def _get_mi_reg_loss(self, z, actions):
        # Mutual information regularization
        z = z + torch.randn_like(z) * self.hparams.noise_level
        true_inp = torch.cat([z, actions], dim=-1)
        perm = torch.randperm(z.shape[0], device=z.device)
        fake_inp = torch.cat([z[perm], actions], dim=-1)
        inp = torch.stack([true_inp, fake_inp], dim=1).flatten(0, 1)
        model_out = self.action_mi_estimator(inp.detach()).reshape(z.shape[0], 2)
        model_loss = -F.log_softmax(model_out, dim=1)[:,0].mean()
        model_acc = (model_out[:,0] > model_out[:,1]).float().mean()
        self.log('train_mi_reg_model_acc', model_acc)

        for p1, p2 in zip(self.action_mi_estimator.parameters(), self.action_mi_estimator_copy.parameters()):
            p2.data.copy_(p1.data)
        latents_out = self.action_mi_estimator_copy(inp).reshape(z.shape[0], 2)
        latents_loss = -F.log_softmax(latents_out, dim=1).mean()

        return model_loss, latents_loss

    def _get_nce_loss(self, z):
        batch_size, latent_dim = z.size()
        
        real_pairs = torch.cat([z, z.detach()], dim=1)
        
        # shuffle z and concatenate
        shuffled_idxs = torch.randperm(batch_size)
        noise_pairs = torch.cat([z, z[shuffled_idxs].detach()], dim=1)
        
        labels_real = torch.ones(batch_size, 1, device=z.device)
        labels_noise = torch.zeros(batch_size, 1, device=z.device)
        
        preds_real = self.latent_nce_classifier(real_pairs)
        preds_noise = self.latent_nce_classifier(noise_pairs)
        
        loss_real = self.latent_nce_loss(preds_real, labels_real)
        loss_noise = self.latent_nce_loss(preds_noise, labels_noise)
        nce_loss = (loss_real + loss_noise) / 2
        return nce_loss
    
    # def _get_infonce_loss(self, z, c, num_negative_samples):
    #     batch_size, seq_len, _ = z.shape

    #     z_t_plus_1 = z[:, 1:].reshape(batch_size * (seq_len - 1), -1)
    #     # identical_indices = self.find_identical_indices(z_t_plus_1, similarity_threshold=0.99)
    #     c_t = c[:, :-1].reshape(batch_size * (seq_len - 1), -1)

    #     if self.hparams.use_scoring_network:
    #         positive_pairs = self.scoring_network(torch.cat([z_t_plus_1, c_t], dim=1))
    #     else:
    #         # Positive pairs are exp(z_{t+k}^T @ W_k @ c_t), k=1
    #         transformed_c = self.W_1(c_t) # [B * (seq_len - 1), num_latents]
    #         positive_pairs = torch.exp(torch.sum(z_t_plus_1 * transformed_c, dim=-1, keepdim=True))


    #     # negative_samples = torch.randint(0, batch_size * (seq_len - 1), (num_negative_samples,), device=z.device)
    #     exclude_indices = exclude_indices = torch.arange(batch_size * (seq_len - 1), device=z.device)
    #     negative_sample_indices = self.get_negative_samples(batch_size, seq_len, num_negative_samples, z.device,
    #                                                    exclude_indices, mode=self.negative_sampling_mode,
    #                                                    temporal_bias_strength=self.temporal_bias_strength,
    #                                                    identical_indices=[set() for _ in range(z_t_plus_1.shape[0])])

    #     if self.hparams.use_scoring_network:
    #         z_expanded = z_t_plus_1.unsqueeze(1).expand(-1, num_negative_samples, -1).reshape(-1, z_t_plus_1.size(-1))
    #         c_repeated = c_t[negative_sample_indices].reshape(batch_size * (seq_len - 1) * num_negative_samples, -1)
    #         z_c_pairs = torch.cat([z_expanded, c_repeated], dim=1)
    #         negative_pairs = self.scoring_network(z_c_pairs)
    #         negative_pairs = negative_pairs.reshape(batch_size * (seq_len - 1), num_negative_samples)
    #     else:
    #         negative_pairs = torch.exp(torch.sum(z_t_plus_1.unsqueeze(1) * self.W_1(c_t[negative_sample_indices]), dim=-1))

    #     logits = torch.cat([positive_pairs, negative_pairs], dim=1)

    #     labels = torch.zeros(batch_size * (seq_len - 1), dtype=torch.long, device=z.device)
    #     loss = F.cross_entropy(logits, labels)

    #     return loss


    def _get_infonce_loss(self, z, c, num_negative_samples):
        batch_size, seq_len, _ = z.shape
        z_t_plus_1 = z[:, 1:].reshape(batch_size * (seq_len - 1), -1)
        c_t = c[:, :-1].reshape(batch_size * (seq_len - 1), -1)

        identical_mask = self.find_identical_indices(z_t_plus_1, similarity_threshold=1-1e-8)

        if self.hparams.use_scoring_network:
            positive_pairs = self.scoring_network(torch.cat([z_t_plus_1, c_t], dim=1))
        else:
            # Positive pairs are exp(z_{t+k}^T @ W_k @ c_t), k=1
            transformed_c = self.W_1(c_t)  # [B * (seq_len - 1), num_latents]
            positive_pairs = torch.exp(torch.sum(z_t_plus_1 * transformed_c, dim=-1, keepdim=True))

        exclude_indices = torch.arange(batch_size * (seq_len - 1), device=z.device)
        negative_sample_indices = self.get_negative_samples(z, batch_size, seq_len, num_negative_samples, z.device, exclude_indices, identical_mask, mode=self.negative_sampling_mode, temporal_bias_strength=self.temporal_bias_strength)

        if self.hparams.use_scoring_network:
            c_repeated = c_t.unsqueeze(1).expand(-1, num_negative_samples, -1).reshape(-1, c_t.size(-1))
            z_expanded = z.reshape(-1, z.size(-1))[negative_sample_indices]
            z_c_pairs = torch.cat([z_expanded, c_repeated], dim=1)
            negative_pairs = self.scoring_network(z_c_pairs)
            negative_pairs = negative_pairs.reshape(batch_size * (seq_len - 1), num_negative_samples)
        else:
            negative_pairs = torch.exp(torch.sum(z.reshape(-1, z.size(-1))[negative_sample_indices] * self.W_1(c_t).unsqueeze(1), dim=-1))

        logits = torch.cat([positive_pairs, negative_pairs], dim=1)
        labels = torch.zeros(batch_size * (seq_len - 1), dtype=torch.long, device=z.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def find_identical_indices(self, z_t, similarity_threshold=0.99):
        # Normalize z_t to unit vectors
        norms = torch.norm(z_t, p=2, dim=1, keepdim=True)
        normalized_z_t = z_t / norms

        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(normalized_z_t, normalized_z_t.transpose(0, 1))

        # Create a mask for identical indices (excluding self-similarity)
        identical_mask = (similarity_matrix > similarity_threshold) & (~torch.eye(z_t.shape[0], dtype=torch.bool, device=z_t.device))

        return identical_mask

    def get_negative_samples(self, z, batch_size, seq_len, num_negative_samples, device, exclude_indices, identical_mask, mode='random', temporal_bias_strength=1.0):
        """ Generate negative samples with various strategies, excluding identical contexts. """
        total_samples = batch_size * seq_len
        negative_samples = torch.zeros((batch_size * (seq_len - 1), num_negative_samples), dtype=torch.long, device=device)

        for i in range(batch_size * (seq_len - 1)):
            if mode == 'random':
                z_t_plus_1_indices = torch.arange(batch_size * (seq_len - 1), device=device)
                available_indices = z_t_plus_1_indices[~identical_mask[i]]
                available_indices = available_indices[available_indices != i]
                if available_indices.size(0) < num_negative_samples:
                    chosen_negatives = available_indices.repeat(num_negative_samples // available_indices.size(0) + 1)[:num_negative_samples]
                else:
                    chosen_negatives = available_indices[torch.randperm(available_indices.size(0))[:num_negative_samples]]
            elif mode == 'same_episode':
                episode_size = seq_len - 1
                episode_start = (i // episode_size) * episode_size
                episode_end = episode_start + episode_size
                available_indices = torch.arange(episode_start, episode_end, device=device)[~identical_mask[i, episode_start:episode_end]]
                available_indices = available_indices[available_indices != i % episode_size]
                possible_indices = available_indices.size(0)
                chosen_negatives = available_indices[torch.randperm(possible_indices)[:num_negative_samples]]
                if possible_indices < num_negative_samples:
                    chosen_negatives = chosen_negatives.repeat(num_negative_samples // possible_indices + 1)[:num_negative_samples]
            elif mode == 'temporal_distance':
                episode_size = seq_len - 1
                episode_start = (i // episode_size) * episode_size
                episode_end = episode_start + episode_size
                distances = torch.abs(torch.arange(episode_start, episode_end, device=device) - (i % episode_size))
                weights = 1 / (distances + temporal_bias_strength)
                weights[identical_mask[i, episode_start:episode_end]] = 0  # Exclude identical encodings
                weights[i % episode_size] = 0  # Exclude the positive context
                possible_indices = weights.nonzero().sum()
                available_indices = torch.arange(episode_start, episode_end, device=device)
                chosen_negatives = available_indices[torch.multinomial(weights, num_negative_samples, replacement=(possible_indices < num_negative_samples).item())]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            negative_samples[i] = chosen_negatives

        return negative_samples

    # def get_negative_samples(self, batch_size, seq_len, num_negative_samples, device, exclude_indices, identical_indices, mode='random', temporal_bias_strength=1.0):
    #     """
    #     Generate negative samples with various strategies, excluding identical contexts.
        
    #     Args:
    #     - batch_size (int): Number of episodes in the batch.
    #     - seq_len (int): Episode length.
    #     - num_negative_samples (int): Number of negative samples to generate.
    #     - device (torch.device): Device to place the generated indices.
    #     - exclude_indices (torch.Tensor): Indices of the positive samples to exclude.
    #     - identical_indices (list of sets): For each index in the batch, a set of indices that are considered identical and should be excluded from being negative samples.
    #     - mode (str): Sampling mode ('random', 'same_episode', 'temporal_distance').
    #     - temporal_bias_strength (float): Strength of the bias towards temporally closer samples (higher means stronger bias).
        
    #     Returns:
    #     - torch.Tensor: Indices of the negative samples.
    #     """
    #     total_samples = batch_size * (seq_len - 1)
    #     negative_samples = torch.zeros((total_samples, num_negative_samples), dtype=torch.long, device=device)
        
    #     for i in range(total_samples):
    #         if mode == 'random':
    #             available_indices = list(set(range(total_samples)) - {exclude_indices[i].item()} - identical_indices[i])
    #             chosen_negatives = torch.tensor(np.random.choice(available_indices, num_negative_samples, replace=False), device=device)
    #         elif mode == 'same_episode':
    #             episode_size = seq_len - 1
    #             episode_start = (i // episode_size) * episode_size
    #             available_indices = set(range(episode_start, episode_start + episode_size)) - {exclude_indices[i].item()} - identical_indices[i]
    #             chosen_negatives = torch.tensor(np.random.choice(list(available_indices), num_negative_samples, replace=False), device=device)
    #         elif mode == 'temporal_distance':
    #             episode_size = seq_len - 1
    #             episode_start = (i // episode_size) * episode_size
    #             distances = np.abs(np.arange(episode_start, episode_start + episode_size) - i)
    #             weights = 1 / (distances + temporal_bias_strength)  # Add bias
    #             if identical_indices[i]:
    #                 weights[list(identical_indices[i])] = 0  # Exclude identical encodings
    #             weights[exclude_indices[i] % episode_size] = 0  # Exclude the positive context
    #             available_indices = np.arange(episode_start, episode_start + episode_size)
    #             available_weights = weights / weights.sum()
    #             chosen_negatives = torch.tensor(np.random.choice(available_indices, num_negative_samples, replace=False, p=available_weights), device=device)
    #         else:
    #             raise ValueError(f"Unknown mode: {mode}")
            
    #         negative_samples[i] = chosen_negatives
        
    #     return negative_samples

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='val')
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='val')
        self.log('test_loss', loss)

    @staticmethod
    def get_callbacks(exmp_inputs=None, cluster=False, **kwargs):
        if kwargs.get('reconstructive', False):
            img_callback = AELogCallback(exmp_inputs, every_n_epochs=1)
        if kwargs.get('triplet_contrastive', False) or kwargs.get('whole_episode_contrastive', False):
            img_callback = CombinedVisualizationCallback(
                every_n_epochs=1, perplexity=30, n_neighbors=100, min_dist=0.1, max_samples=2000, dataloader=kwargs['dataloader'])
        lr_callback = LearningRateMonitor('step')
        return [lr_callback, img_callback]
    
    def covariance_regularizer(self, z):
        """
        Computes the covariance regularization term for the latent variables.
        
        Parameters:
        - z: Latent variables (batch_size x num_latents)
        
        Returns:
        - cov_reg: The covariance regularization term
        """
        z_centered = z - z.mean(dim=0)
        
        cov_matrix = (z_centered.T @ z_centered) / (z_centered.size(0) - 1)
        
        identity = torch.eye(z.shape[1], device=z.device)
        cov_diff = cov_matrix - identity
        
        cov_reg = (cov_diff ** 2).sum()

        return cov_reg
    
    def pairwise_hsic_loss(self, latent_matrix, hsic_module):
        n_latents = latent_matrix.size(1)
        hsic_loss = 0.0
        for i in range(n_latents):
            for j in range(i + 1, n_latents):
                hsic_loss += hsic_module(latent_matrix[:, i:i+1], latent_matrix[:, j:j+1])
        return hsic_loss

    def estimate_sigma(self, Z, norm=float('inf')):
        if Z.dim() > 2:
            Z = Z.view(Z.size(0), -1)
        dists = torch.cdist(Z, Z, p=norm)
        sigma = torch.median(dists[dists.nonzero(as_tuple=True)].view(-1))
        return sigma

    def _get_triplet_loss(self, z_anc, z_pos, z_neg, similarity_function, margin=1.0):
        """
        Compute triplet loss.
        
        Parameters:
        - z_anc: Tensor - Latent representations of anchor samples.
        - z_pos: Tensor - Latent representations of positive samples.
        - z_neg: Tensor - Latent representations of negative samples.
        - similarity_function: SimilarityFunction - A callable similarity function.
        - margin: float - The margin for triplet loss.

        Returns:
        - loss: The computed loss.
        """
        positive_similarity = similarity_function(z_anc, z_pos)
        negative_similarity = similarity_function(z_anc, z_neg)
        losses = F.relu(margin - positive_similarity + negative_similarity)
        return losses.mean()

    def _get_contrastive_loss(self, z, timesteps):
        """
        Computes the temporal contrastive loss for each episode in the batch
        with random selection of anchor frame and comparing it with all other frames
        in the same episode.

        Parameters:
        - z: Tensor representing the latent space encodings reshaped to the original batch size and episode length.
        - timesteps: Tensor representing the relative timesteps within each episode.

        Returns:
        - The average contrastive loss across the batch of episodes.
        """
        batch_size, episode_len, _ = z.size()
        losses = []

        for i in range(batch_size):
            # Randomly select anchor
            anchor_idx = torch.randint(0, episode_len, (1,)).item()
            anchor_z = z[i, anchor_idx].unsqueeze(0)

            # Exclude the anchor frame and get all other frames as positives
            positive_idxs = [idx for idx in range(episode_len) if idx != anchor_idx]
            positive_zs = z[i, positive_idxs]

            # Calculate pairwise cosine similarity between the anchor and all positives
            similarities = F.cosine_similarity(anchor_z, positive_zs)
            # Rescale the cosine similarities to range (0, 2)
            # similarities = (similarities + 1) / 2.0

            # Calculate the temporal weights based on the relative positions of the frames
            anchor_time = timesteps[i, anchor_idx].unsqueeze(0)
            positive_times = timesteps[i, positive_idxs]
            temporal_distances = torch.abs(anchor_time - positive_times)
            weights = F.softmax(-temporal_distances, dim=0)

            # Calculate the contrastive loss
            weighted_similarities = similarities * weights
            loss = -torch.log(weighted_similarities.sum() / (episode_len - 1))
            losses.append(loss)

        # Return the mean loss across the batch
        return torch.stack(losses).mean()

class HSIC(nn.Module):
    """Base class for the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
    ..math:: HSIC (X, Y) := || C_{x, y} ||^2_{HS}, where HSIC (X, Y) = 0 iif X and Y are independent.

    Empirically, we use the finite sample estimator of HSIC (with m observations) by,
    (1) biased estimator (HSIC_0)
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        :math: (m - 1)^2 tr KHLH.
        where K_{ij} = kernel_x (x_i, x_j), L_{ij} = kernel_y (y_i, y_j), H = 1 - m^{-1} 1 1 (Hence, K, L, H are m by m matrices).
    (2) unbiased estimator (HSIC_1)
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        :math: \frac{1}{m (m - 3)} \bigg[ tr (\tilde K \tilde L) + \frac{1^\top \tilde K 1 1^\top \tilde L 1}{(m-1)(m-2)} - \frac{2}{m-2} 1^\top \tilde K \tilde L 1 \bigg].
        where \tilde K and \tilde L are related to K and L by the diagonal entries of \tilde K_{ij} and \tilde L_{ij} are set to zero.

    Parameters
    ----------
    sigma_x : float
        the kernel size of the kernel function for X.
    sigma_y : float
        the kernel size of the kernel function for Y.
    algorithm: str ('unbiased' / 'biased')
        the algorithm for the finite sample estimator. 'unbiased' is used for our paper.
    reduction: not used (for compatibility with other losses).
    """
    def __init__(self, sigma_x, sigma_y=None, algorithm='unbiased',
                 reduction=None):
        super(HSIC, self).__init__()

        if sigma_y is None:
            sigma_y = sigma_x

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        if algorithm == 'biased':
            self.estimator = self.biased_estimator
        elif algorithm == 'unbiased':
            self.estimator = self.unbiased_estimator
        else:
            raise ValueError('invalid estimator: {}'.format(algorithm))

    def _kernel_x(self, X):
        raise NotImplementedError

    def _kernel_y(self, Y):
        raise NotImplementedError

    def biased_estimator(self, input1, input2):
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        K = self._kernel_x(input1)
        L = self._kernel_y(input2)

        KH = K - K.mean(0, keepdim=True)
        LH = L - L.mean(0, keepdim=True)

        N = len(input1)

        return torch.trace(KH @ LH / (N - 1) ** 2)

    def unbiased_estimator(self, input1, input2):
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        kernel_XX = self._kernel_x(input1)
        kernel_YY = self._kernel_y(input2)

        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)

        N = len(input1)

        hsic = (
            torch.trace(tK @ tL)
            + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )

        return hsic / (N * (N - 3))

    def forward(self, input1, input2, **kwargs):
        return self.estimator(input1, input2)


class RbfHSIC(HSIC):
    """Radial Basis Function (RBF) kernel HSIC implementation.
    """
    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    def _kernel_x(self, X):
        return self._kernel(X, self.sigma_x)

    def _kernel_y(self, Y):
        return self._kernel(Y, self.sigma_y)

class SimilarityFunction(nn.Module):
    def forward(self, z_anc, z_pos_neg):
        raise NotImplementedError

class EuclideanSimilarity(SimilarityFunction):
    def forward(self, z_anc, z_pos_neg):
        return -torch.norm(z_anc.unsqueeze(1) - z_pos_neg.unsqueeze(0), dim=2, p=2)

class CosineSimilarity(SimilarityFunction):
    def forward(self, z_anc, z_pos_neg):
        z_anc_normalized = F.normalize(z_anc, p=2, dim=1)
        z_pos_neg_normalized = F.normalize(z_pos_neg, p=2, dim=1)
        return torch.mm(z_anc_normalized, z_pos_neg_normalized.transpose(0, 1))


class AELogCallback(pl.Callback):
    """ Callback for visualizing predictions """

    def __init__(self, exmp_inputs, every_n_epochs=5, prefix=''):
        super().__init__()
        if isinstance(exmp_inputs, (tuple, list)):
            self.imgs, self.actions = exmp_inputs
        else:
            self.imgs, self.actions = exmp_inputs, None
        self.every_n_epochs = every_n_epochs
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module):
        def log_fig(tag, fig):
            if isinstance(trainer.logger, WandbLogger):
                fig = wandb.Image(fig)
                trainer.logger.experiment.log({f'{self.prefix}{tag}': fig}, step=trainer.global_step)
            else:
                trainer.logger.experiment.add_image(f'{self.prefix}{tag}', fig, global_step=trainer.global_step, dataformats='HWC')

        if self.imgs is not None and (trainer.current_epoch+1) % self.every_n_epochs == 0:
            images = self.imgs.to(trainer.model.device)
            trainer.model.eval()
            rand_idxs = np.random.permutation(images.shape[0])
            if self.actions is None or pl_module.hparams.action_size <= 0:
                actions = None
            else:
                actions = self.actions.to(trainer.model.device)
            log_fig(f'reconstruction_seq', visualize_ae_reconstruction(trainer.model, images[:8], 
                                                                       actions[:8] if actions is not None else None))
            log_fig(f'reconstruction_rand', visualize_ae_reconstruction(trainer.model, images[rand_idxs[:8]], 
                                                                        actions[rand_idxs[:8]] if actions is not None else None))
            trainer.model.train()
           
           

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import numpy as np
import torch
import wandb
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
 
class CombinedVisualizationCallback(Callback):
    def __init__(self, every_n_epochs=1, perplexity=30, n_neighbors=30, min_dist=0.1, max_samples=500, dataloader=None, use_triplets=False):
        """
        Initializes the combined visualization callback.

        Args:
            every_n_epochs (int): Frequency of epochs at which to generate visualizations.
            perplexity (int): The perplexity parameter for t-SNE.
            n_neighbors (int): The size of local neighborhood used for manifold approximation in UMAP.
            min_dist (float): The minimum distance apart that points are allowed to be in the low-dimensional representation in UMAP.
            max_samples (int): Maximum number of samples to include in the visualizations.
            dataloader (DataLoader): The dataloader from which to fetch the data.
            use_triplets (bool): Whether to handle data as triplets or as whole episodes.
        """
        self.every_n_epochs = every_n_epochs
        self.perplexity = perplexity
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.max_samples = max_samples
        self.dataloader = dataloader
        self.use_triplets = use_triplets

    def on_validation_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            pl_module.eval()
            embeddings, temporal_positions = [], []
            sample_count = 0

            for batch in self.dataloader:
                if sample_count >= self.max_samples:
                    break

                if self.use_triplets:
                    # Handling data as triplets
                    ((anchor_frames, anchor_positions), (positive_frames, positive_positions), _) = batch
                    concatenated_frames = torch.cat([anchor_frames, positive_frames], dim=0).to(pl_module.device).squeeze()
                    z = pl_module(concatenated_frames)
                    embeddings.append(z.cpu().detach())
                    temporal_positions.extend(anchor_positions.cpu().numpy())
                    temporal_positions.extend(positive_positions.cpu().numpy())
                    sample_count += len(anchor_frames) * 2
                else:
                    # Handling whole episodes
                    imgs, frame_positions = batch
                    imgs = imgs.view(-1, *imgs.shape[2:])
                    frame_positions = frame_positions.view(-1)
                    if sample_count + imgs.shape[0] > self.max_samples:
                        limit = self.max_samples - sample_count
                        imgs = imgs[:limit]
                        frame_positions = frame_positions[:limit]
                    z = pl_module(imgs.to(pl_module.device)).cpu().detach()
                    embeddings.append(z)
                    temporal_positions.extend(frame_positions.cpu().numpy())
                    sample_count += imgs.shape[0]

            if sample_count == 0:
                return  # Exit if no data was processed

            embeddings = torch.cat(embeddings, dim=0)[:self.max_samples].numpy()
            temporal_positions = np.array(temporal_positions)[:self.max_samples]

            # t-SNE Visualization
            z_tsne = TSNE(n_components=2, perplexity=self.perplexity).fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            scatter_tsne = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=temporal_positions, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter_tsne, label='Frame Position within Episode')
            plt.title('t-SNE Visualization')
            plt.xlabel('t-SNE dimension 1')
            plt.ylabel('t-SNE dimension 2')
            if isinstance(trainer.logger, WandbLogger):
                wandb.log({"t-SNE Visualization": [wandb.Image(plt)]})
            plt.close()

            # UMAP Visualization
            reducer = umap.UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, n_components=2)
            z_umap = reducer.fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            scatter_umap = plt.scatter(z_umap[:, 0], z_umap[:, 1], c=temporal_positions, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter_umap, label='Frame Position within Episode')
            plt.title('UMAP Visualization')
            plt.xlabel('UMAP dimension 1')
            plt.ylabel('UMAP dimension 2')
            if isinstance(trainer.logger, WandbLogger):
                wandb.log({"UMAP Visualization": [wandb.Image(plt)]})
            plt.close()

            pl_module.train()