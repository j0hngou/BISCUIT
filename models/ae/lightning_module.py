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


    def forward(self, x, actions=None, return_z=False):
        z = self.encoder(x)
        # Adding noise to latent encodings preventing potential latent space collapse
        z_samp = z + torch.randn_like(z) * self.hparams.noise_level
        if actions is not None and self.hparams.action_size > 0:
            z_samp = torch.cat([z_samp, actions], dim=-1)
        x_rec = self.decoder(z_samp)
        if return_z:
            return x_rec, z
        else:
            return x_rec 

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train', weighted_loss=(0.2, 0.8)):
        # Trained by standard MSE loss
        if self.hparams.whole_episode_contrastive:
            imgs, frame_positions, actions = batch, None
            self.batch_nums, self.episode_len = imgs.shape[0], imgs.shape[1]
            # Images are of shape (batch, episode_len, C, H, W) 
            imgs = imgs.view(-1, *imgs.shape[2:])
            # Adjust frame positions to match the image view
            frame_positions = frame_positions.view(-1, *frame_positions.shape[2:])
        elif:
            isinstance(batch, (tuple, list)) and len(batch) == 2:
                imgs, actions = batch
        else:
            imgs, actions = batch, None
        x_rec, z = self.forward(imgs, actions=actions, return_z=True)
        # Check if all channels are close to 0 (black pixels)
        is_black = torch.all(imgs <= 1e-6, dim=1, keepdim=True)

        weights = torch.where(is_black, weighted_loss[0] * torch.ones_like(imgs), weighted_loss[1] * torch.ones_like(imgs))

        loss_rec = torch.mean(weights * (x_rec - imgs) ** 2)
        # loss_rec = F.mse_loss(x_rec, imgs)
        loss_reg = (z ** 2).mean()
        loss_cov_reg = self.covariance_regularizer(z)
        self.log(f'{mode}_loss_rec', loss_rec)
        self.log(f'{mode}_loss_reg', loss_reg)
        self.log(f'{mode}_loss_cov_reg', loss_cov_reg)
        self.log(f'{mode}_loss_reg_weighted', loss_reg * self.hparams.regularizer_weight)
        
        with torch.no_grad():
            self.log(f'{mode}_loss_rec_mse', F.mse_loss(x_rec, imgs))
            self.log(f'{mode}_loss_rec_abs', torch.abs(x_rec - imgs).mean())
        
            noncompressed_rec = F.mse_loss(x_rec, imgs, reduction='none')
            self.log(f'{mode}_loss_rec_max', noncompressed_rec.max())
            self.log(f'{mode}_loss_rec_smaller_01', (noncompressed_rec < 0.1).float().mean())
            self.log(f'{mode}_loss_rec_smaller_001', (noncompressed_rec < 0.01).float().mean())
            self.log(f'{mode}_loss_rec_smaller_0001', (noncompressed_rec < 0.001).float().mean())
        loss = loss_rec + loss_reg * self.hparams.regularizer_weight + loss_cov_reg * self.hparams.cov_reg_weight

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
        img_callback = AELogCallback(exmp_inputs, every_n_epochs=1)
        # Create learning rate callback
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