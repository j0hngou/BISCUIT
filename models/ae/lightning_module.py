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
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            imgs, actions = batch
        else:
            imgs, actions = batch, None
        x_rec, z = self.forward(imgs, actions=actions, return_z=True)
        # Check for which pixels all channels are close to 0 (black pixels)
        is_black = torch.all(imgs <= 1e-6, dim=1, keepdim=True)

        weights = torch.where(is_black, weighted_loss[0] * torch.ones_like(imgs), weighted_loss[1] * torch.ones_like(imgs))

        loss_rec = torch.mean(weights * (x_rec - imgs) ** 2)
        # loss_rec = F.mse_loss(x_rec, imgs)
        loss_reg = (z ** 2).mean()
        
        corr_matrix_pearson = self._pearson_correlation(z)
        corr_matrix_spearman = self._spearman_correlation(z)
        
        off_diagonal_mask = ~torch.eye(corr_matrix_pearson.size(0), dtype=bool, device=corr_matrix_pearson.device)
        corr_loss_pearson = corr_matrix_pearson.abs()[off_diagonal_mask].mean()
        corr_loss_spearman = corr_matrix_spearman.abs()[off_diagonal_mask].mean()

        corr_reg_weight = self.hparams.get('corr_reg_weight', 0.0)
        hsic_reg_weight = self.hparams.get('hsic_reg_weight', 0.0)
        
        hsic_loss = self.hsic_loss(z)
        
        self.log(f'{mode}_loss_rec', loss_rec)
        self.log(f'{mode}_loss_reg', loss_reg)
        self.log(f'{mode}_loss_reg_weighted', loss_reg * self.hparams.regularizer_weight)
        self.log(f'{mode}_loss_corr_pearson', corr_loss_pearson)
        self.log(f'{mode}_loss_corr_spearman', corr_loss_spearman)
        self.log(f'{mode}_loss_corr_pearson_weighted', corr_loss_pearson * corr_reg_weight)
        self.log(f'{mode}_loss_corr_spearman_weighted', corr_loss_spearman * corr_reg_weight)
        self.log(f'{mode}_loss_hsic', hsic_loss)
        self.log(f'{mode}_loss_hsic_weighted', hsic_loss * hsic_reg_weight)
        with torch.no_grad():
            self.log(f'{mode}_loss_rec_mse', F.mse_loss(x_rec, imgs))
            self.log(f'{mode}_loss_rec_abs', torch.abs(x_rec - imgs).mean())
        
            noncompressed_rec = F.mse_loss(x_rec, imgs, reduction='none')
            self.log(f'{mode}_loss_rec_max', noncompressed_rec.max())
            self.log(f'{mode}_loss_rec_smaller_01', (noncompressed_rec < 0.1).float().mean())
            self.log(f'{mode}_loss_rec_smaller_001', (noncompressed_rec < 0.01).float().mean())
            self.log(f'{mode}_loss_rec_smaller_0001', (noncompressed_rec < 0.001).float().mean())
        loss = loss_rec + loss_reg * self.hparams.regularizer_weight + corr_loss_pearson * corr_reg_weight + corr_loss_spearman * corr_reg_weight + hsic_loss * hsic_reg_weight

        if self.action_mi_estimator is not None and mode == 'train':
            # Mutual information regularization
            loss_mi_reg_model, loss_mi_reg_latents = self._get_mi_reg_loss(z, actions)
            loss = loss + (loss_mi_reg_model + loss_mi_reg_latents) * self.hparams.mi_reg_weight
            self.log(f'{mode}_loss_mi_reg_model', loss_mi_reg_model)
            self.log(f'{mode}_loss_mi_reg_latents', loss_mi_reg_latents)
            self.log(f'{mode}_loss_mi_reg_latents_weighted', loss_mi_reg_latents * self.hparams.mi_reg_weight)
        return loss
    
    def _pearson_correlation(self, x):
        """
        Compute Pearson correlation matrix for batch of latent variables.
        """
        x = x - x.mean(dim=0)
        cov = x.T @ x / (x.shape[0] - 1)
        std = x.std(dim=0, unbiased=True)
        corr_matrix = cov / torch.outer(std, std)
        return corr_matrix

    def _spearman_correlation(self, x):
        """
        Compute Spearman's rank correlation matrix for batch of latent variables.
        """
        rank_x = x.argsort().argsort().to(torch.float32)
        return self._pearson_correlation(rank_x)
    

    def rbf_kernel(X, sigma=None):
        """
        Computes the RBF (Gaussian) kernel matrix of X.
        """
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        R = X_sqnorms.unsqueeze(1) - 2*XX + X_sqnorms.unsqueeze(0)
        if sigma is None:
            sigma = torch.median(R[R > 0])
        K = torch.exp(-R / (2 * sigma ** 2))
        return K

    def hsic_loss(Z, sigma=None):
        """
        Computes the HSIC value as a regularization term for the latent representations Z.
        """
        n = Z.size(0)
        K = rbf_kernel(Z, sigma=sigma)
        H = torch.eye(n) - 1.0/n * torch.ones((n, n))
        H = H.to(Z.device)

        HSIC = torch.trace(K @ H @ K @ H)
        
        HSIC /= (n-1) ** 2
        
        return HSIC


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