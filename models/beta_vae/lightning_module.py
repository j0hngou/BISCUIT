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


class BetaVAE(pl.LightningModule):
    def __init__(self, num_latents, c_in=3, c_hid=64, lr=1e-3, warmup=500, 
                 max_iters=100000, img_width=64, beta=10.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Select the correct Encoder and Decoder based on image width
        if self.hparams.img_width == 32:
            EncoderClass = SimpleEncoder
            DecoderClass = SimpleDecoder
        else:
            EncoderClass = Encoder
            DecoderClass = Decoder
        use_coordconv = kwargs.get('use_coordconv', False)

        # Initialize variational encoder and decoder
        self.encoder = EncoderClass(num_latents=self.hparams.num_latents,
                                    c_hid=self.hparams.c_hid,
                                    c_in=self.hparams.c_in,
                                    width=self.hparams.img_width,
                                    act_fn=nn.SiLU,
                                    residual=True,
                                    num_blocks=2,
                                    variational=True,
                                    use_coordconv=use_coordconv)
        self.decoder = DecoderClass(num_latents=self.hparams.num_latents,
                                    c_hid=self.hparams.c_hid,
                                    c_out=self.hparams.c_in,
                                    width=self.hparams.img_width,
                                    num_blocks=2,
                                    act_fn=nn.SiLU,
                                    use_coordconv=use_coordconv)

    def forward(self, x):
        z_mean, z_log_std = self.encoder(x)
        z_sample = z_mean + torch.randn_like(z_mean) * z_log_std.exp()
        x_rec = self.decoder(z_sample)
        return x_rec, z_sample, z_mean, z_log_std

    def encode(self, x, random=True):
        z_mean, z_log_std = self.encoder(x)
        if random:
            z_sample = z_mean + torch.randn_like(z_mean) * z_log_std.exp()
        else:
            z_sample = z_mean
        return z_sample

    def _get_loss(self, batch, mode='train'):
        imgs, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        x_rec, z_sample, z_mean, z_log_std = self.forward(imgs)
        loss_rec = F.mse_loss(x_rec, imgs)
        kld = kl_divergence(z_mean, z_log_std).sum(dim=1).mean()
        loss = loss_rec + self.hparams.beta * kld
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_kld', kld)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

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
        img_callback = VAELogCallback(exmp_inputs, every_n_epochs=1)
        # Create learning rate callback
        lr_callback = LearningRateMonitor('step')
        return [lr_callback, img_callback]

def kl_divergence(mean1, log_std1, mean2=None, log_std2=None):
    """ Returns the KL divergence between two Gaussian distributions """
    if mean2 is None:
        mean2 = torch.zeros_like(mean1)
    if log_std2 is None:
        log_std2 = torch.zeros_like(log_std1)
    var1, var2 = (2*log_std1).exp(), (2*log_std2).exp()
    KLD = (log_std2 - log_std1) + (var1 + (mean1 - mean2) ** 2) / (2 * var2) - 0.5
    return KLD.sum(dim=1)

class VAELogCallback(pl.Callback):
    """ Callback for visualizing predictions """

    def __init__(self, exmp_inputs, every_n_epochs=3, prefix=''):
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
            log_fig(f'reconstruction_seq', visualize_vae_reconstruction(trainer.model, images[:8]))
            log_fig(f'reconstruction_rand', visualize_vae_reconstruction(trainer.model, images[rand_idxs[:8]]))
            trainer.model.train()