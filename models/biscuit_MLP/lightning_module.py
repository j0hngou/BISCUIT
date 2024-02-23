from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from models.shared import get_act_fn, ImageLogCallback, NextStepCallback
from models.shared import CosineWarmupScheduler, InteractionVisualizationCallback, PermutationCorrelationMetricsLogCallback, create_interaction_prior
from models.biscuit_vae import BISCUITVAE
from models.ae import Autoencoder
from torch.optim import AdamW
from copy import deepcopy

class ResBlock(nn.Module):
    """A simple Residual Block with batch normalization"""
    def __init__(self, in_features, out_features, act_fn='silu'):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.act_fn = get_act_fn(act_fn)()
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act_fn(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity
        return self.act_fn(out)

class MLP(nn.Module):
    """MLP with ResNet blocks"""
    def __init__(self, input_dim, output_dim, num_blocks=4, hidden_dim=128, act_fn='silu'):
        super().__init__()
        layers = [ResBlock(input_dim if i == 0 else hidden_dim, hidden_dim, act_fn=act_fn) for i in range(num_blocks)]
        self.blocks = nn.Sequential(*layers)
        self.final_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.blocks(x)
        return self.final_fc(x)

class BISCUITMLP(BISCUITVAE):
    """
    BISCUITMLP is an adaptation of the BISCUIT architecture using MLP encoders
    instead of Normalizing Flows for latent space transformations.
    """

    def __init__(self, *args, autoencoder_checkpoint=None, num_samples=2,
                 noise_level=-1, text=False,
                 text_encoder='sentence_transformer', lr_text=1e-4,
                 num_blocks=8, act_fn='silu', **kwargs):
        """
        Initializes the BISCUITMLP model with MLP-based architecture.
        """
        kwargs['use_flow_prior'] = False
        super().__init__(*args, **kwargs)

        # Basic settings
        self.text = text
        self.text_only = kwargs.get('text_only', False)
        self.num_samples = num_samples

        # Initialize the input and target MLP encoders
        self.input_encoder = MLP(self.hparams.num_latents, self.hparams.num_latents,
                                 num_blocks=num_blocks, act_fn=act_fn)
        self.target_encoder = deepcopy(self.input_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        # Setup autoencoder if a checkpoint is provided
        if autoencoder_checkpoint is not None:
            self.autoencoder = Autoencoder.load_from_checkpoint(autoencoder_checkpoint)
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)
            if noise_level < 0.0:
                self.hparams.noise_level = self.autoencoder.hparams.noise_level
        else:
            self.autoencoder = None
            self.hparams.noise_level = noise_level if noise_level >= 0.0 else 0.0

        # Update learning rate for text encoder if specified
        if self.text:
            self.hparams.lr_text = lr_text

    def encode(self, x, random=True):
        # Map input to disentangled latents, e.g. for correlation metrics
        if random:
            x = x + torch.randn_like(x) * self.hparams.noise_level
        z = self.input_encoder(x)
        return z

    def forward(self, x):
        """
        Forward pass through the model, encoding inputs and targets with respective MLPs.
        """
        z_input = self.input_encoder(x)

        with torch.no_grad():
            z_target = self.target_encoder(x)

        return z_input, z_target

    def momentum_update_target_encoder(self, momentum=0.999):
        """
        Updates the target encoder's weights as a moving average of the input encoder's weights.
        """
        with torch.no_grad():
            for param_q, param_k in zip(self.input_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1.0 - momentum) * param_q.data

    def _get_loss(self, batch, mode='train'):
        """ Main training method for calculating the loss """
        if len(batch) == 2:
            x_enc, action = batch
        elif len(batch) == 3:
            x_enc, action, intv_targets = batch
        elif len(batch) == 5:
            x_enc, action, input_ids, token_type_ids, attention_mask = batch
        else:
            x_enc, _, action = batch
        with torch.no_grad():
            # Expand encodings over samples and add noise to 'sample' from the autoencoder
            # latent distribution
            x_enc = x_enc[...,None,:].expand(-1, -1, self.hparams.num_samples, -1)
            batch_size, seq_len, num_samples, num_latents = x_enc.shape
            x_sample = x_enc + torch.randn_like(x_enc) * self.hparams.noise_level
            x_t = x_sample[:, 0:1, :, :]
            x_t1 = x_sample[:, 1:, :, :]
            x_t_flat = x_t.flatten(0, 2)
            x_t1_flat = x_t1.flatten(0, 2)
        if self.text:
            tokenized_description = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            tokenized_description = None
        # Execute the flow
        z_t1 = self.target_encoder(x_t1_flat).detach()
        z_t1 = z_t1.unflatten(0, (batch_size, seq_len-1, num_samples))
        z_t = self.input_encoder(x_t_flat)
        z_t = z_t.unflatten(0, (batch_size, 1, num_samples))
        z_sample = torch.cat([z_t, z_t1], dim=1)
        # Calculate the negative log likelihood of the transition prior
        nll = self.prior_t1.sample_based_nll(z_t=z_sample[:,:-1].flatten(0, 1),
                                             z_t1=z_sample[:,1:].flatten(0, 1),
                                             tokenized_description=tokenized_description,
                                             intv_targets=intv_targets if len(batch) == 3 else None,
                                             action=action.flatten(0, 1))
        # NLL is the full loss
        loss = nll
        loss = (loss * (seq_len - 1)).mean()

        # Logging
        self.log(f'{mode}_nll', nll.mean())

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler for training.
        """
        if self.text:
            # Optimizer setup for text processing
            optimizer = AdamW([{'params': self.input_encoder.parameters(), 'lr': self.hparams.lr},
                               {'params': self.target_encoder.parameters(), 'lr': self.hparams.lr_text}],
                              lr=self.hparams.lr, weight_decay=0.01)
        else:
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)

        lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    @staticmethod
    def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, next_step_dataset=False, correlation_dataset=False, correlation_test_dataset=None, action_data_loader=None, **kwargs):
        img_callback = ImageLogCallback([None, None], dataset, every_n_epochs=10, cluster=cluster)
        # Create learning rate callback
        lr_callback = LearningRateMonitor('step')
        callbacks = [lr_callback, img_callback]
        corr_callback = PermutationCorrelationMetricsLogCallback(correlation_dataset, 
                                                                 cluster=cluster, 
                                                                 test_dataset=correlation_test_dataset)
        callbacks.append(corr_callback)
        if action_data_loader is not None:
            actionvq_callback = InteractionVisualizationCallback(action_data_loader=action_data_loader)
            callbacks.append(actionvq_callback)
        return callbacks