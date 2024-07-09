import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim import AdamW

import sys
sys.path.append('../')
from models.shared import get_act_fn, ImageLogCallback, NextStepCallback
from models.ae import Autoencoder
from models.biscuit_vae import BISCUITVAE
from models.shared import AutoregNormalizingFlow
from models.shared import InteractionVisualizationCallback, PermutationCorrelationMetricsLogCallback, CosineWarmupScheduler


class BISCUITNF(BISCUITVAE):
    """ 
    The main module implementing BISCUIT-NF.
    It is a subclass of BISCUIT-VAE to inherit several functionality.
    """

    def __init__(self, *args,
                        autoencoder_checkpoint=None,
                        num_flows=4,
                        hidden_per_var=16,
                        num_samples=8,
                        flow_act_fn='silu',
                        noise_level=-1,
                        **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs - see BISCUIT-VAE for the full list
        autoencoder_checkpoint : str
                                 Path to the checkpoint of the autoencoder
                                 which should be used for training the flow
                                 on.
        num_flows : int
                    Number of flow layers to use
        hidden_per_var : int
                         Hidden dimensionality per latent variable to use
                         in the autoregressive networks.
        num_samples : int
                      Number of samples to take from an input encoding
                      during training. Larger sample sizes give smoother
                      gradients.
        flow_act_fn : str
                      Activation function to use in the networks of the flow
        noise_level : float
                      Standard deviation of the added noise to the encodings.
                      If smaller than zero, the std of the autoencoder is used.
        """
        kwargs['no_encoder_decoder'] = True
        super().__init__(*args, **kwargs)
        self.text = kwargs.get('text', False)
        self.stop_grad = kwargs.get('stop_grad', False)
        self.scale_latents = kwargs.get('scale_latents', False)
        self.flow_init_std_factor = kwargs.get('flow_init_std_factor', 0.2)
        # Initialize the flow
        self.flow = AutoregNormalizingFlow(self.hparams.num_latents, 
                                           self.hparams.num_flows,
                                           act_fn=get_act_fn(self.hparams.flow_act_fn),
                                           hidden_per_var=self.hparams.hidden_per_var)
        self.pass_gt_causals = kwargs.get('pass_gt_causals', False)
        if self.pass_gt_causals:
            self.frozen_flow = AutoregNormalizingFlow(self.hparams.num_latents, 
                                           self.hparams.num_flows,
                                           act_fn=get_act_fn(self.hparams.flow_act_fn),
                                           hidden_per_var=self.hparams.hidden_per_var,
                                           init_std_factor=1.0).eval()
            for p in self.frozen_flow.parameters():
                p.requires_grad_(False)
        # Setup autoencoder
        if self.hparams.autoencoder_checkpoint is not None:
            self.autoencoder = Autoencoder.load_from_checkpoint(self.hparams.autoencoder_checkpoint)
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)
            
            if self.hparams.noise_level < 0.0:
                self.hparams.noise_level = self.autoencoder.hparams.noise_level
        else:
            self.autoencoder = None
            self.hparams.noise_level = 0.0

    def encode(self, x, random=True):
        # Map input to disentangled latents, e.g. for correlation metrics
        if random:
            x = x + torch.randn_like(x) * self.hparams.noise_level
        z, _ = self.flow(x)
        return z

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
            # latent distribution for the input but not for the target
            x_enc = x_enc[...,None,:].expand(-1, -1, self.hparams.num_samples, -1) * self.hparams.scale_latents
            batch_size, seq_len, num_samples, num_latents = x_enc.shape
            # x_sample = x_enc + torch.randn_like(x_enc) * self.hparams.noise_level
            x_sample = torch.cat([x_enc[:, 0:1, :, :] + torch.randn_like(x_enc[:, 0:1, :, :]) * self.hparams.noise_level,
                                  x_enc[:, 1:, :, :]], dim=1)
            if self.stop_grad:
                x_t = x_sample[:, 0:1, :, :].detach()
                x_t1 = x_sample[:, 1:, :, :]
                x_t_flat = x_t.flatten(0, 2)
                x_t1_flat = x_t1.flatten(0, 2)
            else:
                x_sample = x_sample.flatten(0, 2)
        if self.text:
            tokenized_description = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            tokenized_description = None
        # Execute the flow
        if self.stop_grad:
            z_t1, ldj_t1 = self.flow(x_t1_flat)
            z_t1 = z_t1.unflatten(0, (batch_size, seq_len-1, num_samples))
            ldj_t1 = ldj_t1.reshape(batch_size, seq_len-1, num_samples)
            z_t, ldj_t = self.flow(x_t_flat.detach())
            z_t = z_t.unflatten(0, (batch_size, 1, num_samples))
            ldj_t = ldj_t.reshape(batch_size, 1, num_samples)
            z_sample = torch.cat([z_t, z_t1], dim=1)
            ldj = torch.cat([ldj_t, ldj_t1], dim=1)
        else:
            z_sample, ldj = self.flow(x_sample)
            z_sample = z_sample.unflatten(0, (batch_size, seq_len, num_samples))
            ldj = ldj.reshape(batch_size, seq_len, num_samples)
        # Calculate the negative log likelihood of the transition prior
        nll = self.prior_t1.sample_based_nll(z_t=z_sample[:,:-1].flatten(0, 1),
                                             z_t1=z_sample[:,1:].flatten(0, 1),
                                             tokenized_description=tokenized_description,
                                             intv_targets=intv_targets if len(batch) == 3 else None,
                                             action=action.flatten(0, 1))
        # Add LDJ and prior NLL for full loss
        ldj = ldj[:,1:].flatten(0, 1).mean(dim=-1)  # Taking the mean over samples
        loss = nll + ldj
        loss = (loss * (seq_len - 1)).mean()

        # Logging
        self.log(f'{mode}_nll', nll.mean())
        self.log(f'{mode}_ldj', ldj.mean())

        return loss

    def configure_optimizers(self):
        """ Setup the optimizer """
        if self.text:
            prior_text_params = list(self.prior_t1.text_MLP.parameters())
            prior_t1_params = list(self.prior_t1.parameters())
            flow_params = list(self.flow.parameters())

            all_params = prior_t1_params + flow_params

            # if self.hparams.prior_action_add_prev_state:
            #     action_MLP_params = list(self.prior_t1.action_MLP.parameters())
            #     all_params += action_MLP_params

            prior_text_params_set = set(prior_text_params)
            all_params_set = set(all_params)
            rest_params = list(all_params_set - prior_text_params_set)

            optimizer = AdamW([{'params': prior_text_params, 'lr': self.hparams.lr_text, 'weight_decay': 0.01},
                            {'params': rest_params, 'lr': self.hparams.lr}],
                            lr=self.hparams.lr, weight_decay=0.0)

        else:
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                            warmup=self.hparams.warmup,
                                            max_iters=self.hparams.max_iters)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    @staticmethod
    def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, next_step_dataset=False, correlation_dataset=False, correlation_test_dataset=None, action_data_loader=None, **kwargs):
        img_callback = ImageLogCallback([None, None], dataset, every_n_epochs=10, cluster=cluster)
        # Create learning rate callback
        lr_callback = LearningRateMonitor('step')
        next_step_callback = NextStepCallback(dataset=next_step_dataset, every_n_epochs=1)
        next_step_callback_train = NextStepCallback(dataset=dataset, every_n_epochs=1, split_name='train')
        callbacks = [lr_callback, img_callback, next_step_callback, next_step_callback_train]
        corr_callback = PermutationCorrelationMetricsLogCallback(correlation_dataset, 
                                                                 cluster=cluster, 
                                                                 test_dataset=correlation_test_dataset)
        callbacks.append(corr_callback)
        if action_data_loader is not None:
            actionvq_callback = InteractionVisualizationCallback(action_data_loader=action_data_loader)
            callbacks.append(actionvq_callback)
        return callbacks