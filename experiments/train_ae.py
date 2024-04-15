"""
Run file to train an autoencoder.
"""

import argparse
import torch.utils.data as data
import pytorch_lightning as pl

import sys
sys.path.append('../')
from models.ae import Autoencoder
from experiments.datasets import VoronoiDataset, CausalWorldDataset, iTHORDataset, GridworldDataset
from experiments.utils import train_model, print_params
import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--c_hid', type=int, default=64)
    parser.add_argument('--num_latents', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--noise_level', type=float, default=0.05)
    parser.add_argument('--regularizer_weight', type=float, default=0.0)
    parser.add_argument('--mi_reg_weight', type=float, default=0.0)
    parser.add_argument('--logger_name', type=str, default='')
    parser.add_argument('--files_to_save', type=str, nargs='+', default='')
    parser.add_argument('--subsample_percentage', type=float, default=1.0)
    parser.add_argument('--use_coordconv', action='store_true')
    parser.add_argument('--cov_reg_weight', type=float, default=0.0)
    parser.add_argument('--latent_mi_reg_weight', type=float, default=0.0)
    parser.add_argument('--latents_pwhsic_reg_weight', type=float, default=0.0)
    parser.add_argument('--reconstructive', action='store_true', default=False) # if False, use predictive
    parser.add_argument('--triplet_contrastive', action='store_true', default=False)
    parser.add_argument('--whole_episode_contrastive', action='store_true', default=False)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--use_infonce_loss', action='store_true', default=False)
    parser.add_argument('--num_negative_samples', type=int, default=10)
    parser.add_argument('--use_scoring_network', action='store_true', default=False)
    parser.add_argument('--c_hid_ctx', type=int, default=128)
    parser.add_argument('--num_cpc_steps', type=int, default=1)
    parser.add_argument('--negative_sampling_mode', type=str, default='temporal_distance')
    parser.add_argument('--temporal_bias_strength', type=float, default=1.0)
    
    args = parser.parse_args()
    assert args.reconstructive or args.triplet_contrastive or args.whole_episode_contrastive, 'Must specify at least one of reconstructive, triplet_contrastive, or whole_episode_contrastive'
    if (args.triplet_contrastive or args.whole_episode_contrastive):
        dataset_args = {'triplet' : args.triplet_contrastive, 'return_whole_episode' : args.whole_episode_contrastive}
    pl.seed_everything(args.seed)

    print('Loading datasets...')
    if 'voronoi' in args.data_dir:
        DataClass = VoronoiDataset
    elif 'causal_world' in args.data_dir:
        DataClass = CausalWorldDataset
    elif 'ithor' in args.data_dir:
        DataClass = iTHORDataset
    elif 'gridworld' in args.data_dir:
        DataClass = GridworldDataset
    else:
        assert False, 'Unknown dataset'
    
    train_dataset = DataClass(
        data_folder=args.data_dir, split='train', single_image=True, seq_len=1, cluster=args.cluster, subsample_percentage=args.subsample_percentage, **dataset_args)
    val_dataset = DataClass(
        data_folder=args.data_dir, split='val', single_image=True, seq_len=1, cluster=args.cluster, subsample_percentage=args.subsample_percentage, **dataset_args)
    test_dataset = DataClass(
        data_folder=args.data_dir, split='test_indep', single_image=True, seq_len=1,
        causal_vars=train_dataset.target_names(), cluster=args.cluster, subsample_percentage=args.subsample_percentage, **dataset_args)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)
    print(f'Length training dataset: {len(train_dataset)} / Train loader: {len(train_loader)}')
    print(f'Length val dataset: {len(val_dataset)} / Test loader: {len(val_loader)}')

    args.max_iters = args.max_epochs * len(train_loader)
    model_args = vars(args)
    model_args['img_width'] = train_dataset.get_img_width()
    if hasattr(train_dataset, 'get_inp_channels'):
        model_args['c_in'] = train_dataset.get_inp_channels()
    print(f'Image size: {model_args["img_width"]}')
    model_class = Autoencoder
    data_dir_name = args.data_dir.split('/')[-1]
    contrastive_str = 'recon' if args.reconstructive else 'triplet' if args.triplet_contrastive else 'whole_episode' if args.whole_episode_contrastive else ''
    if contrastive_str == 'whole_episode':
        contrastive_str += f'_num_neg_{args.num_negative_samples}_num_cpc_steps_{args.num_cpc_steps}_neg_sampling_mode_{args.negative_sampling_mode}_temporal_bias_strength_{args.temporal_bias_strength}'
    logger_name = f'AE_{args.num_latents}l_{args.c_hid}hid_{data_dir_name}_' + contrastive_str
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name
    callback_kwargs = {
        'reconstructive' : args.reconstructive,
        'triplet_contrastive' : args.triplet_contrastive,
        'whole_episode_contrastive' : args.whole_episode_contrastive,
        'dataloader' : val_loader,
    }
    

    print_params(logger_name, model_args)
    
    w = wandb.init(project='gridworld-biscuit', name=logger_name, config=model_args)
    
    logger = pl.loggers.WandbLogger(name=logger_name, offline=args.offline, id=w.id)

    train_model(model_class=model_class,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                progress_bar_refresh_rate=0 if args.cluster else 1,
                logger_name=logger_name,
                wandb_logger=logger,
                check_val_every_n_epoch=min(1, args.max_epochs),
                gradient_clip_val=0.1,
                action_size=train_dataset.action_size() if DataClass == CausalWorldDataset else -1,
                callback_kwargs=callback_kwargs,
                **model_args)