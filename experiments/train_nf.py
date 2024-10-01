"""
Run file for training BISCUIT-NF on a pretrained autoencoder architecture.
"""

import os
import torch
import torch.utils.data as data
import sys
# print current dir
print(os.getcwd())
sys.path.append('../')
from models.biscuit_nf import BISCUITNF
from experiments.utils import train_model, load_datasets, get_default_parser, print_params
import wandb
from pytorch_lightning.loggers import WandbLogger


def encode_dataset(model, datasets, args):
    if isinstance(datasets, data.Dataset):
        datasets = [datasets]
    if any([isinstance(d, dict) for d in datasets]):
        new_datasets = []
        for d in datasets:
            if isinstance(d, dict):
                new_datasets += list(d.values())
            else:
                new_datasets.append(d)
        datasets = new_datasets
    for dataset in datasets:
        if dataset.encodings_active:
            continue
        autoencoder_folder = model.hparams.autoencoder_checkpoint.rsplit('/', 1)[0]
        encoding_folder = os.path.join(autoencoder_folder, 'encodings/')
        os.makedirs(encoding_folder, exist_ok=True)
        encoding_filename = os.path.join(encoding_folder, f'{model.hparams.data_folder}_{dataset.split_name}.pt')
        if not os.path.exists(encoding_filename):
            encodings = dataset.encode_dataset(lambda batch: model.autoencoder.encoder(batch.to(model.device)).cpu())
            torch.save(encodings, encoding_filename)
        else:
            if 'gridworld' in args.data_dir:
                dataset.load_encodings(encoding_filename, lambda batch: batch)
            else:
                dataset.load_encodings(encoding_filename)

if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True)
    parser.add_argument('--c_hid', type=int, default=64)
    parser.add_argument('--num_flows', type=int, default=6)
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--flow_act_fn', type=str, default='silu')
    parser.add_argument('--hidden_per_var', type=int, default=16)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--num_latents', type=int, default=40)
    parser.add_argument('--prior_action_add_prev_state', action="store_true")
    parser.add_argument('--try_encodings', type=bool, default=True)
    parser.add_argument('--logit_reg_factor', type=float, default=0.004)
    parser.add_argument('--text', default=False, action="store_true")
    parser.add_argument('--lr_text', type=float, default=1e-4, help='Learning rate for text model')
    parser.add_argument('--wandb', default=True, action="store_true")
    parser.add_argument('--text_encoder', type=str, default='sentence_transformer', help='Which text encoder to use')
    parser.add_argument('--subsample_percentage', type=float, default=1.0)
    parser.add_argument('--subsample_chunk', type=int, default=1)
    parser.add_argument('--debug_data', default=False, action="store_true")
    parser.add_argument('--perfect_intv', default=False, action="store_true")
    parser.add_argument('--text_only', default=False, action="store_true")
    parser.add_argument('--stop_grad', default=False, action="store_true")
    parser.add_argument('--noise_level', type=float, default=-1)
    parser.add_argument('--pass_gt_causals', default=False, action="store_true")
    parser.add_argument('--transform_gt', default=False, action="store_true")
    parser.add_argument('--scale_latents', type=float, default=1.0)
    parser.add_argument('--flow_init_std_factor', type=float, default=0.2)
    parser.add_argument('--max_steps', type=int, default=-1, help='Maximum number of training steps')

    args = parser.parse_args()
    model_args = vars(args)
    if not args.text and args.text_only:
        raise ValueError("Cannot have text_only=True without text=True")

    datasets, data_loaders, data_name = load_datasets(args)

    # Calculate max_epochs if max_steps is specified
    if args.max_steps > 0:
        model_args['max_iters'] = args.max_steps
        steps_per_epoch = len(data_loaders['train'])
        model_args['max_epochs'] = (args.max_steps + steps_per_epoch - 1) // steps_per_epoch
    else:
        model_args['max_epochs'] = args.max_epochs
        model_args['max_iters'] = args.max_epochs * len(data_loaders['train'])

    model_args['data_folder'] = [s for s in args.data_dir.split('/') if len(s) > 0][-1]
    model_args['width'] = datasets['train'].get_img_width()
    model_args['img_width'] = datasets['train'].get_img_width()
    if hasattr(datasets['train'], 'action_size'):
        model_args['action_size'] = datasets['train'].action_size()
    model_args['text'] = args.text
    model_args['text_only'] = args.text_only
    model_args['stop_grad'] = args.stop_grad
    model_args['noise_level'] = args.noise_level
    model_args['pass_gt_causals'] = args.pass_gt_causals
    model_args['flow_init_std_factor'] = args.flow_init_std_factor
    model_class = BISCUITNF
    textornot = 'text' if args.text else 'notext'
    textornot += '_textonly' if args.text_only else ''
    logger_name = f'BISCUITNF_{args.num_latents}l_{datasets["train"].num_vars()}b_{args.c_hid}hid_{data_name}_{textornot}_ss{args.subsample_percentage}_sc{args.subsample_chunk}_{"perfect_intv" if args.perfect_intv else "est_intv"}_numsamples{args.num_samples}_noiselvl{args.noise_level}_passgtcausals{args.pass_gt_causals}_transformgtcausals{args.transform_gt}_scalelatents{args.scale_latents}_frozenflowinitstd{args.flow_init_std_factor}'
    args_logger_name = model_args.pop('logger_name', "")
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name

    callback_kwargs = {
        'dataset': datasets['train'], 
        'correlation_dataset': datasets['val'],
        'correlation_test_dataset': datasets['test'],
        'next_step_dataset': datasets['val_seq'],
    }
    if model_class == BISCUITNF and 'action' in data_loaders:
        callback_kwargs['action_data_loader'] = data_loaders['action']

    print_params(logger_name, model_args)

    if args.wandb:
        run = wandb.init(project="BISCUIT", name=logger_name, config=model_args, dir='/scratch-shared/gkounto/wandb')
        wandb.config.update(args)
        wandb.config.update(model_args)
        wandb.config.update({'data_name': data_name})
        logger = WandbLogger(name=logger_name, project="BISCUIT", config=model_args, experiment=run)
        logger.log_hyperparams(model_args)
    else:
        logger = None

    check_val_every_n_epoch = model_args.pop('check_val_every_n_epoch', -1)
    if check_val_every_n_epoch <= 0:
        check_val_every_n_epoch = 5 if not args.cluster else 10

    train_model(
        model_class=model_class,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val_seq'],
        test_loader=data_loaders['test_seq'],
        logger_name=logger_name,
        check_val_every_n_epoch=check_val_every_n_epoch,
        progress_bar_refresh_rate=0 if args.cluster else 1,
        callback_kwargs=callback_kwargs,
        var_names=datasets['train'].target_names(),
        op_before_running=lambda model: encode_dataset(model, list(datasets.values()), args),
        save_last_model=True,
        cluster_logging=args.cluster,
        causal_var_info=datasets['train'].get_causal_var_info(),
        wandb_logger=logger,
        **model_args
    )
