"""
PyTorch dataset classes for loading all datasets.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import json
import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm
from glob import glob
import sys
sys.path.append('../')
from data_generation.data_generation_ithor import create_targets
from random import sample


class VoronoiDataset(data.Dataset):

    VAR_INFO = OrderedDict({
        'c0': 'continuous_2.8',
        'c1': 'continuous_2.8',
        'c2': 'continuous_2.8',
        'c3': 'continuous_2.8',
        'c4': 'continuous_2.8',
        'c5': 'continuous_2.8',
        'c6': 'continuous_2.8',
        'c7': 'continuous_2.8',
        'c8': 'continuous_2.8'
    })

    def __init__(self, data_folder='../data/voronoi/', split='train', single_image=False, return_latents=False, triplet=False, seq_len=2, causal_vars=None, return_robot_state=False, return_targets=True, **kwargs):
        super().__init__()
        filename = split
        if triplet:
            filename += '_triplets'
        self.split_name = filename
        data_file = os.path.join(data_folder, f'{filename}.npz')
        if split.startswith('val') and not os.path.isfile(data_file):
            self.split_name = self.split_name.replace('val', 'test')
            print('[!] WARNING: Could not find a validation dataset. Falling back to the standard test set...')
            data_file = os.path.join(data_folder, f'{filename.replace("val", "test")}.npz')
        assert os.path.isfile(data_file), f'Could not find VoronoiDataset dataset at {data_file}'

        arr = np.load(data_file)
        self.imgs = torch.from_numpy(arr['images'])
        self.latents = torch.from_numpy(arr['latents'])
        self.targets = torch.from_numpy(arr['targets'])
        if 'actions' in arr:
            self.actions = torch.from_numpy(arr['actions'])
            self.actions = (self.actions - 0.5) / 0.75
        else:
            self.actions = torch.zeros(*self.latents.shape[:-1], 2)
            if return_robot_state and split == 'train':
                print(f'[!] WARNING: No actions found in {self.split_name} dataset despite return_robot_state=True. Returning zeros instead.')
        self.keys = [key.replace('_', '-') for key in arr['keys'].tolist()]
        self._load_settings(data_folder)
        self._clean_up_data(causal_vars)

        self.return_latents = return_latents
        self.return_robot_state = return_robot_state
        self.return_targets = return_targets
        self.triplet = triplet
        self.single_image = single_image
        self.encodings_active = False
        self.seq_len = seq_len if not (single_image or triplet) else 1

    def _clean_up_data(self, causal_vars=None):
        if len(self.imgs.shape) == 5:
            self.imgs = self.imgs.permute(0, 1, 4, 2, 3)  # Push channels to PyTorch dimension
        else:
            self.imgs = self.imgs.permute(0, 3, 1, 2)

        all_latents, all_targets = [], []
        target_names = [] if causal_vars is None else causal_vars
        keys_var_info = list(VoronoiDataset.VAR_INFO.keys())
        for key in keys_var_info:
            if key not in self.keys:
                VoronoiDataset.VAR_INFO.pop(key)
        for i, key in enumerate(self.keys):
            latent = self.latents[...,i]
            target = self.targets[...,i]
            if causal_vars is not None:
                if key in causal_vars:
                    all_targets.append(target)
            elif target.sum() > 0:
                all_targets.append(target)
                target_names.append(key)
            all_latents.append(latent)
        self.latents = torch.stack(all_latents, dim=-1)
        self.targets = torch.stack(all_targets, dim=-1)
        self.target_names_l = target_names
        print(f'Using the causal variables {self.target_names_l}')

    def _load_settings(self, data_folder):
        self.temporal_adj_matrix = None
        self.adj_matrix = None
        self.settings = {}
        filename = os.path.join(data_folder, 'settings.json')
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                self.settings = json.load(f)
            self.adj_matrix = torch.Tensor(self.settings['causal_graph'])
            if 'temporal_causal_graph' in self.settings:
                self.temporal_adj_matrix = torch.Tensor(self.settings['temporal_causal_graph'])

    @torch.no_grad()
    def encode_dataset(self, encoder, batch_size=512):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        encoder.eval()
        encoder.to(device)
        encodings = None
        for idx in tqdm(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
            batch = self.imgs[idx:idx+batch_size].to(device)
            batch = self._prepare_imgs(batch)
            if len(batch.shape) == 5:
                batch = batch.flatten(0, 1)
            batch = encoder(batch)
            if len(self.imgs.shape) == 5:
                batch = batch.unflatten(0, (-1, self.imgs.shape[1]))
            batch = batch.detach().cpu()
            if encodings is None:
                encodings = torch.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=batch.dtype, device='cpu')
            encodings[idx:idx+batch_size] = batch
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    def load_encodings(self, filename):
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            return imgs

    def label_to_img(self, label):
        return (label + 1.0) / 2.0

    def action_size(self):
        return self.actions.shape[-1] if self.return_robot_state else -1

    def num_labels(self):
        return -1

    def num_vars(self):
        return self.targets.shape[-1]

    def target_names(self):
        return self.target_names_l

    def get_img_width(self):
        return self.imgs.shape[-2]

    def get_inp_channels(self):
        return self.imgs.shape[-3]

    def get_causal_var_info(self):
        return VoronoiDataset.VAR_INFO

    def get_adj_matrix(self):
        return self.adj_matrix

    def get_temporal_adj_matrix(self):
        return self.temporal_adj_matrix

    def __len__(self):
        return self.imgs.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        returns = []

        if self.triplet:
            img_pair = self.imgs[idx]
            pos = self.latents[idx]
            target = self.targets[idx]
            rob = self.actions[idx]
        else:
            img_pair = self.imgs[idx:idx+self.seq_len]
            pos = self.latents[idx:idx+self.seq_len]
            target = self.targets[idx:idx+self.seq_len-1]
            rob = self.actions[idx+1:idx+self.seq_len]

        if self.single_image:
            img_pair = img_pair[0]
            pos = pos[0]
            if target.shape[0] > 0:
                target = target[0]
            if rob.shape[0] > 0:
                rob = rob[0]
        img_pair = self._prepare_imgs(img_pair)
        returns = [img_pair] + returns

        if self.return_robot_state:
            returns += [rob]
        if self.return_targets:
            returns += [target]
        if self.return_latents:
            returns += [pos]

        return tuple(returns) if len(returns) > 1 else returns[0]


def flatten_arrays_in_dict(array_dict, start_idx, end_idx):
    for key in array_dict:
        arr = array_dict[key]
        if isinstance(arr, np.ndarray):
            array_dict[key] = np.reshape(arr, arr.shape[:start_idx] + (-1,) + arr.shape[end_idx:])
        elif isinstance(arr, dict):
            flatten_arrays_in_dict(arr, start_idx, end_idx)

def dict_numpy_to_torch(array_dict):
    for key in array_dict:
        arr = array_dict[key]
        if isinstance(arr, np.ndarray):
            array_dict[key] = torch.from_numpy(arr).float()
        elif isinstance(arr, dict):
            dict_numpy_to_torch(arr)

def create_empty_stacked_dict(exmp_dict, stack_size):
    stacked_dict = {}
    for key in exmp_dict:
        if isinstance(exmp_dict[key], np.ndarray):
            stacked_dict[key] = np.zeros((stack_size, *exmp_dict[key].shape),
                                         dtype=exmp_dict[key].dtype)
        elif isinstance(exmp_dict[key], dict):
            stacked_dict[key] = create_empty_stacked_dict(exmp_dict[key], stack_size)
    return stacked_dict

def add_data_to_stacked_dict(data, sub_data, idx):
    for key in sub_data.keys():
        if isinstance(sub_data[key], np.ndarray):
            data[key][idx] = sub_data[key]
        elif isinstance(sub_data[key], dict):
            add_data_to_stacked_dict(data[key], sub_data[key], idx)
        else:
            assert False, f'Unknown data type detected for {key}: {sub_data[key]}'

def tqdm_track(tqdm_iter, cluster=False, **kwargs):
    if cluster:
        return tqdm_iter
    else:
        return tqdm(tqdm_iter, **kwargs)


class CausalWorldDataset(data.Dataset):

    VAR_INFO = OrderedDict({
        'robot0_color': 'continuous_1',
        'robot1_color': 'continuous_1',
        'robot2_color': 'continuous_1',
        'floor_color': 'continuous_1',
        'stage_color': 'continuous_1',
        'object_color': 'continuous_1',
        'object_position_dist': 'continuous_1',
        'object_position_angle': 'angle',
        'object_orientation_angle': 'angle',
        'object_velocity': 'continuous_1'
    })

    def __init__(self, 
                 data_folder, 
                 split='train', 
                 single_image=False, 
                 return_robot_state=True, 
                 return_targets=False, 
                 return_latents=False, 
                 triplet=False,
                 downsample_images=True, 
                 seq_len=2, 
                 cluster=False,
                 **kwargs):
        super().__init__()
        self.cluster = cluster
        filename = split
        if triplet:
            filename += '_triplets'
        self.split_name = filename
        data_split_folder = os.path.join(data_folder, f'{filename}')
        if split.startswith('val') and not os.path.isdir(data_split_folder):
            self.split_name = self.split_name.replace('val', 'test')
            print('[!] WARNING: Could not find a validation dataset. Falling back to the standard test set. Do not use it for selecting the best model!')
            data_split_folder = os.path.join(data_folder, f'{filename.replace("val", "test")}')
        assert os.path.isdir(data_split_folder), f'Could not find CausalWorld dataset at {data_split_folder}'
        data = self.load_data_from_folder(data_split_folder)
        
        self.return_robot_state = return_robot_state
        self.return_targets = return_targets
        self.return_latents = return_latents
        self.triplet = triplet
        self.single_image = single_image
        self.encodings_active = False
        self.seq_len = seq_len if not (single_image or triplet) else 1

        # Preparing images
        if 'optimized_images' in data:
            self.imgs = data.pop('optimized_images')
        else:
            if 'diff_frames' not in data:
                data['diff_frames'] = np.zeros((*data['observations'].shape[:-1], 1), dtype=np.float32)
            else:
                data['diff_frames'] = (data['diff_frames'].astype(np.float32) + 255.) / 2.
            self.imgs = np.concatenate([data.pop('observations'), 
                                        data.pop('diff_frames')], axis=-1)
            self.imgs = np.moveaxis(self.imgs, -1, -3)
            self.imgs = np.reshape(self.imgs, self.imgs.shape[:-4] + (-1,) + self.imgs.shape[-2:])
        if len(self.imgs.shape) > 4 and not triplet:
            arange_0 = np.arange(self.imgs.shape[0], dtype=np.int32)
            arange_1 = np.arange(start=0,
                                 stop=self.imgs.shape[1]-self.seq_len+1,
                                 dtype=np.int32)
            self.indices = arange_1[None,:] + arange_0[:,None] * self.imgs.shape[1]
            self.indices = np.reshape(self.indices, (-1,))
            self.imgs = np.reshape(self.imgs, (-1,) + self.imgs.shape[2:])
            flatten_arrays_in_dict(data, 0, 2)
        else:
            self.indices = np.arange(start=0,
                                     stop=self.imgs.shape[0]-self.seq_len+1,
                                     dtype=np.int32)
        self.imgs = torch.from_numpy(self.imgs)
        dict_numpy_to_torch(data)

        # Resize images if needed
        if downsample_images and self.imgs.shape[-1] == 128:
            if triplet:
                self.imgs = self.imgs.flatten(0, 1)
            self.imgs = F.interpolate(self.imgs.float(), 
                                      scale_factor=(0.5, 0.5),
                                      mode='bilinear')
            if triplet:
                self.imgs = self.imgs.unflatten(0, (-1, 3))

        # Interventions
        if 'interventions' in data:
            self.keys_causal = sorted(list(data['interventions'].keys()))
            print(self.keys_causal)
            self.targets = torch.stack([data['interventions'][key] for key in self.keys_causal], dim=-1)
        elif triplet and 'triplet_masks' in data:
            causal_var_list = data['causal_var_list']
            argsort = [0] + sorted(range(1, len(causal_var_list)), key=causal_var_list.__getitem__)
            self.keys_causal = [causal_var_list[i] for i in argsort]
            self.targets = data['triplet_masks'][:,argsort]
        else:
            self.keys_causal = []
            self.targets = torch.zeros((self.imgs.shape[0], 0))

        # Robotic state
        if 'infos' in data:
            self.robot_state = torch.cat([data['infos']['joint_positions'],
                                          data['infos']['joint_positions']], dim=-1)
            action_size = self.robot_state.shape[-1]
            if not triplet and self.seq_len > 1:
                # Instead of velocities at the recorded frame, we take the previous robot position to have full motions
                self.robot_state[self.indices,action_size//2:] = self.robot_state[self.indices-1,:action_size//2]
        elif 'causal_vars' in data:
            self.robot_state = torch.cat([data['causal_vars']['intervention_dict/joint_positions'],
                                          data['causal_vars']['intervention_dict/joint_positions']], dim=-1)
        else:
            print('Should not happen, we have left in data:', data.keys())
            self.robot_state = torch.zeros(self.imgs.shape[0], 0)

        # Causal vars
        if 'causal_vars' in data:
            causal_vars = data['causal_vars']
            for key in causal_vars:
                if key.endswith('color') and key in CausalWorldDataset.VAR_INFO and CausalWorldDataset.VAR_INFO[key].startswith('continuous'):
                    causal_vars[key] = causal_vars[key] * 2. - 1.
                elif key.endswith('angle'):
                    if causal_vars[key].min() < 0:
                        causal_vars[key] = torch.where(causal_vars[key] < 0, 2*np.pi + causal_vars[key], causal_vars[key])
                elif key == 'object_position_dist':
                    causal_vars[key] = causal_vars[key].clamp(min=0.0, max=0.21) / 0.21 * 2. - 1.
                elif key == 'object_velocity':
                    causal_vars[key] = causal_vars[key].clamp(min=0.0, max=1.0) * 2.0 - 1.0
            self.latent_state = torch.stack([
                    causal_vars[key] for key in CausalWorldDataset.VAR_INFO
                ], dim=-1)
        else:
            self.latent_state = torch.zeros((self.imgs.shape[0], len(CausalWorldDataset.VAR_INFO)))

    def load_data_from_folder(self, data_folder):
        data = {}
        obs_file = os.path.join(data_folder, 'observations.npz')
        opt_file = os.path.join(data_folder, 'optimized_images.npz')
        if not os.path.isfile(obs_file) and not os.path.isfile(opt_file):
            # Try finding all subfolders
            sub_folders = sorted(glob(os.path.join(data_folder, 'seq_*/')))
            assert len(sub_folders) > 0, f'File {obs_file} could not be found and no subfolders were detected'
            data = None
            for i, f in enumerate(tqdm_track(sub_folders, cluster=self.cluster)): # tqdm
                sub_data = self.load_data_from_folder(f)
                if data is None:
                    data = create_empty_stacked_dict(sub_data, stack_size=len(sub_folders))
                add_data_to_stacked_dict(data, sub_data, i)
        else:
            def load_np_file(filename, file_key=None, dict_key=None):
                if not filename.startswith(data_folder):
                    filename = os.path.join(data_folder, filename)
                if not os.path.isfile(filename):
                    return
                if dict_key is None:
                    dict_key = filename.split('/')[-1].split('.')[0]
                if filename.endswith('.npz') or filename.endswith('.npy'):
                    file_data = np.load(filename)
                elif filename.endswith('.json'):
                    with open(filename, 'r') as f:
                        file_data = json.load(f)
                if file_key is not None:
                    data[dict_key] = file_data[file_key]
                elif isinstance(file_data, list):
                    data[dict_key] = file_data
                else:
                    data[dict_key] = {key: file_data[key] for key in file_data}

            load_np_file(opt_file, file_key='imgs')
            if 'optimized_images' not in data:
                load_np_file(obs_file, file_key='obs')
                load_np_file('diff_frames.npz', file_key='diff')
            load_np_file('interventions.npz')
            load_np_file('causal_vars.npz')
            load_np_file('step_infos.npz', dict_key='infos')
            load_np_file('infos.npz')
            load_np_file('triplet_masks.npz', file_key='mask')
            load_np_file('causal_var_list.json')
        return data

    def encode_dataset(self, encode_func, batch_size=512):
        encodings = None
        for idx in tqdm_track(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False, cluster=self.cluster):
            batch = self.imgs[idx:idx+batch_size]
            batch = self._prepare_imgs(batch)
            robot = self.robot_state[idx:idx+batch_size]
            if len(batch.shape) == 5:  # Triplets
                batch = batch.flatten(0, 1)
                robot = robot.flatten(0, 1)
            with torch.no_grad():
                batch = encode_func(batch)
            if len(self.imgs.shape) == 5:
                batch = batch.reshape((-1, self.imgs.shape[1]) + batch.shape[1:])
            if encodings is None:
                encodings = np.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=np.float32)
            encodings[idx:idx+batch_size] = batch
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    def load_encodings(self, filename):
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            return imgs

    def action_size(self):
        return self.robot_state.shape[-1]

    def num_labels(self):
        return -1

    def num_vars(self):
        return self.targets.shape[-1]

    def target_names(self):
        return self.keys_causal

    def get_img_width(self):
        return self.imgs.shape[-1]

    def get_inp_channels(self):
        return self.imgs.shape[-3]

    def get_causal_var_info(self):
        return CausalWorldDataset.VAR_INFO

    def get_adj_matrix(self):
        return np.zeros((self.num_vars, self.num_vars), dtype=np.float32)

    def get_temporal_adj_matrix(self):
        return np.eye(self.num_vars, dtype=np.float32)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        idx = self.indices[idx]
        returns = []

        if self.triplet:
            img_pair = self.imgs[idx]
            rob = self.robot_state[idx]
            lat = self.latent_state[idx]
            target = self.targets[idx]
        else:
            img_pair = self.imgs[idx:idx+self.seq_len]
            rob = self.robot_state[idx+1:idx+self.seq_len]
            lat = self.latent_state[idx:idx+self.seq_len]
            target = self.targets[idx+1:idx+self.seq_len]

        if self.single_image:
            img_pair = img_pair[0]
            lat = lat[0]
            if rob.shape[0] > 0:
                rob = rob[0]
            else:
                rob = self.robot_state[idx]
            if target.shape[0] > 0:
                target = target[0]
            else:
                target = self.targets[idx]
        img_pair = self._prepare_imgs(img_pair)
        returns = [img_pair]

        if self.return_robot_state:
            returns += [rob]
        if self.return_targets:
            returns += [target]
        if self.return_latents:
            returns += [lat]

        return tuple(returns) if len(returns) > 1 else returns[0]


class iTHORDataset(data.Dataset):

    VAR_INFO = OrderedDict({})
    NORM_STATS = {
        'Apple_f33eaaa0_center_x': (0.45, 1.05),
        'Apple_f33eaaa0_center_y': (0.90, 1.50),
        'Apple_f33eaaa0_center_z': (-1.05, -0.30),
        'Egg_afaaaca3_center_x': (0.35, 1.15),
        'Egg_afaaaca3_center_y': (0.90, 1.90),
        'Egg_afaaaca3_center_z': (-1.30, -0.30),
        'Knife_e6601c42_center_x': (0.35, 1.15),
        'Knife_e6601c42_center_y': (0.90, 1.60),
        'Knife_e6601c42_center_z': (-1.05, -0.30),
        'Plate_49b95a7a_center_x': (0.40, 1.10),
        'Plate_49b95a7a_center_y': (0.90, 1.90),
        'Plate_49b95a7a_center_z': (-1.20, -0.30),
    }
    ACTION_TYPES = [
        'ToggleObject',
        'OpenObject',
        'PickupObject',
        'PutObject',
        'MoveObject',
        'NoOp'
    ]
    OBJECT_NAMES = [
        'Cabinet_47fc321b',
        'CounterTop_f8092513',
        'Egg',
        'Egg_Cooked',  # Becomes independent of the previous time step
        'Egg_to_CounterTop_f8092513',
        'Egg_to_Pan',
        'Microwave',
        'Plate',
        'Plate_to_CounterTop_f8092513',
        'Plate_to_Microwave',
        'StoveKnob_690d0d5d',
        'StoveKnob_cf670576',
        'StoveKnob_c8955f66',
        'StoveKnob_38c1dbc2',
        'Toaster',
        'NoObject1',
        'NoObject2',
        'NoObject3',
    ]

    def __init__(self, 
                 data_folder, 
                 split='train', 
                 single_image=False, 
                 return_robot_state=True, 
                 return_targets=False, 
                 return_latents=False, 
                 triplet=False,
                 seq_len=2, 
                 cluster=False,
                 try_encodings=False,
                 categorical_actions=False,
                 return_text=False,
                 subsample_percentage=1.0,
                 subsample_chunk=0,
                 debug_data=False,
                 **kwargs):
        super().__init__()
        self.cluster = cluster
        filename = split
        if triplet:
            print('[!] Warning: Triplets are not supported for iTHOR datasets yet. Using fake setups instead.')
            # filename += '_triplets'
        self.split_name = filename + ('_triplets' if triplet else '')
        data_split_folder = os.path.join(data_folder, f'{filename}')
        if split.startswith('val') and not os.path.isdir(data_split_folder):
            self.split_name = self.split_name.replace('val', 'test')
            print('[!] WARNING: Could not find a validation dataset. Falling back to the standard test set. Do not use it for selecting the best model!')
            data_split_folder = os.path.join(data_folder, f'{filename.replace("val", "test")}')
        assert os.path.isdir(data_split_folder), f'Could not find iTHOR dataset at {data_split_folder}'
        
        self.return_robot_state = return_robot_state
        self.return_targets = return_targets
        self.return_latents = return_latents
        self.triplet = triplet
        self.single_image = single_image
        self.encodings_active = False
        self.try_encodings = try_encodings
        self.categorical_actions = categorical_actions
        self.seq_len = seq_len if not (single_image or triplet) else 1
        self.return_text = return_text
        self.subsample_percentage = subsample_percentage if not debug_data else 0.05
        self.subsample_chunk = subsample_chunk if not debug_data else 0

        if kwargs.get('perfect_interactions', False):
            self.return_latents = True
            self.return_targets = True
        # Loading data
        data = self.load_data_from_folder(data_split_folder)
        # Preparing images
        self.imgs = data.pop('frames')
        # Fake triplets
        if triplet:
            self.imgs = np.stack([self.imgs] * 3, axis=2)
        if len(self.imgs.shape) > (4 if not self.encodings_active else 2) and not triplet:
            arange_0 = np.arange(self.imgs.shape[0], dtype=np.int32)
            arange_1 = np.arange(start=0,
                                 stop=self.imgs.shape[1]-self.seq_len+1,
                                 dtype=np.int32)
            self.indices = arange_1[None,:] + arange_0[:,None] * self.imgs.shape[1]
            self.indices = np.reshape(self.indices, (-1,))
        else:
            self.indices = np.arange(start=0,
                                     stop=np.prod(self.imgs.shape[:2])-self.seq_len+1,
                                     dtype=np.int32)
        self.imgs = np.reshape(self.imgs, (-1,) + self.imgs.shape[2:])
        flatten_arrays_in_dict(data, 0, 2)
        self.imgs = torch.from_numpy(self.imgs)
        dict_numpy_to_torch(data)

        # Causal vars
        self.keys_causal = data['causal_keys']
        self.latent_state = data['latents']
        for i, key in enumerate(self.keys_causal):
            if any([key.endswith(postfix) for postfix in ['_open', '_on', '_sliced', '_broken', '_pickedup', '_cooked']]):
                iTHORDataset.VAR_INFO[key] = 'categ_2'
            elif any([key.endswith(f'_center_{postfix}') for postfix in ['x', 'y', 'z']]):
                if key in iTHORDataset.NORM_STATS:
                    new_min, new_max = iTHORDataset.NORM_STATS[key]
                    self.latent_state[:,i] = (self.latent_state[:,i] - new_min) / (new_max - new_min)
                else:
                    print(f'WARNING: No normalization stats for {key}!')
                    self.latent_state[:,i] = (self.latent_state[:,i] - self.latent_state[:,i].min()) / (self.latent_state[:,i].max() - self.latent_state[:,i].min())
                self.latent_state[:,i] = (self.latent_state[:,i] - 0.5) * 2.
                iTHORDataset.VAR_INFO[key] = 'continuous_1.0'
                if self.latent_state[:,i].min() < -1.0 or self.latent_state[:,i].max() > 1.0:
                    print(f'WARNING: Latent {key} out of bounds!')
                    print(self.latent_state[:,i].min(), self.latent_state[:,i].max())
                    # self.latent_state[:,i] = torch.clip(self.latent_state[:,i], -1.0, 1.0)
                    print('Ignoring for now...')
            else:
                assert False, f'Unknown causal key {key}'
        self.targets = data['targets'].float()
        if self.return_text:
            self.input_ids = data['input_ids'].long()
            self.token_type_ids = data['token_type_ids'].long()
            self.attention_mask = data['attention_mask'].long()

        # Robotic state
        if self.categorical_actions:
            action_type = F.one_hot(data['action_type'].long(), num_classes=len(iTHORDataset.ACTION_TYPES)).float()
            object_name = F.one_hot(data['object_name'].long(), num_classes=len(iTHORDataset.OBJECT_NAMES)).float()
            print(object_name.shape, action_type.shape, self.latent_state.shape)
            latent_index = self.keys_causal.index('Egg_afaaaca3_cooked')
            action_index = iTHORDataset.OBJECT_NAMES.index('Egg_Cooked')
            # Add action for cooking egg
            object_name[1:,action_index] = torch.logical_and(self.latent_state[:-1,latent_index] == 0., self.latent_state[1:,latent_index] == 1.).float()
            self.robot_state = torch.concatenate([action_type, object_name], dim=-1)
            print(self.robot_state[1])
            print(self.robot_state[105])
        else:
            self.robot_state = data['actions'] * 2. - 1.  # [-1, 1]
            assert not torch.isnan(self.robot_state).any()

    def load_data_from_folder(self, data_folder):
        data = {}
        all_seq_files = sorted(glob(os.path.join(data_folder, '*seq_*.npz')))
        num_to_load = int(len(all_seq_files) * self.subsample_percentage)
        # Check if the subsample chunk is valid
        if (self.subsample_chunk + 1) * num_to_load > len(all_seq_files):
            print(f'WARNING: Subsample chunk {self.subsample_chunk} is too large and it overflows the dataset. Setting it to 0.')
            self.subsample_chunk = 0
        offset = self.subsample_chunk * num_to_load
        # seq_files = sample(all_seq_files, num_to_load)  # randomly sample files
        # seq_files = sorted(glob(os.path.join(data_folder, '*seq_*.npz')))
        seq_files = all_seq_files[offset:offset+num_to_load]
        seq_files = [f for f in seq_files if not f.endswith('_encodings.npz')]
        skipped = []
        for file_idx, file in enumerate(tqdm_track(seq_files, desc=f'Loading sequences of {self.split_name}', leave=False, cluster=self.cluster)):
            data_seq = np.load(file, allow_pickle=True)
            # del data_seq['descriptions']
            if file_idx == 0:
                num_latents = data_seq['latents'].shape[1]
            if data_seq['latents'].shape[1] != num_latents:
                skipped.append(file)
                print(f'WARNING: Skipping {file} because of inconsistent latent size!')
                continue
            data_seq_keys = sorted(list(data_seq.keys()))
            if any(['description' in key for key in data_seq_keys]):
                data_seq_keys = [key for key in data_seq_keys if not 'description' in key]
            # if 'collected_descriptions' in data_seq_keys:
            #     data_seq_keys.remove('collected_descriptions')
            if self.try_encodings and os.path.isfile(file.replace('.npz', '_encodings.npz')):
                data_seq_enc = np.load(file.replace('.npz', '_encodings.npz'), allow_pickle=True)
                data_seq_keys.remove('frames')
                data_seq_keys.append('encodings')
            action_info_file = file.replace('.npz', '_infos.json')
            with open(action_info_file, 'r') as f:
                action_info = json.load(f)
            action_info['object_name'] = [f'{pic}_to_{ob}' if ac == 'PutObject' else ob for ob, ac, pic in zip(action_info['object_name'], action_info['action_type'], [None] + action_info['pickedObject'][:-1])]
            action_info['action_type'].insert(0, 'NoOp')
            action_info['object_name'].insert(0, 'NoObject1')
            data_seq_keys += ['action_type', 'object_name']
            if 'targets' not in data_seq_keys:
                data_seq_keys.append('targets')
                targets = create_targets(action_info, action_info['causal_keys'])
            else:
                targets = None
            if np.isnan(data_seq['actions']).any():
                print(f'WARNING: NaNs in actions of {file}!')
                continue
            for key in data_seq_keys:
                if key == 'encodings':
                    val = data_seq_enc[key]
                elif key == 'targets' and targets is not None:
                    val = targets
                elif key == 'action_type':
                    val = np.array([iTHORDataset.ACTION_TYPES.index(ac) for ac in action_info['action_type']])
                elif key == 'object_name':
                    val = np.array([iTHORDataset.OBJECT_NAMES.index(ob) for ob in action_info['object_name']])
                else:
                    val = data_seq[key]
                if key == 'frames' and val.ndim == 4 and val.shape[-1] == 3:
                    val = val.transpose(0, 3, 1, 2)
                elif key == 'encodings':
                    self.encodings_active = True
                    key = 'frames'
                if key == 'causal_keys':
                    val = val.tolist()
                    if key is not None:
                        data[key] = val
                    else:
                        assert data[key] == val, f'Inconsistent causal keys in {file}'
                else:
                    if key not in data:
                        data[key] = np.zeros((len(seq_files),) + val.shape, dtype=val.dtype)
                    data[key][file_idx] = val
        return data

    def encode_dataset(self, encode_func, batch_size=16):
        encodings = None
        for idx in tqdm_track(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False, cluster=self.cluster):
            batch = self.imgs[idx:idx+batch_size]
            batch = self._prepare_imgs(batch)
            if len(batch.shape) == 5:  # Triplets
                batch = batch.flatten(0, 1)
            with torch.no_grad():
                batch = encode_func(batch)
            if len(self.imgs.shape) == 5:
                batch = batch.reshape((-1, self.imgs.shape[1]) + batch.shape[1:])
            if encodings is None:
                encodings = np.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=np.float32)
            encodings[idx:idx+batch_size] = batch
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    def load_encodings(self, filename):
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            return imgs

    def action_size(self):
        return self.robot_state.shape[-1]

    def num_labels(self):
        return -1

    def num_vars(self):
        return len(self.keys_causal)

    def target_names(self):
        return self.keys_causal

    def get_img_width(self):
        return self.imgs.shape[-1] if not self.encodings_active else 256

    def get_inp_channels(self):
        return self.imgs.shape[-3] if not self.encodings_active else self.imgs.shape[-1]

    def get_causal_var_info(self):
        return iTHORDataset.VAR_INFO

    def get_adj_matrix(self):
        return np.zeros((self.num_vars, self.num_vars), dtype=np.float32)

    def get_temporal_adj_matrix(self):
        return np.eye(self.num_vars, dtype=np.float32)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        idx = self.indices[idx]
        returns = []

        if self.triplet:
            img_pair = self.imgs[idx]
            rob = self.robot_state[idx]
            lat = self.latent_state[idx]
            target = self.targets[idx]
        else:
            img_pair = self.imgs[idx:idx+self.seq_len]
            rob = self.robot_state[idx+1:idx+self.seq_len]
            lat = self.latent_state[idx:idx+self.seq_len]
            target = self.targets[idx+1:idx+self.seq_len]

        if self.single_image:
            img_pair = img_pair[0]
            lat = lat[0]
            if rob.shape[0] > 0:
                rob = rob[0]
            if target.shape[0] > 0:
                target = target[0]
        img_pair = self._prepare_imgs(img_pair)
        returns = [img_pair]

        if self.return_robot_state:
            returns += [rob]
        if self.return_targets:
            returns += [target]
        if self.return_text:
            returns += [self.input_ids[idx + 1:idx+self.seq_len], self.token_type_ids[idx + 1:idx+self.seq_len], self.attention_mask[idx + 1:idx+self.seq_len]]
        if self.return_latents:
            returns += [lat]

        return tuple(returns) if len(returns) > 1 else returns[0]

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import json
from glob import glob
from tqdm import tqdm as tqdm_track
import torch.nn.functional as F

class GridworldDataset(Dataset):

    def __init__(self, data_folder, split='train', single_image=False, return_robot_state=True, 
                 return_targets=False, return_latents=False, triplet=False,
                 seq_len=2, cluster=False, try_encodings=False, categorical_actions=False,
                 return_text=False, subsample_percentage=1.0, subsample_chunk=0,
                 debug_data=False, **kwargs):

        super().__init__()
        self.cluster = cluster
        self.split_name = split
        self.return_robot_state = return_robot_state
        self.return_targets = return_targets
        self.return_latents = return_latents
        self.triplet = triplet
        self.single_image = single_image
        self.encodings_active = False
        self.try_encodings = try_encodings
        self.categorical_actions = categorical_actions
        self.seq_len = seq_len if not (single_image or triplet) else 1
        self.return_text = return_text
        self.subsample_percentage = subsample_percentage if not debug_data else 0.05
        self.subsample_chunk = subsample_chunk if not debug_data else 0
        # Store the file paths instead of loading all data
        self.data_files = sorted(glob(os.path.join(data_folder, f'{split}', '*.npz')))
        self.data_files = self.data_files[int(self.subsample_chunk * self.subsample_percentage * len(self.data_files)):int((self.subsample_chunk + 1) * self.subsample_percentage * len(self.data_files))]
        self.indices = range(len(self.data_files) * (seq_len - 1))
        self.indices_per_file = 40
        # Load metadata
        self.load_metadata(data_folder, split)
        
        # Load first file to get the image size
        data_seq = np.load(self.data_files[0], allow_pickle=True)
        self.imgs = data_seq['frames']
        # self.imgs = self._prepare_imgs(self.imgs)
        

        # Loading data
        data = self.load_data_from_folder(data_folder, split)
        # print(data)
        self.imgs = data.pop('frames')

        if len(self.imgs.shape) > 4 and not triplet:
            arange_0 = np.arange(self.imgs.shape[0], dtype=np.int32)
            arange_1 = np.arange(start=0, stop=self.imgs.shape[1]-self.seq_len+1, dtype=np.int32)
            self.indices = arange_1[None,:] + arange_0[:,None] * self.imgs.shape[1]
            self.indices = np.reshape(self.indices, (-1,))
        else:
            self.indices = np.arange(start=0, stop=np.prod(self.imgs.shape[:2])-self.seq_len+1, dtype=np.int32)
        
        # self.imgs = np.reshape(self.imgs, (-1,) + self.imgs.shape[2:])
        self.imgs = torch.from_numpy(self.imgs)

        self.robot_state = data['actions']
        self.targets = data['interventions']
        self.latent_state = data['causals']

        if self.return_text:
            self.input_ids = data['input_ids']
            self.token_type_ids = data['token_type_ids']
            self.attention_mask = data['attention_mask']

    def load_data_from_folder(self, data_folder, split):
        data_folder = os.path.join(data_folder, f'{split}')
        # data_files = sorted(glob(os.path.join(data_folder, '*.npz')))
        data = {}
        for file in tqdm(self.data_files, desc=f'Loading sequences of {self.split_name}', leave=False, total=len(self.data_files)):
            data_seq = np.load(file, allow_pickle=True)
            for key, val in data_seq.items():
                if key not in data:
                    data[key] = val
                else:
                    data[key] = np.concatenate([data[key], val], axis=0)
        return data

    def load_metadata(self, data_folder, split):
        metadata_file = os.path.join(data_folder, f'{split}_metadata.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        self.object_names = metadata['object_names']
        self.action_types = metadata['action_types']
        self.causal_keys = metadata['causal_keys']
        self.flattened_causals = metadata['flattened_causals']
        self.are_interventions_stochastic = metadata['are_interventions_stochastic']

    def encode_dataset(self, encode_func, batch_size=16):
        # Function to encode the dataset using a provided encoding function
        encodings = None
        for idx in tqdm_track(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
            batch = self.imgs[idx:idx+batch_size]
            batch = self._prepare_imgs(batch)
            with torch.no_grad():
                batch = encode_func(batch)
            if encodings is None:
                encodings = np.zeros((self.imgs.shape[0],) + batch.shape[1:], dtype=np.float32)
            encodings[idx:idx+batch_size] = batch.cpu().numpy()
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    def load_encodings(self, filename):
        # Load precomputed encodings from a file
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            if len(imgs.shape) == 4:
                imgs = imgs.permute(0, 3, 1, 2)
            else:
                imgs = imgs.permute(2, 0, 1)
            return imgs

    def __len__(self):
        return len(self.data_files) * (self.indices_per_file if self.single_image else self.seq_len - 1)
    
    def action_size(self):
        # Return the size of the action space
        return len(self.action_types)
    
    def num_labels(self):
        return len(self.flattened_causals)
    
    def num_vars(self):
        return len(self.flattened_causals)
    
    def target_names(self):
        return self.flattened_causals
    
    def get_img_width(self):
        return self.imgs.shape[-2]
    
    def get_inp_channels(self):
        return self.imgs.shape[-1]

    def __getitem__(self, idx):
        idx = self.indices[idx]
        returns = []

        img_pair = self.imgs[idx:idx+self.seq_len]
        rob = self.robot_state[idx+1:idx+self.seq_len]
        lat = self.latent_state[idx:idx+self.seq_len]
        target = self.targets[idx+1:idx+self.seq_len]

        if self.single_image:
            img_pair = img_pair[0]
            lat = lat[0]
            if rob.shape[0] > 0:
                rob = rob[0]
            if target.shape[0] > 0:
                target = target[0]
        img_pair = self._prepare_imgs(img_pair)
        returns = [img_pair]

        if self.return_robot_state:
            returns += [rob]
        if self.return_targets:
            returns += [target]
        if self.return_text:
            returns += [self.input_ids[idx + 1:idx+self.seq_len], self.token_type_ids[idx + 1:idx+self.seq_len], self.attention_mask[idx + 1:idx+self.seq_len]]
        if self.return_latents:
            returns += [lat]

        return tuple(returns) if len(returns) > 1 else returns[0]
    # def __getitem__(self, idx):
    #     if self.single_image:
    #         # Calculate which file and which index in the file we need
    #         file_idx = idx // self.indices_per_file
    #         idx_within_file = idx % self.indices_per_file
            
    #         # Load only the specific .npz file
    #         data_seq = np.load(self.data_files[file_idx], allow_pickle=True)
            
    #         # Get the specific data points
    #         img = data_seq['frames'][idx_within_file]
    #         img = self._prepare_imgs(torch.from_numpy(img).float())

    #         # Prepare other data if requested
    #         returns = [img]
    #         if self.return_robot_state:
    #             rob = data_seq['actions'][idx_within_file - 1]
    #             returns.append(torch.from_numpy(rob).float())
    #         if self.return_latents:
    #             lat = data_seq['latents'][idx_within_file]
    #             returns.append(torch.from_numpy(lat).float())
    #         if self.return_targets:
    #             target = data_seq['targets'][idx_within_file - 1]
    #             returns.append(torch.from_numpy(target).float())

    #         return tuple(returns) if len(returns) > 1 else returns[0]
    #     else:
    #         # Calculate which file and which index in the file we need for sequences
    #         file_idx = idx // self.indices_per_file
    #         idx_within_file = idx % self.indices_per_file
            
    #         # Adjust index to get a sequence
    #         start_idx = max(idx_within_file - self.seq_len + 1, 0)
    #         end_idx = start_idx + self.seq_len

    #         # Load only the specific .npz file
    #         data_seq = np.load(self.data_files[file_idx], allow_pickle=True)
            
    #         # Get the specific data points for the sequence
    #         img_seq = data_seq['frames'][start_idx:end_idx]
    #         img_seq = self._prepare_imgs(torch.from_numpy(img_seq).float())

    #         # Prepare other data if requested
    #         returns = [img_seq]
    #         if self.return_robot_state:
    #             rob_seq = data_seq['actions'][start_idx:end_idx-1]
    #             returns.append(torch.from_numpy(rob_seq).float())
    #         if self.return_latents:
    #             lat_seq = data_seq['latents'][start_idx:end_idx]
    #             returns.append(torch.from_numpy(lat_seq).float())
    #         if self.return_targets:
    #             target_seq = data_seq['targets'][start_idx:end_idx-1]
    #             returns.append(torch.from_numpy(target_seq).float())

    #         return tuple(returns) if len(returns) > 1 else returns[0]
