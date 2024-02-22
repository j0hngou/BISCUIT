import json
import multiprocessing
import os
import random
from functools import partial
from multiprocessing import Pool

import numpy as np
from gridworld import GridEntity, Gridworld
from tqdm import tqdm
import argparse


def save_metadata(dataset_name, split, gridworld, simplified=False):
    object_names = ['_'.join(a.split('_')[:-1]) for a in sorted(gridworld.get_causals())]
    action_types = ['turn_left', 'turn_right', 'turn_up', 'turn_down', 'change_state', 'move_to_left', 'move_to_right', 'move_to_up', 'move_to_down']
    causal_vector = gridworld.get_causal_vector(are_light_positions_fixed=True)
    flattened_causals = sorted(gridworld.get_flattened_causals().keys())
    if simplified:
        flattened_causals = [a for a in flattened_causals if 'vehicle_(255, 0, 0)_position_x' not in a]
    causal_keys = list(gridworld.get_causals().keys())
    are_interventions_stochastic = True
    metadata = {
            'object_names': object_names,
            'action_types': action_types,
            'causal_keys': causal_keys,
            'flattened_causals': flattened_causals,
            'are_interventions_stochastic': are_interventions_stochastic,
            'grid_x': gridworld.width,
            'grid_y': gridworld.height,
            'sprite_size': gridworld.sprite_size,
    }

    # Save the metadata
    with open(f'data/{dataset_name}/{split}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)        


def run_simulation(seed, split, dataset_name='gridworld', grid_x=16, grid_y=16, sprite_size=32, fixed_light_positions=None, save_metadata_flag=False, pre_intervention_step=False):

    random.seed(seed)
    np.random.seed(seed)
    
    car_colors = [
        (255, 0, 0), # Red
        (0, 0, 255), # Blue
        # (0, 255, 255), # Cyan
        # (192, 192, 192), # Silver
        # (255, 165, 0), # Orange
    ]
    light_colors = [
        # (0, 0, 255), # Blue
        (0, 255, 255), # Cyan
        (192, 192, 192), # Silver
        # (255, 165, 0), # Orange
        # (100, 100, 0), # Dark Olive
    ]
    boulder_colors = [
        # (255, 0, 0), # Red
        # (0, 0, 255), # Blue
        # (0, 255, 255), # Cyan
        # (192, 192, 192), # Silver
        (255, 165, 0), # Orange
    ]

    # Preload sprites for each entity type with their specific color subsets
    colors_dict = {
        'cars': car_colors,
        'lights': light_colors,
        'boulders': boulder_colors
    }
    orientations = ['up', 'down', 'left', 'right']
    GridEntity.preload_sprites(colors_dict, orientations, sprite_path='sprites/', sprite_size=sprite_size)

    # Create an instance of Gridworld
    gridworld = Gridworld(grid_x, grid_y, sprite_size=sprite_size)

    # Initialize the gridworld with vehicles, traffic lights, and boulders
    gridworld.randomly_initialize(car_colors, light_colors, boulder_colors, num_cars=2, num_lights=2, num_boulders=1, fixed_light_positions=fixed_light_positions, x_percent=50, y_percent=10, z_percent=20)

    # Run the simulation
    gridworld.step()  # Initial step to set up the environment
    initial_frame = gridworld.render()
    initial_causal_vector = gridworld.get_causal_vector(are_light_positions_fixed=True)
    # initial_causal_vector = initial_causal_vector[:4] + initial_causal_vector[5:]

    frames = [initial_frame.copy()]  # List of frames, starting with the initial frame
    causals = [initial_causal_vector]  # List of causals, starting with the initial state
    actions = []  # List of actions
    action_descriptions = []  # List of action descriptions
    interventions = []  # List of interventions

    # Generation loop
    for _ in range(1, 20):  # Start from 1 since we already have the initial state
        if pre_intervention_step:
            gridworld.step()
        pre_step_causals = gridworld.get_causals()
        action, intervention = gridworld.semi_random_intervention()
        if not pre_intervention_step:
            intervention = gridworld.step(intervention, pre_step_causals)
        # Append action and intervention information
        actions.append(action)
        # Simplified, so the car position x is unidentifiable, so we remove it
        # intervention = {key: value for key, value in intervention.items() if key != 'vehicle_(255, 0, 0)_position_x'}
        interventions.append([intervention[key] for key in sorted(intervention.keys())])

        # Append causal information
        # Simplified, so the car position x is unidentifiable, so we remove it
        causal_vector = gridworld.get_causal_vector(are_light_positions_fixed=True)
        # causal_vector = causal_vector[:4] + causal_vector[5:]
        
        causals.append(causal_vector)
        
        # Append action description
        action_description = gridworld.describe_action(pre_step_causals, action)
        action_descriptions.append(action_description)

        # Render and save the frame
        frame = gridworld.render()
        # frame_name = f"{seed}_{gridworld.step_count}.png"
        # frame.save(f"frames/{frame_name}")
        frames.append(frame.copy())
    
    # Create a directory if it doesn't exist
    if not os.path.exists(f'data/{dataset_name}/{split}'):
        os.makedirs(f'data/{dataset_name}/{split}', exist_ok=True)
    
    np.savez_compressed(f'data/{dataset_name}/{split}/gridworld_episode_{seed}.npz', 
                frames=np.array(frames),
                causals=np.array(causals),
                actions=np.array(actions),
                interventions=np.array(interventions),
                action_descriptions=np.array(action_descriptions),
    )
    
    if save_metadata_flag:
        save_metadata(dataset_name, split, gridworld)
    



def run_batch(batch_seeds, split, dataset_name='gridworld', grid_x=16, grid_y=16, sprite_size=32, fixed_light_positions=[], pre_intervention_step=False):
    for seed in batch_seeds:
        run_simulation(seed, split, dataset_name, grid_x, grid_y, sprite_size, fixed_light_positions, pre_intervention_step=pre_intervention_step)

def gen_data(seeds, batch_size, split, dataset_name='gridworld', grid_x=16, grid_y=16, sprite_size=32, fixed_light_positions=[], pre_intervention_step=False):
    # Run the first batch to save the metadata
    run_simulation(seeds[0], split, dataset_name, grid_x, grid_y, sprite_size, fixed_light_positions, save_metadata_flag=True, pre_intervention_step=pre_intervention_step)
    run_batch_split = partial(run_batch, split=split, dataset_name=dataset_name, grid_x=grid_x, grid_y=grid_y, sprite_size=sprite_size, fixed_light_positions=fixed_light_positions, pre_intervention_step=pre_intervention_step)
    batches = [seeds[i:i + batch_size] for i in range(0, len(seeds), batch_size)]
    
    with Pool() as pool:
        # Wrap pool.map with tqdm for progress tracking
        list(tqdm(pool.imap_unordered(run_batch_split, batches), total=len(batches)))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_x', type=int, default=8, help='Number of grid cells in the x-axis')
    parser.add_argument('--grid_y', type=int, default=8, help='Number of grid cells in the y-axis')
    parser.add_argument('--sprite_size', type=int, default=32, help='Size of the grid sprites')
    parser.add_argument('--fixed_light_positions', nargs='+', default=None, help='List of fixed light positions')
    parser.add_argument('--train_seeds', type=int, default=1000, help='Number of seeds for the train split')
    parser.add_argument('--val_seeds', type=int, default=100, help='Number of seeds for the validation split')
    parser.add_argument('--test_seeds', type=int, default=100, help='Number of seeds for the test split')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--dataset_name', type=str, default='gridworld_simplified_12c_3d', help='Name of the dataset')
    parser.add_argument('--pre_intervention_step', default=False, action="store_true", help="""
        If true, the intervention is applied before the step function is called.
        This means that the intervention's effects will be visible in the next frame.
        If false, the intervention is applied after the step function is called
        but it freezes the dynamics of the environment pertaining to the intervention.
        """)
    args = parser.parse_args()

    train_seeds = args.train_seeds
    val_seeds = args.val_seeds
    test_seeds = args.test_seeds
    batch_size = args.batch_size
    grid_x = args.grid_x
    grid_y = args.grid_y
    sprite_size = args.sprite_size
    dataset_name = args.dataset_name
    pre_intervention_step = args.pre_intervention_step

    if args.fixed_light_positions is None:
        # fixed_light_positions = [(0, 0, 'down'), (3, grid_y - 1, 'up'), (grid_x - 3, 0, 'down')]
        fixed_light_positions = [(0, 0, 'down'), (3, grid_y - 1, 'up')]#, (grid_x - 3, 0, 'down')]
    else:
        fixed_light_positions = args.fixed_light_positions

    seeds = range(train_seeds)

    gen_data(seeds, batch_size, 'train', dataset_name=dataset_name, grid_x=grid_x, grid_y=grid_y, sprite_size=sprite_size, fixed_light_positions=fixed_light_positions, pre_intervention_step=pre_intervention_step)

    seeds = range(train_seeds, train_seeds + val_seeds)
    print(f'Generating {seeds} seeds for the validation split')
    gen_data(seeds, batch_size, 'val', dataset_name=dataset_name, grid_x=grid_x, grid_y=grid_y, sprite_size=sprite_size, fixed_light_positions=fixed_light_positions, pre_intervention_step=pre_intervention_step)

    seeds = range(train_seeds + val_seeds, train_seeds + val_seeds + test_seeds)
    print(f'Generating {seeds} seeds for the test split')
    gen_data(seeds, batch_size, 'test', dataset_name=dataset_name, grid_x=grid_x, grid_y=grid_y, sprite_size=sprite_size, fixed_light_positions=fixed_light_positions, pre_intervention_step=pre_intervention_step)
    # # for i in range(15, 25):
    # run_simulation(15, 'check', dataset_name, grid_x, grid_y, sprite_size, fixed_light_positions, save_metadata_flag=True, pre_intervention_step=args.pre_intervention_step)