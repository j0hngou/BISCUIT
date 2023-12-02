""" 
Dataset generation of the iTHOR environment
"""

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from copy import deepcopy
from random import shuffle
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import imageio
import os
import torch
import torch.nn.functional as F
from collections import defaultdict
import json
import time
from hashlib import sha256
from scipy import signal
from concurrent.futures import ProcessPoolExecutor
import concurrent
from tqdm import tqdm

# The simulator itself runs on 512x512, but the images are downsampled by factor 2 afterward (256x256)
RESOLUTION = 512  
SIMPLE_SET = False
OBJECT_NAMES = [
    'Apple',
    'Knife',
    'Stove',
    'Microwave',
    'Cabinet',
    'Toaster',
    'Counter'
] + ([
    'Egg',
    'Pan',
    'Plate',
    'Potato'
] if not SIMPLE_SET else [])
MIN_DIST = 0.22
GRID_STRIDE = MIN_DIST / 10
DISTANCE_KNIFE_PLATE = 0.23

NOT_MOVABLE = [
    'Window',
    'Stove',
    'Sink',
    'Shelf',
    'LightSwitch'
]

FIXED_POSITION_DICT = [
    {
        "objectName": "Toaster",
        "rotation": {
            "x": 0, "y": 270, "z": 0
        },
        "position": {
            "x": 0.98, "y": 0.98, "z": -1.75
        }
    }
]
MOVABLE_POSITION_DICT = [
    {
        "objectName": "Knife",
        "rotation": {
            "x": 0, "y": 90, "z": 90
        }
    },
    {
        "objectName": "Apple",
        "rotation": {
            "x": 0, "y": 0, "z": 0
        }
    },
    {
        "objectName": "Egg",
        "rotation": {
            "x": 0, "y": 0, "z": 0
        }
    },
    {
        "objectName": "Potato",
        "rotation": {
            "x": 0, "y": 0, "z": 0
        }
    },
    {
        "objectName": "Plate",
        "rotation": {
            "x": 0, "y": 0, "z": 0
        }
    }
]
COUNTER_POSITIONS = [
    {
        "objectName": None,
        "position": {
            "x": 0.75, "y": 0.98, "z": -0.35
        }
    },
    {
        "objectName": None,
        "position": {
            "x": 1.03, "y": 0.98, "z": -0.35
        }
    },
    {
        "objectName": None,
        "position": {
            "x": 0.65, "y": 0.98, "z": -0.55
        }
    },
    {
        "objectName": None,
        "position": {
            "x": 1.00, "y": 0.98, "z": -0.81
        }
    },
    # {
    #     "objectName": None,
    #     "position": {
    #         "x": 1.03, "y": 0.98, "z": -0.55
    #     }
    # },
    {
        "objectName": None,
        "position": {
            "x": 0.68, "y": 0.98, "z": -0.81
        }
    }
]
CATEGORICAL_POSITION_DICT = [
    {
        "objectName": "Pan",
        "rotation": {
            "x": 0, "y": 0, "z": 0
        },
        "position": [
            {"x": 0.85, "y": 0.95, "z": -1.20},
            # {"x": 1.08, "y": 0.95, "z": -1.10},
            # {"x": 0.85, "y": 0.95, "z": -1.50},
            # {"x": 1.08, "y": 0.98, "z": -1.50},
        ]
    }
]
ACTIONS = {
    'PickupObject': [
        'Apple', 'Knife'
    ] + (['Egg', 'Plate'] if not SIMPLE_SET else []),
    'PutObject': (['Pan'] if not SIMPLE_SET else [])  + 
    [
        'Microwave', 'CounterTop_f8092513'
    ],
    'ToggleObject': [
        'Toaster', 'Microwave', 'StoveKnob_38c1dbc2', 'StoveKnob_690d0d5d', 'StoveKnob_c8955f66', 'StoveKnob_cf670576'
    ],
    'SliceObject': ([
        # 'Potato', 'Apple'
        'Apple',
    ] if not SIMPLE_SET else []),
    'OpenObject': [
        'Microwave', 'Cabinet_47fc321b'
    ],
    'NoOp': [
        'NoObject1', 'NoObject2', 'NoObject3' # 'NoObject3', 'NoObject4'
    ]
}
INTERACT_OBJS = list(set([obj for action_key in ['PickupObject', 'ToggleObject', 'SliceObject', 'OpenObject']
                              for obj in ACTIONS[action_key]]))
PICKUP = {'position': None}



def move_objects_into_position(controller, event, positions):
    object_poses = [{key: obj[key] for key in ["name", "rotation", "position"]} 
                    for obj in event.metadata['objects'] if not any([obj['name'].startswith(n) for n in NOT_MOVABLE])]
    for obj in object_poses:
        obj['objectName'] = obj.pop('name')
    for position in positions:
        for obj in object_poses:
            if obj["objectName"].startswith(position["objectName"]):
                for key in ["position", "rotation"]:
                    obj[key] = position[key]
    event = controller.step(action='SetObjectPoses',
                            objectPoses=object_poses)
    return event


def initialize_environment(seed:int=42):
    np.random.seed(seed)
    random.seed(seed)

    controller = Controller(width=RESOLUTION, 
                            height=RESOLUTION, 
                            gridSize=0.1, 
                            platform=CloudRendering,
                            renderInstanceSegmentation=True)

    # Move the agent to the starting position
    event = controller.step(action="Stand")
    event = controller.step(action="MoveAhead")
    for _ in range(3):
        event = controller.step(action="MoveLeft")
    event = controller.step(action='LookDown', degrees=10)

    # Move the objects into position
    position_dict = deepcopy(FIXED_POSITION_DICT)
    movable_position_dict = MOVABLE_POSITION_DICT
    success_positions = False
    while not success_positions:
        success_positions = True
        for i, p in enumerate(COUNTER_POSITIONS):
            pos_found = False
            num_tries = 0
            while not pos_found and num_tries < 10:
                pos_found = True
                num_tries += 1
                x = np.random.uniform(0.65, 1.00)
                z = np.random.uniform(-0.35, -0.81)
                for p_alt in COUNTER_POSITIONS[:i]:
                    if np.abs(x - p_alt['position']['x']) < MIN_DIST and np.abs(z - p_alt['position']['z']) < MIN_DIST:
                        pos_found = False
                        break
            if pos_found:
                p['position']['x'] = x
                p['position']['z'] = z
            else:
                success_positions = False
                break
    assert all([pos['objectName'] is None for pos in COUNTER_POSITIONS]), str(COUNTER_POSITIONS)
    for mov_pos, count_pos in zip(movable_position_dict, COUNTER_POSITIONS):
        count_pos['objectName'] = mov_pos['objectName']
        mov_pos['position'] = count_pos['position']
        mov_pos['counter_position'] = count_pos
    position_dict.extend(deepcopy(movable_position_dict))
    categorical_position_dict = deepcopy(CATEGORICAL_POSITION_DICT)
    for pos in categorical_position_dict:
        pos['position'] = pos['position'][np.random.randint(0, len(pos['position']))]
    position_dict.extend(categorical_position_dict)
    event = move_objects_into_position(controller, event, position_dict)

    # Removing unnecessary objects
    objects = list(event.metadata['objects'])
    for obj in objects:
        if not any([obj['name'].startswith(name) for name in OBJECT_NAMES]):
            event = controller.step(action="DisableObject",
                                    objectId=obj['objectId'])

    # Place potato on plate
    if not SIMPLE_SET:
        for obj in objects:
            if obj['name'].startswith('Potato'):
                event = controller.step(action="PickupObject",
                                        objectId=obj['objectId'])
                break
        for obj in objects:
            if obj['name'].startswith('Plate'):
                event = controller.step(action="PutObject",
                                        objectId=obj['objectId'])
                break
        for mov_pos in MOVABLE_POSITION_DICT:
            if mov_pos['objectName'] == 'Potato':
                MOVABLE_POSITION_DICT.remove(mov_pos)
                break

    return controller, event


def get_environment_state(event : dict):
    state = {}
    for obj in event.metadata['objects']:
        object_name = obj['name']
        if not obj['visible'] or not any([object_name.startswith(name) for name in INTERACT_OBJS]):
            continue
        # if any([k in object_name for k in ['_Slice_', '_Sliced_', '_Cracked_']]):
            # continue
        if obj['pickupable']:
            state[object_name + '_pickedup'] = int(obj['isPickedUp'])
            for key in obj['position']:
                state[object_name + '_center_' + key] = obj['position'][key]
                if obj['position'][key] == 0:
                    print('Suspicious position', object_name, obj['position'])
        if obj['openable']:
            state[object_name + '_open'] = int(obj['isOpen'])
        if obj['toggleable']:
            state[object_name + '_on'] = int(obj['isToggled'])
        if obj['sliceable'] and not obj['breakable'] and 'Knife' in OBJECT_NAMES:
            state[object_name + '_sliced'] = int(obj['isSliced'])
        if obj['breakable'] and object_name.startswith('Egg'):
            state[object_name + '_broken'] = int(obj['isBroken'])
        if obj['cookable']:
            state[object_name + '_cooked'] = int(obj['isCooked'])
    return state


def get_object_id(event, object_name):
    for obj in event.metadata['objects']:
        if obj['name'].startswith(object_name):
            return obj['objectId'], obj
    return None, None


def get_object_segmentations(event : dict):
    segmentations = {}
    for obj_name in INTERACT_OBJS:
        loc_objId, _ = get_object_id(event, obj_name)
        if not loc_objId in event.instance_masks:
            continue
        loc_mask = event.instance_masks[loc_objId]
        segmentations[obj_name] = loc_mask
    return segmentations


def get_action_pos(event : dict, objectId : str, action_type : str, randomize : bool = True, obj : dict = None):
    if action_type == 'PutObject':
        for sub_obj in event.metadata['objects']:
            if sub_obj['isPickedUp']:
                objectId = sub_obj['objectId']
                break
    if action_type == 'NoOp':
        instance_mask = None
        for obj_name in INTERACT_OBJS:
            loc_objId, _ = get_object_id(event, obj_name)
            if not loc_objId in event.instance_masks:
                continue
            loc_mask = event.instance_masks[loc_objId]
            if instance_mask is None:
                instance_mask = loc_mask
            else:
                instance_mask = instance_mask + loc_mask
        instance_mask = signal.convolve2d(instance_mask, np.ones((9, 9)), mode='same', boundary='fill', fillvalue=0)
        object_positions = np.where(instance_mask == 0)
    elif objectId not in event.instance_masks:
        print('Invalid action, try again')
        return None
    else:
        instance_mask = event.instance_masks[objectId]
        object_positions = np.where(instance_mask)
        # For the Microwave, we want to split the action into two parts
        # One toggles the microwave on/off, the other opens the door
        if objectId.startswith('Microwave'):
            q_high = np.percentile(object_positions[1], q=80)
            q_low = np.percentile(object_positions[1], q=30)
            if action_type == 'ToggleObject':
                pos_mask = object_positions[1] > q_high
            elif action_type == 'OpenObject' and not obj['isOpen']:
                pos_mask = object_positions[1] <= q_high
            elif action_type == 'OpenObject' and obj['isOpen']:
                pos_mask = object_positions[1] <= q_low
            elif action_type == 'PutObject':
                pos_mask = (object_positions[1] > q_low) * (object_positions[1] < q_high)
            else:
                print('Invalid action, try again')
                return None
            object_positions = tuple(pos[pos_mask] for pos in object_positions)
    if randomize:
        pos = np.random.randint(0, object_positions[0].shape[0])
    else:
        pos = 0
    object_position = [p[pos] for p in object_positions]
    action_pos = np.zeros(2, dtype=np.float32)
    for i in range(action_pos.shape[0]):
        action_pos[i] = object_position[i] * 1. / instance_mask.shape[i]
    return action_pos

# MOVE ACTION HELPERS
def generate_possible_positions(min_x, max_x, min_z, max_z, stride):
    x_positions = np.arange(min_x, max_x, stride)
    z_positions = np.arange(min_z, max_z, stride)
    positions = np.array(np.meshgrid(x_positions, z_positions)).T.reshape(-1, 2)
    # Filter out the positions where the knife is too close to the plate
    # plate_pos = None
    # knife_pos = None
    # for mov_pos in MOVABLE_POSITION_DICT:
    #     if mov_pos['objectName'] == 'Knife':
    #         if mov_pos['counter_position'] is None:
    #             knife_pos = None
    #         else:
    #             knife_pos = mov_pos['counter_position']['position']['x'], mov_pos['counter_position']['position']['z']
    #     elif mov_pos['objectName'] == 'Plate':
    #         if mov_pos['counter_position'] is None:
    #             plate_pos = None
    #         else:
    #             plate_pos = mov_pos['counter_position']['position']['x'], mov_pos['counter_position']['position']['z']
    # if knife_pos is not None and plate_pos is not None:
    #     distances = np.linalg.norm(positions - knife_pos, axis=1, ord=1)
    #     positions = positions[np.where(distances > DISTANCE_KNIFE_PLATE)]
    #     distances = np.linalg.norm(positions - plate_pos, axis=1, ord=1)
    #     positions = positions[np.where(distances > DISTANCE_KNIFE_PLATE)]
    return positions


def is_position_valid(pos, existing_positions, min_dist, is_knife=False, is_plate=False):
    distances = np.linalg.norm(existing_positions - pos, axis=1, ord=1)
    # if is_knife or is_plate:
    #     # knife_pos = MOVABLE_POSITION_DICT[0]['counter_position']['position']['x'], MOVABLE_POSITION_DICT[0]['counter_position']['position']['z']
    #     # plate_pos = MOVABLE_POSITION_DICT[1]['counter_position']['position']['x'], MOVABLE_POSITION_DICT[1]['counter_position']['position']['z']
    #     # Find the knife in the movable positions dict
    #     knife_pos = None
    #     plate_pos = None
    #     for mov_pos in MOVABLE_POSITION_DICT:
    #         if mov_pos['objectName'] == 'Knife':
    #             if mov_pos['counter_position'] is None:
    #                 knife_pos = None
    #                 return np.all(distances > min_dist)
    #             else:
    #                 knife_pos = mov_pos['counter_position']['position']['x'], mov_pos['counter_position']['position']['z']
    #         elif mov_pos['objectName'] == 'Plate':
    #             if mov_pos['counter_position'] is None:
    #                 plate_pos = None
    #                 return np.all(distances > min_dist)
    #             else:
    #                 plate_pos = mov_pos['counter_position']['position']['x'], mov_pos['counter_position']['position']['z']
        
    #     if knife_pos is None and plate_pos is None:
    #         return np.all(distances > min_dist)
    #     if is_plate:
    #         if np.abs(pos[0] - knife_pos[0]) < DISTANCE_KNIFE_PLATE \
    #             and np.abs(pos[1] - knife_pos[1]) < DISTANCE_KNIFE_PLATE:
    #             return False
    #     elif is_knife:
    #         if np.abs(pos[0] - plate_pos[0]) < DISTANCE_KNIFE_PLATE \
    #             and np.abs(pos[1] - plate_pos[1]) < DISTANCE_KNIFE_PLATE:
    #             return False

    return np.all(distances > min_dist)
        

        

    # if object_name.startswith('Knife') and 'Plate' in MOVABLE_POSITION_DICT[0]['counter_position']['objectName'] \
    #     or object_name.startswith('Plate') and 'Knife' in MOVABLE_POSITION_DICT[0]['counter_position']['objectName']:
    #     if np.abs(pos[0] - MOVABLE_POSITION_DICT[0]['counter_position']['position']['x']) < DISTANCE_KNIFE_PLATE \
    #         and np.abs(pos[1] - MOVABLE_POSITION_DICT[0]['counter_position']['position']['z']) < DISTANCE_KNIFE_PLATE:
    #         continue
    

# def predict_future_placement(current_pos, possible_positions, existing_positions, min_dist):
#     """
#     We want to make sure of two things:
#     1. There is at least one position where we can place a new object
#     2. There is at least one position where we can move an existing object
#     Note that moving the existing objects is constrained by a box around the original position
#     of the existing object. This is to prevent objects from being moved too far away from their
#     original position.
#     """
#     # Check if we can place a new object
#     future_placement_possible = False
#     for future_pos in possible_positions:
#         if not np.array_equal(future_pos, current_pos):
#             # Check for placing a new object on the counter
#             if is_position_valid(future_pos, existing_positions, min_dist):
#                 future_placement_possible = True
#             # Check if we can wiggle all of the existing objects around after placing the new object
#             booleans = []
#             if current_pos in existing_positions:
#                 existing_positions = np.delete(existing_positions, np.where(np.all(existing_positions == current_pos, axis=1)), axis=0)
#             else:
#                 new_existing_positions = np.concatenate([existing_positions, [current_pos]])
#             for pos in new_existing_positions:
#                 # Create list of possible positions around original position
#                 original_positions = pos[0], pos[1]
#                 min_x, max_x = max(original_positions[0] - 0.2, 0.65), min(original_positions[0] + 0.2, 1.00)
#                 min_z, max_z = max(original_positions[1] - 0.2, -0.81), min(original_positions[1] + 0.2, -0.35)
#                 grid_stride = min_dist / 10
#                 possible_positions_move = generate_possible_positions(min_x, max_x, min_z, max_z, grid_stride)
#                 # Check if we can move the object to any of the possible positions
#                 for future_pos in possible_positions_move:
#                     if not np.array_equal(future_pos, original_positions):
#                         booleans.append(is_position_valid(future_pos, new_existing_positions, min_dist))



#     # booleans = []

#     # for pos in existing_positions:
#     #     # Create list of possible positions around original position
#     #     original_positions = pos[0], pos[1]
#     #     min_x, max_x = max(original_positions[0] - 0.2, 0.65), min(original_positions[0] + 0.2, 1.00)
#     #     min_z, max_z = max(original_positions[1] - 0.2, -0.81), min(original_positions[1] + 0.2, -0.35)
#     #     grid_stride = min_dist / 10
#     #     possible_positions_move = generate_possible_positions(min_x, max_x, min_z, max_z, grid_stride)
#     #     # Check if we can move the object to any of the possible positions
#     #     for future_pos in possible_positions_move:
#     #         if not np.array_equal(future_pos, original_positions):
#     #             booleans.append(is_position_valid(future_pos, existing_positions, min_dist))
    
#     future_placement_possible = future_placement_possible and np.all(booleans)        

def check_future_placement(state, movement_distance):
    """
    Use the current state to predict whether we can place a new object and move an existing object
    state: passed as a numpy array of shape (num_objects, positions)
    Check whether we can 
    a) place a new object:
    b) move all existing object
    c) The knife is far enough away from the plate
    """
    future_placement_possible = False
    future_movement_possible = False
    knife_away_from_plate = False
    # If all objects are on the counter, we don't care about future placement
    # The MovablePositionDict contains all the objects that can be moved and
    # if the counter_position is None, then the object is not on the counter
    num_objects_on_counter = sum([pos['counter_position'] is not None for pos in MOVABLE_POSITION_DICT])
    if num_objects_on_counter == len(MOVABLE_POSITION_DICT):
        future_placement_possible = True
    else:
        # Check if we can place a new object
        possible_positions = generate_possible_positions(0.65, 1.00, -0.81, -0.35, GRID_STRIDE)
        for future_pos in possible_positions:
            if is_position_valid(future_pos, state, MIN_DIST):
                future_placement_possible = True
                break
    
    # Check if we can move all of the existing objects
    booleans = [False for _ in range(len(state))]
    for i, pos in enumerate(state):
        # Create list of possible positions around original position
        min_x, max_x = max(pos[0] - 0.2, 0.65), min(pos[0] + 0.2, 1.00)
        min_z, max_z = max(pos[1] - 0.2, -0.81), min(pos[1] + 0.2, -0.35)
        possible_positions_move = generate_possible_positions(min_x, max_x, min_z, max_z, GRID_STRIDE)
        # Check if we can move the object to any of the possible positions
        for future_pos in possible_positions_move:
            if not np.array_equal(future_pos, pos):
                if is_position_valid(future_pos, state, movement_distance):
                    booleans[i] = True
                    break
        future_movement_possible = np.all(booleans)
        if future_movement_possible:
            break

    
    return future_placement_possible and future_movement_possible
    

def plot_possible_positions_on_frame(frame, possible_positions, existing_positions):

    fig, ax = plt.subplots()
    frame = frame.copy()
    possible_positions_proj = ((possible_positions + 1.0) / 2.0 * 512).astype(np.int32)
    existing_positions_proj = ((existing_positions + 1.0) / 2.0 * 512).astype(np.int32)
    frame[possible_positions_proj[:, 0], possible_positions_proj[:, 1]] = [0, 255, 0]
    rects = [patches.Rectangle((pos[1] - 5, pos[0] - 5), 10, 10, linewidth=1, edgecolor='r', facecolor='none') for pos in existing_positions_proj]
    for rect in rects:
        ax.add_patch(rect)
    # frame[existing_positions_proj[:, 0], existing_positions_proj[:, 1]] = [255, 0, 0]
    plt.imshow(frame)

    plt.show()

def perform_action(controller : Controller, action_type : str, object_name : str, event : dict, step_number : int = -1):
    print((f"[{step_number:3d}] " if step_number >= 0 else "") + f"Performing action {action_type} on {object_name}")
    objectId, obj = get_object_id(event, object_name)
    action_pos = get_action_pos(event, objectId, action_type, obj=obj)
    if action_pos is not None:
        if action_type == 'PickupObject':
            event = controller.step(action='PickupObject',
                                    objectId=objectId)
            orig_pos = None
            for mov_pos in MOVABLE_POSITION_DICT:
                if object_name.startswith(mov_pos['objectName']):
                    if mov_pos['counter_position'] is not None:
                        orig_pos = mov_pos['counter_position']['position']
                        mov_pos['counter_position']['objectName'] = None
                        mov_pos['counter_position'] = None
                    break
            if orig_pos is None:
                PICKUP['position'] = (np.random.uniform(-0.15, 0.15), np.random.uniform(0.08, 0.15))
            else:
                right_init = ((orig_pos['z'] + 0.81) / (0.81 - 0.35)) * 0.3 - 0.15
                up_init = ((orig_pos['x'] - 0.65) / (1.00 - 0.65)) * 0.07 + 0.08
                PICKUP['position'] = (np.clip(-right_init + np.random.rand() * 0.05, -0.15, 0.15),
                                      np.clip(up_init + np.random.rand() * 0.02, 0.08, 0.15))
            event = controller.step(action="MoveHeldObject",
                                    ahead=0.0,
                                    right=PICKUP['position'][0],
                                    up=PICKUP['position'][1],
                                    forceVisible=False)
        elif action_type == 'PutObject':
            picked_object = None
            for sub_obj in event.metadata['objects']:
                if sub_obj['isPickedUp']:
                    picked_object = sub_obj
                    break
            if picked_object is None:
                print('No object picked up')
                return event, None
            event = controller.step(action='PutObject',
                                    objectId=objectId)
            PICKUP['position'] = None
            if object_name.startswith('CounterTop'):
                event, _ = perform_action(controller, action_type='MoveObject', object_name=picked_object['name'], event=event, step_number=step_number)
            elif object_name.startswith('Microwave'):
                pass
        elif action_type == 'MoveObject':
            is_knife, is_plate = object_name.startswith('Knife'), object_name.startswith('Plate')
            orig_pos = None
            for mov_pos in MOVABLE_POSITION_DICT:
                if object_name.startswith(mov_pos['objectName']):
                    orig_pos = mov_pos
                    break
            total_objects = len(MOVABLE_POSITION_DICT)

            # Move from counter to counter
            if orig_pos['counter_position'] is not None:

                original_positions = orig_pos['counter_position']['position']['x'], orig_pos['counter_position']['position']['z']
                # Create list of possible positions around original position
                counter_min_x, counter_max_x = max(original_positions[0] - 0.2, 0.65), min(original_positions[0] + 0.2, 1.00)
                counter_min_z, counter_max_z = max(original_positions[1] - 0.2, -0.81), min(original_positions[1] + 0.2, -0.35)
                possible_positions = generate_possible_positions(counter_min_x, counter_max_x, counter_min_z, counter_max_z, GRID_STRIDE)
                existing_positions = [pos['counter_position']['position'] 
                                            for pos in MOVABLE_POSITION_DICT 
                                            if pos['counter_position'] is not None]
                existing_positions = np.array([list(pos.values()) for pos in existing_positions])
                existing_positions = existing_positions[:, [0, 2]]
                valid_positions = []
                for pos in possible_positions:
                    # Check if knife is far enough away from plate
                    if is_position_valid(pos, existing_positions, MIN_DIST, is_knife, is_plate):
                        # Predictive analysis for future placement
                        # We moved an object from the counter to the counter, so we need to check
                        # at this new state whether we can place a new object and move an existing object
                        existing_positions_without_current = np.delete(existing_positions, np.where(np.all(existing_positions == original_positions, axis=1)), axis=0)
                        new_state = np.concatenate([existing_positions_without_current, [pos]])
                        if check_future_placement(new_state, 0.22):
                            valid_positions.append(pos)

                # plot_possible_positions_on_frame(event.frame, possible_positions, existing_positions)
                if len(valid_positions) > 0:
                    x, z = valid_positions[np.random.randint(0, len(valid_positions))]
                else:
                    print('No valid positions found')
                    raise Exception('No valid positions found')
            else: # Move from hand to counter

                # Generate list of possible positions
                counter_min_x, counter_max_x = 0.65, 1.00
                counter_min_z, counter_max_z = -0.81, -0.35
                possible_positions = generate_possible_positions(counter_min_x, counter_max_x, counter_min_z, counter_max_z, GRID_STRIDE)

                # Get existing positions from MOVABLE_POSITION_DICT
                existing_positions = [pos['counter_position']['position'] 
                                            for pos in MOVABLE_POSITION_DICT 
                                            if pos['counter_position'] is not None]
                existing_positions = np.array([list(pos.values()) for pos in existing_positions])
                # Throw away y coordinate
                existing_positions = existing_positions[:, [0, 2]]

                # Filter valid positions
                valid_positions = []
                for pos in possible_positions:
                    if is_position_valid(pos, existing_positions, MIN_DIST, is_knife, is_plate):
                        # Predictive analysis for future placement
                        # We moved an object from the hand to the counter, so we need to check
                        # at this new state whether we can place a new object and move an existing object
                        new_state = np.concatenate([existing_positions, [pos]])
                        if check_future_placement(new_state, 0.2):
                            valid_positions.append(pos)

                if len(valid_positions) > 0:
                    x, z = valid_positions[np.random.randint(0, len(valid_positions))]
                else:
                    print('No valid positions found')
                    raise Exception('No valid positions found')
                    # return event, None

            for mov_pos in MOVABLE_POSITION_DICT:
                if object_name.startswith(mov_pos['objectName']):
                    mov_pos['counter_position'] = {'position': {'x': x, 'y': 0.98, 'z': z}, 'objectName': object_name}
                    event = controller.step(
                        action="PlaceObjectAtPoint",
                        objectId=objectId,
                        position=mov_pos['counter_position']['position'],
                        rotation=mov_pos['rotation']
                    )
                    break
            
            if PICKUP['position'] is not None:
                event = controller.step(action="MoveHeldObject",
                                        ahead=0.0,
                                        right=PICKUP['position'][0],
                                        up=PICKUP['position'][1],
                                        forceVisible=False)
        elif action_type == 'ToggleObject':
            event = controller.step(action='ToggleObjectOn' if not obj['isToggled'] else 'ToggleObjectOff',
                                    objectId=objectId)
        elif action_type == 'SliceObject':
            event = controller.step(action='SliceObject',
                                    objectId=objectId)
            if object_name in ACTIONS['PickupObject']:
                ACTIONS['PickupObject'].remove(object_name)
            ACTIONS['SliceObject'].remove(object_name)
        elif action_type == 'OpenObject':
            if not obj['isOpen']:
                event = controller.step(action='OpenObject',
                                        objectId=objectId,
                                        openness=1.0)
            else:
                event = controller.step(action='CloseObject',
                                        objectId=objectId)
        elif action_type == 'NoOp':
            event = controller.step(action='Stand')
    return event, action_pos


def perform_random_action(controller : Controller, last_step : dict):
    possible_actions = deepcopy(ACTIONS)
    microwave = get_object_id(last_step['event'], 'Microwave')[1]
    if microwave['isOpen']:
        # When microwave is open, we cannot toggle it on
        possible_actions['ToggleObject'].remove('Microwave')
    elif microwave['isToggled']:
        # When microwave is toggled on, we cannot open it
        possible_actions['OpenObject'].remove('Microwave')
    if last_step['pickedObject'] == None:
        possible_actions.pop('SliceObject')
        possible_actions.pop('PutObject')
        possible_actions['PickupObject'] *= 2
    else:
        # possible_actions.pop('PickupObject')
        possible_actions['PickupObject'].remove(last_step['pickedObject'])
        possible_actions['MoveObject'] = possible_actions.pop('PickupObject') * 2
        if (not microwave['isOpen'] or not last_step['pickedObject'].startswith('Plate')) and 'Microwave' in possible_actions['PutObject']:
            possible_actions['PutObject'].remove('Microwave')
        if not last_step['pickedObject'].startswith('Knife'):
            possible_actions.pop('SliceObject')
            if last_step['pickedObject'].startswith('Egg') and 'Pan' in possible_actions['PutObject']:
                if last_step['step_number'] < 40:
                    possible_actions['PutObject'].remove('Pan')
                else:
                    possible_actions['PutObject'] += ['Pan'] * 2
            elif 'Pan' in possible_actions['PutObject']:
                possible_actions['PutObject'].remove('Pan')
        else:
            possible_actions['PutObject'] = possible_actions['PutObject'][-1:] * 2
            if last_step['step_number'] < 50:
                possible_actions['SliceObject'].remove('Apple')
            possible_actions['SliceObject'] *= 2
        possible_actions['PutObject'] *= 1
        possible_actions['PutObject'] += possible_actions['PutObject'][-1:] * 2
    possible_actions = [(action_type, object_name) for action_type, object_names in possible_actions.items() for object_name in object_names]
    # Some actions are invalid, so we try to perform an action until we get a valid one
    action_pos = None
    action_counter = 0
    while action_pos is None and action_counter < 20:
        action_type, object_name = possible_actions[np.random.randint(0, len(possible_actions))]
        event, action_pos = perform_action(controller, action_type, object_name, last_step['event'], step_number=last_step['step_number']+1)
        action_counter += 1
    assert action_pos is not None, 'Could not perform any action'
    if last_step['pickedObject'] is not None and \
        last_step['pickedObject'].startswith('Egg') and \
            action_type == 'PutObject' and object_name == 'Pan' and event.metadata['lastActionSuccess']:
        event = controller.step(action='BreakObject',
                                objectId=get_object_id(event, 'Egg')[0])
        ACTIONS['PickupObject'].remove('Egg')
    # Advance physics time step
    event = controller.step(action='Stand')
    event = controller.step(action='Stand')

    pickedObject = None
    if action_type == 'PickupObject':
        pickedObject = object_name
    elif action_type == 'PutObject':
        pickedObject = None
    else:
        pickedObject = last_step['pickedObject']

    description = f'{action_type} {object_name}'
    new_step = {
        'event': event,
        'pickedObject': pickedObject,
        'action_pos': action_pos,
        'debug_info': {
            'action_type': action_type, 
            'object_name': object_name,
            'pickedObject': pickedObject,
            'action_counter': action_counter
        },
        'step_number': last_step['step_number'] + 1,
        'description': description
    }
    return new_step

def create_targets(debug_info : dict, causal_keys : list, latents: np.ndarray):
    targets = np.zeros((len(debug_info['action_type']) + 1, len(causal_keys)), dtype=np.uint8)
    for i in range(len(debug_info['action_type'])):
        object_name = debug_info['object_name'][i]
        action_type = debug_info['action_type'][i]
        if action_type == 'PutObject':
            object_name = debug_info['pickedObject'][i - 1]
        elif action_type == 'NoOp':
            continue
        for j, key in enumerate(causal_keys):
            if key.startswith(object_name):
                if action_type == 'OpenObject' and not key.endswith('_open'):
                    continue
                if action_type == 'ToggleObject' and not key.endswith('_on'):
                    continue
                if action_type in ['PickupObject', 'PutObject'] and \
                    not (key.endswith('_pickedup') or any([key.endswith(f'_center_{pf}') for pf in ['x', 'y', 'z']])):
                    continue
                if action_type == 'SliceObject' and not key.endswith('_sliced'):
                    continue
                targets[i + 1, j] = 1
    # Fix egg broken and egg cooked based on latents
    egg_broken_idx = causal_keys.index('Egg_afaaaca3_broken')
    egg_cooked_idx = causal_keys.index('Egg_afaaaca3_cooked')
    egg_broken_ = latents[:, egg_broken_idx].argmax()
    egg_cooked_ = latents[:, egg_cooked_idx].argmax()
    targets[:, egg_broken_idx] = 0
    targets[:, egg_cooked_idx] = 0
    targets[egg_broken_, egg_broken_idx] = 1 if egg_broken_ > 0 else 0
    targets[egg_cooked_, egg_cooked_idx] = 1 if egg_cooked_ > 0 else 0

    return targets

def simplify_latents(latents : np.ndarray, causal_keys : list):
    latents_dict = {key: latents[:, i] for i, key in enumerate(causal_keys)}

    apple_slice_keys = [k for k in causal_keys if k.startswith('Apple_10_Sliced')]
    num_apple_slices = len(set([k.split('_')[3] for k in apple_slice_keys]))
    if num_apple_slices > 0:
        set_sliced = False
        for key in apple_slice_keys:
            if not set_sliced and key.endswith('center_x'):
                latents_dict['Apple_f33eaaa0_sliced'] += (latents_dict[key] != 0.0).astype(np.float32)
                set_sliced = True
            orig_key = 'Apple_f33eaaa0_' + key.split('_', 4)[-1]
            latents_dict[orig_key] += latents_dict[key] * (1 / num_apple_slices)
            latents_dict.pop(key)
    
    egg_cracked_keys = [k for k in causal_keys if k.startswith('Egg_Cracked')]
    set_broken = False
    for key in egg_cracked_keys:
        if not set_broken and key.endswith('center_x'):
            latents_dict['Egg_afaaaca3_broken'] += (latents_dict[key] != 0.0).astype(np.float32)
            set_broken = True
        orig_key = 'Egg_afaaaca3_' + key.split('_', 3)[-1]
        if orig_key not in latents_dict:
            latents_dict[orig_key] = latents_dict[key]
        else:
            latents_dict[orig_key] += latents_dict[key]
        latents_dict.pop(key)
    if 'Egg_afaaaca3_cooked' not in latents_dict:
        latents_dict['Egg_afaaaca3_cooked'] = np.zeros_like(latents_dict['Egg_afaaaca3_center_x'])

    potato_keys = [k for k in causal_keys if k.startswith('Potato')]
    set_sliced = False
    for key in potato_keys:
        if key.endswith('sliced'):
            continue
        elif not set_sliced and key.endswith('center_x') and '_Slice_' in key:
            latents_dict['Potato_e4559da4_sliced'] = (latents_dict[key] != 0.0).astype(np.float32)
            set_sliced = True
        latents_dict.pop(key)
    
    plate_keys = [k for k in causal_keys if k.startswith('Plate')]
    for t in range(1, latents.shape[0]):
        if latents_dict[plate_keys[0]][t] == 0.0: # center x being zero - in Microwave
            for k in plate_keys:
                latents_dict[k][t] = latents_dict[k][t-1]
    
    causal_keys = sorted(list(latents_dict.keys()))
    latents = np.stack([latents_dict[key] for key in causal_keys], axis=1)
    return latents, causal_keys

def downscale_images(images : np.ndarray):
    images = torch.from_numpy(images)
    images = images.permute(0, 3, 1, 2)
    images = F.interpolate(images.float(), 
                           scale_factor=(0.5, 0.5),
                           mode='bilinear')
    images = images.permute(0, 2, 3, 1)
    images = images.numpy()
    # To numpy uint8
    images = images.astype(np.uint8)
    return images

def generate_sequence(seed : int, output_folder : str, num_frames : int = 200, prefix : str = "", save_segmentations : bool = False, randomize_materials : bool = False, randomize_colors : bool = False, randomize_lighting : bool = False):
    print('-> Seed:', seed)
    attempts = 0
    success = False
    while not success and attempts < 20:
        try:
            controller, event = initialize_environment(seed)
            success = True
        except Exception as e:
            attempts += 1
            print(e)
            continue
    assert success, 'Failed to initialize environment after 10 attempts'
    last_step = {'event': event, 'pickedObject': None, 'step_number': 0}
    collected_frames = np.zeros((num_frames, RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    collected_actions = np.zeros((num_frames, 2), dtype=np.float32)
    collected_states = defaultdict(lambda : np.zeros((num_frames,), dtype=np.float32))
    collected_segmentations = dict()
    collected_descriptions = []
    debug_info = defaultdict(lambda : [])
    collected_frames[0] = event.frame
    for key, val in get_environment_state(event).items():
        collected_states[key][0] = val
    if save_segmentations:
        for key, val in get_object_segmentations(event).items():
            collected_segmentations[key] = np.zeros((num_frames, *val.shape), dtype=val.dtype)
            collected_segmentations[key][0] = val
    if randomize_materials:
        controller.step(
            action='RandomizeMaterials',
            useTrainMaterials=True if prefix.startswith('train') else False,
            useValMaterials=True if prefix.startswith('val') else False,
            useTestMaterials=True if prefix.startswith('test') else False,
            inRoomTypes='Kitchen',
        )
    if randomize_colors:
        controller.step(action="RandomizeColors")
    if randomize_lighting:
        controller.step(
            action="RandomizeLighting",
            brightness=(0.5, 1.5),
            randomizeColor=True,
            hue=(0, 1),
            saturation=(0.5, 1),
            synchronized=False
        )
    for i in range(1,num_frames):
        last_step = perform_random_action(controller, last_step)
        collected_frames[i] = last_step['event'].frame
        collected_actions[i] = last_step['action_pos']
        collected_descriptions.append(last_step['description'])
        for key, val in get_environment_state(last_step['event']).items():
            collected_states[key][i] = val
        for key, val in last_step['debug_info'].items():
            debug_info[key].append(val)
        if save_segmentations:
            for key, val in get_object_segmentations(last_step['event']).items():
                collected_segmentations[key][i] = val

    controller.stop()
    collected_frames = downscale_images(collected_frames)
    causal_keys = sorted(list(collected_states.keys()))
    latents = np.stack([collected_states[key] for key in causal_keys], axis=1)
    latents, causal_keys = simplify_latents(latents, causal_keys)
    targets = create_targets(debug_info, causal_keys, latents)
    np.savez_compressed(os.path.join(output_folder, f'{prefix}seq_{str(seed).zfill(6)}.npz'), 
                        frames=collected_frames.transpose(0, 3, 1, 2), 
                        actions=collected_actions, 
                        latents=latents, 
                        targets=targets,
                        causal_keys=causal_keys,
                        collected_descriptions=collected_descriptions,
                        **{'segm_'+k: collected_segmentations[k] for k in collected_segmentations})
    debug_info['causal_keys'] = causal_keys
    debug_info['seed'] = int(seed)
    with open(os.path.join(output_folder, f'{prefix}seq_{str(seed).zfill(6)}_infos.json'), 'w') as f:
        json.dump(debug_info, f, indent=4)
    return collected_frames, collected_actions, collected_states

def print_time(time : float):
    string = ''
    if time > 3600:
        string += f'{int(time // 3600)}h '
        time = time % 3600
    if time > 60:
        string += f'{int(time // 60)}m '
        time = time % 60
    string += f'{int(time)}s'
    return string


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, default='/scratch-shared/gkounto/biscuit/data/ithor_eggfix/')
    parser.add_argument('--num_frames', type=int, default=100)
    parser.add_argument('--num_sequences', type=int, default=300)
    parser.add_argument('--prefix', type=str, default='train')    
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--save_segmentations', action='store_true', default=False)
    parser.add_argument('--randomize_materials', action='store_true', default=False)
    parser.add_argument('--randomize_colors', action='store_true', default=False)
    parser.add_argument('--randomize_lighting', action='store_true', default=False)
    parser.add_argument('--max_processes', type=int, default=18)
    parser.add_argument('--offset', type=int, default=0, help='Offset for sequence generation')
    parser.add_argument('--chunk_size', type=int, default=18, help='Number of sequences to generate in this run')

    args = parser.parse_args()

    output_folder = args.output_folder
    num_frames = args.num_frames
    num_sequences = args.num_sequences
    prefix = args.prefix
    overwrite = args.overwrite
    save_segmentations = args.save_segmentations
    randomize_materials = args.randomize_materials
    randomize_colors = args.randomize_colors
    randomize_lighting = args.randomize_lighting
    offset = args.offset
    chunk_size = args.chunk_size
    output_folder = os.path.join(output_folder, prefix)
    hash = sha256(prefix.encode())
    offset_seed = np.frombuffer(hash.digest(), dtype='uint32')[0]

    orig_mov_position = MOVABLE_POSITION_DICT
    orig_counter_positions = COUNTER_POSITIONS
    orig_actions = ACTIONS
    orig_pickup = PICKUP
    os.makedirs(output_folder, exist_ok=True)

    seeds = [int(offset_seed + i) for i in range(1 + offset, 1 + offset + chunk_size)]

    def wrapper(seq_idx, seed):
        MOVABLE_POSITION_DICT = deepcopy(orig_mov_position)
        COUNTER_POSITIONS = deepcopy(orig_counter_positions)
        ACTIONS = deepcopy(orig_actions)
        PICKUP = deepcopy(orig_pickup)
        out_file = os.path.join(output_folder, f'{prefix}_seq_{str(seed).zfill(6)}.npz')
        if os.path.exists(out_file) and not overwrite:
            return
        print(f'Generating sequence {seq_idx} of {num_sequences}...')
        generate_sequence(seed, output_folder, num_frames, prefix, save_segmentations, randomize_materials, randomize_colors, randomize_lighting)

    max_processes = args.max_processes
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_processes) as executor:

        futures = {executor.submit(wrapper, seq_idx, seed): seq_idx for seq_idx, seed in zip(range(1 + offset, 1 + offset + chunk_size), seeds)}

        for future in concurrent.futures.as_completed(futures):
            seq_idx = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Sequence {seq_idx} generated an exception: {e}")