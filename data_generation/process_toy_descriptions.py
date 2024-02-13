import numpy as np
import json
import os
import webcolors
from concurrent.futures import ThreadPoolExecutor
import tqdm
from open_clip import get_tokenizer
import random

# Existing functions: closest_color, get_color_name, load_metadata remain unchanged

# PCFG and description generation logic
GRAMMAR = {
    "ADJECTIVES": {
        "traffic light": [
            ("sturdy, illuminated", 0.2), ("metal, tall", 0.2), ("gleaming, automated", 0.2), ("durable, weather-resistant", 0.2)
        ],
        "vehicle": [
            ("sleek, aerodynamic", 0.2), ("speedy, high-performance", 0.2), ("luxurious, comfortable", 0.2), ("compact, fuel-efficient", 0.2), ("rugged, all-terrain", 0.2)
        ],
        "obstacle": [
            ("solid, rocky", 0.2), ("visible, brightly colored", 0.2), ("large, obstructive", 0.2), ("big, heavy", 0.2),
        ]
    },
    "ACTION_MODIFIER": [
        ("skillfully", 0.2), ("efficiently", 0.2), ("carefully", 0.2), ("precisely", 0.2), ("quickly", 0.2)
    ]
}

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_tuple):
    try:
        color_name = webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        color_name = closest_color(rgb_tuple)
    return color_name

def load_metadata(path):
    with open(path, 'r') as file:
        metadata = json.load(file)
    return metadata


def weighted_choice(choices):
    phrases, weights = zip(*choices)
    total = sum(weights)
    probs = [w / total for w in weights]
    return np.random.choice(phrases, p=probs)

def generate_description_probabilistic(action, object_type, color, grammar):
    num_adjectives = random.randint(1, 3)
    adjectives_list = grammar["ADJECTIVES"][object_type]
    adjectives = ', '.join(weighted_choice(adjectives_list) for _ in range(num_adjectives))
    action_modifier = weighted_choice(grammar["ACTION_MODIFIER"])

    full_description = f"You {action_modifier} {action} the {adjectives}, {color} {object_type}."
    return full_description

def translate_description(description, metadata, grammar):
    # Create a dictionary to map RGB tuples to color names for all object types
    object_names = {name: tuple(map(int, name.split('_')[1][1:-1].split(', '))) for name in metadata['object_names']}
    color_names = {name: get_color_name(rgb) for name, rgb in object_names.items()}
    
    # Attempt to find a matching description based on the action and object type in the description
    for action in ['turned', 'changed the state of', 'moved']:
        if action in description:
            for name in metadata['object_names']:
                object_type_raw, rgb_str = name.split('_')[0], name.split('_')[1]
                object_type = 'traffic light' if object_type_raw == 'trafficlight' else object_type_raw
                rgb_tuple = tuple(map(int, rgb_str[1:-1].split(', ')))
                color_name = get_color_name(rgb_tuple)
                
                if object_type in description and color_name in description:
                    # Correctly generate description using the PCFG
                    # Note: Assuming 'action' passed here is already in the desired format for generation
                    # Adjust 'object_type_raw' as needed to fit the grammar's expected keys
                    return generate_description_probabilistic(action, object_type, color_name, grammar)
    return description  # Fallback to the original description if no translation was applied

def process_single_episode(file, tokenizer, metadata, grammar):
    data = dict(np.load(file, allow_pickle=True))
    descriptions = data['action_descriptions']  # Assuming 'descriptions' is the key for action descriptions in your dataset
    
    translated_descriptions = [translate_description(description, metadata, grammar) for description in descriptions]
    tokenized_descriptions = [tokenizer.tokenizer(description, return_token_type_ids=True, padding='max_length', max_length=64) for description in translated_descriptions]  # Adjust context_length if necessary
    input_ids = np.stack([desc['input_ids'] for desc in tokenized_descriptions], axis=0)
    attention_mask = np.stack([desc['attention_mask'] for desc in tokenized_descriptions], axis=0)
    token_type_ids = np.stack([desc['token_type_ids'] for desc in tokenized_descriptions], axis=0)
    
    # Update the dataset with the tokenized descriptions
    data['input_ids'] = input_ids
    data['attention_mask'] = attention_mask
    data['token_type_ids'] = token_type_ids
    
    # Save the updated data back to file
    np.savez_compressed(file.strip('.npz'), **data)

def process_dataset(path, split, tokenizer, grammar):
    metadata = load_metadata(f"{'/'.join(path.split('/')[:-1])}/{split}_metadata.json")
    
    files = [os.path.join(root, filename) 
             for root, dirs, filenames in os.walk(path) 
             for filename in filenames if filename.endswith(".npz")]
                
    with ThreadPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(lambda file: process_single_episode(file, tokenizer, metadata, grammar), files), total=len(files)))
    # for file in files:
    #     process_single_episode(file, tokenizer, metadata, grammar)

if __name__ == '__main__':
    tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')

    # Specify your dataset paths
    path = '/home/john/PhD/BISCUIT/data/gridworld_simplified_5c/'
    
    for split in ['train', 'val', 'test', 'test_indep', 'val_indep']:
        process_dataset(path, split, tokenizer, GRAMMAR)