import transformers
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer


def translate_action_sequence(action_sequence):
    action_translations = {
        "NoOp": "did nothing with",
        "OpenObject": "adjusted",
        "ToggleObject": "toggled",
        "PickupObject": "picked up",
        "PutObject": "placed",
        "MoveObject": "moved"
    }

    base_object_translations = {
        "NoObject1": "no particular object",
        "NoObject2": "no particular object",
        "NoObject3": "no particular object",
        "Microwave": "the microwave",
        "Toaster": "the toaster",
        "Cabinet_47fc321b": "a wooden cabinet",
        "StoveKnob_38c1dbc2": "the first stove knob for the front-right burner",
        "StoveKnob_690d0d5d": "the second stove knob for the front-left burner",
        "StoveKnob_c8955f66": "the third stove knob for the back-left burner",
        "StoveKnob_cf670576": "the fourth stove knob for the back-right burner",
        "Plate": "the ceramic plate",
        "Egg": "the spherical, brown, fragile Egg",
        "Pan": "the flat, metal, sturdy Pan",
        "CounterTop_f8092513": "the granite countertop"
    }

    holding_object = None

    human_readable_sequence = []

    for action_command in action_sequence:
        words = action_command.split(" ")
        action = words[0]
        obj = words[-1]

        translated_action = action_translations.get(action, "did something with")
        translated_object = base_object_translations.get(obj, "an unknown object")

        # Modify the object description based on the action
        if obj == "Microwave":
            if action == "OpenObject":
                translated_object = "the microwave's door"
            elif action == "ToggleObject":
                translated_object = "the microwave's heating element"

        if action == "PutObject" and holding_object is not None:
            description = f"You {translated_action} {holding_object} on {translated_object}"
        else:
            description = f"You {translated_action} {translated_object}"

        human_readable_sequence.append(description)

        if action == "PickupObject":
            holding_object = translated_object
        elif action in ["PutObject"]:
            holding_object = None

    return human_readable_sequence

def tokenize_action_sequence(action_sequence, tokenizer):
    tokenized_sequence = []
    for action_description in action_sequence:
        tokenized_sequence.append(tokenizer(action_description, padding=True, truncation=True, return_tensors='np'))
    return tokenized_sequence

def tokenize_action_sequence_clip(action_sequence, tokenizer):
    tokenized_sequence = []
    for action_description in action_sequence:
        tokenized_sequence.append(tokenizer(action_description, context_length=64))
    return tokenized_sequence

def process_single_file(file, tokenizer):
    data = dict(np.load(file, allow_pickle=True))
    action_sequence = data['collected_descriptions']
    action_sequence = np.concatenate([np.array(['NoOp NoObject1']), action_sequence])
    tokenized_sequence = tokenize_action_sequence(translate_action_sequence(action_sequence), tokenizer)
    max_len = max([len(seq['input_ids'][0]) for seq in tokenized_sequence])
    
    data['input_ids'] = np.stack([np.concatenate([seq['input_ids'][0], np.zeros(max_len - len(seq['input_ids'][0]), dtype=np.int32)]) for seq in tokenized_sequence], axis=0)
    data['token_type_ids'] = np.stack([np.concatenate([seq['token_type_ids'][0], np.zeros(max_len - len(seq['token_type_ids'][0]), dtype=np.int32)]) for seq in tokenized_sequence], axis=0)
    data['attention_mask'] = np.stack([np.concatenate([seq['attention_mask'][0], np.zeros(max_len - len(seq['attention_mask'][0]), dtype=np.int32)]) for seq in tokenized_sequence], axis=0)
    # del data['collected_descriptions']
    
    np.savez_compressed(file.strip('.npz'), **data)


def process_single_file_clip(file, tokenizer):
    data = dict(np.load(file, allow_pickle=True))
    action_sequence = data['collected_descriptions']
    action_sequence = np.concatenate([np.array(['NoOp NoObject1']), action_sequence])
    tokenized_sequence = tokenize_action_sequence_clip(translate_action_sequence(action_sequence), tokenizer)
    input_ids = np.stack(tokenized_sequence, axis=0).squeeze(1)
    
    data['input_ids'] = input_ids
    data['token_type_ids'] = np.zeros_like(input_ids)
    data['attention_mask'] = np.ones_like(input_ids)
    np.savez_compressed(file.strip('.npz'), **data)


def process_dataset(path, tokenizer):
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".npz"):
                files.append(os.path.join(root, filename))
                
    with ThreadPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(lambda file: process_single_file(file, tokenizer), files), total=len(files)))

if __name__ == '__main__':
    tokenizer = transformers.BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')

    path = '/scratch-shared/gkounto/biscuit/data/ithor_new'
    process_dataset(path=path + '/train', tokenizer=tokenizer)
    process_dataset(path=path + '/val', tokenizer=tokenizer)
    process_dataset(path=path + '/val_indep', tokenizer=tokenizer)
    process_dataset(path=path + '/test', tokenizer=tokenizer)
    process_dataset(path=path + '/test_indep', tokenizer=tokenizer)