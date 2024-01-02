import matplotlib.pyplot as plt
from gridworld import Gridworld
import numpy as np
from PIL import Image

def format_causal_dict(causal_dict):
    formatted_text = ""
    for key, value in causal_dict.items():
        formatted_text += f"{key}: {value}\n"
        if "position_y" in key:  # Add a line break after each entity's details
            formatted_text += "\n"
    return formatted_text

def visualize_episode(gridworld, frames, causals, actions, action_descriptions, interventions):
    for i in range(len(frames)):
        debug_causals = Gridworld.causal_vector_to_debug_dict(gridworld, causals[i])
        formatted_causals = format_causal_dict(debug_causals)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax1 = plt.subplot(1, 2, 1)  # Frame subplot
        ax1.imshow(frames[i])
        ax1.set_title(f"Frame {i+1}")
        ax1.axis('off')

        ax2 = plt.subplot(1, 2, 2)  # Textual information subplot
        ax2.axis('off')
        text_info = (
            f"Step {i+1}:\n\n"
            f"Causals (Formatted):\n{formatted_causals}\n"
            f"Action: {actions[i]}\n"
            f"Action Description: {action_descriptions[i]}\n"
            f"Interventions: {interventions[i]}"
        )
        ax2.text(0, 1, text_info, ha='left', va='top', fontsize=8, wrap=True)

        plt.tight_layout()
        plt.show()


# Load an episode
episode = np.load('/home/john/PhD/BISCUIT/data_generation/data/gridworld_small/val/gridworld_episode_1000.npz')
gridworld = Gridworld(8, 8, sprite_size=32)
frames = episode['frames']
causals = episode['causals']
actions = episode['actions']
action_descriptions = episode['action_descriptions']
interventions = episode['interventions']
visualize_episode(gridworld, frames, causals, actions, action_descriptions, interventions)