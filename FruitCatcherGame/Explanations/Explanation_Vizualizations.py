import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap
import pygame
from game import Environment
import matplotlib.pyplot as plt

def load_trained_model(model_path):
    return load_model(model_path)

def visualize_state(state, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(state[0, :, :, 0], cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.colorbar()

def visualize_shap(state, shap_values, action):
    action_names = ['Stay', 'Move Left', 'Move Right']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original state
    ax1.imshow(state[0, :, :, 0], cmap='viridis')
    ax1.set_title("Original State")
    ax1.axis('off')
    
    # SHAP values
    if isinstance(shap_values, list):
        action_shap = shap_values[action][0]
    else:
        action_shap = shap_values[0, ..., action]
    
    im = ax2.imshow(action_shap, cmap='coolwarm', vmin=-np.max(np.abs(action_shap)), vmax=np.max(np.abs(action_shap)))
    ax2.set_title(f"SHAP Values for {action_names[action]}")
    ax2.axis('off')
    
    fig.colorbar(im, ax=ax2, label='SHAP value')
    plt.tight_layout()
    plt.show()

def explain_action_shap(model, state, explainer):
    shap_values = explainer.shap_values(state)
    action = np.argmax(model.predict(state)[0])
    action_names = ['Stay', 'Move Left', 'Move Right']

    print(f"Chosen action: {action_names[action]}")
    print("SHAP values for the chosen action:")

    if isinstance(shap_values, list):
        action_shap = shap_values[action][0]
    else:
        action_shap = shap_values[0, ..., action]

    flat_shap = action_shap.flatten()
    top_features = np.argsort(np.abs(flat_shap))[-3:]

    for feature in top_features:
        row, col = feature // state.shape[2], feature % state.shape[2]
        value = flat_shap[feature]

        print(f"Position ({row}, {col}): {value:.4f}")
        if value > 0:
            print(f"  This position positively influenced the decision to {action_names[action]}.")
        else:
            print(f"  This position negatively influenced the decision to {action_names[action]}.")

        if state[0, row, col, 0] == 1:
            print("  There is a fruit at this position.")
        elif state[0, row, col, 0] == 2:
            print("  The player is at this position.")
        else:
            print("  This position is empty.")

    visualize_shap(state, shap_values, action)
    return action

def visualize_counterfactual(original_state, counterfactual, changes):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original state
    ax1.imshow(original_state[0, :, :, 0], cmap='viridis')
    ax1.set_title("Original State")
    ax1.axis('off')
    
    # Counterfactual state
    ax2.imshow(counterfactual[0, :, :, 0], cmap='viridis')
    ax2.set_title("Counterfactual State")
    ax2.axis('off')
    
    # Changes
    im = ax3.imshow(changes[0, :, :, 0], cmap='coolwarm', vmin=-np.max(np.abs(changes)), vmax=np.max(np.abs(changes)))
    ax3.set_title("Changes")
    ax3.axis('off')
    
    fig.colorbar(im, ax=ax3, label='Change in value')
    plt.tight_layout()
    plt.show()

def explain_action_counterfactual(model, state, num_iterations=10, step_size=0.1):
    original_prediction = model.predict(state)[0]
    original_action = np.argmax(original_prediction)
    action_names = ['Stay', 'Move Left', 'Move Right']

    counterfactual = np.copy(state)
    
    print(f"Original action: {action_names[original_action]}")
    print("Counterfactual explanation:")

    for iteration in range(num_iterations):
        perturbation = np.random.normal(0, step_size, size=counterfactual.shape)
        perturbed = np.clip(counterfactual + perturbation, 0, 1)
        new_prediction = model.predict(perturbed)[0]
        new_action = np.argmax(new_prediction)
        
        if new_action != original_action:
            counterfactual = perturbed
            break
        
        if np.sum(np.abs(new_prediction - original_prediction)) > np.sum(np.abs(model.predict(counterfactual)[0] - original_prediction)):
            counterfactual = perturbed

    if new_action != original_action:
        print(f"Found a counterfactual that changes the action to: {action_names[new_action]}")
        print(f"Changes required to change the action (top 3 most significant):")
        
        changes = counterfactual - state
        flat_changes = changes.flatten()
        top_changes = np.argsort(np.abs(flat_changes))[-3:]
        
        for change in top_changes:
            row, col = change // state.shape[2], change % state.shape[2]
            value = flat_changes[change]

            print(f"Position ({row}, {col}): {value:.4f}")
            print(f"  {'Increase' if value > 0 else 'Decrease'} the value at this position to change the action.")
            print(f"  Originally: {'fruit' if state[0, row, col, 0] == 1 else 'player' if state[0, row, col, 0] == 2 else 'empty'}")
            print(f"  Counterfactual: {'fruit' if counterfactual[0, row, col, 0] == 1 else 'player' if counterfactual[0, row, col, 0] == 2 else 'empty'}")

        visualize_counterfactual(state, counterfactual, changes)
    else:
        print("Could not find a counterfactual example within the given number of iterations.")

    return original_action

def run_model_with_explanations(model_path, num_games=5):    
    model = load_trained_model(model_path)
    env = Environment()

    sample_state = env.reset()
    sample_state = np.reshape(sample_state, (1,) + sample_state.shape + (1,))

    background = np.zeros_like(sample_state)
    explainer = shap.DeepExplainer(model, background)

    pygame.init()
    screen = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    for game in range(num_games):
        state = env.reset()
        state = np.reshape(state, (1,) + state.shape + (1,))
        done = False
        score = 0

        print(f"Starting game {game + 1}")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            print("SHAP Explanation:")
            action_shap = explain_action_shap(model, state, explainer)
            
            print("Counterfactual Explanation:")
            action_cf = explain_action_counterfactual(model, state)
            
            action = action_shap
            
            step_result = env.step(action)
            
            if len(step_result) == 4:
                next_state, reward, done, _ = step_result
            elif len(step_result) == 3:
                next_state, reward, done = step_result
            else:
                raise ValueError(f"Unexpected number of return values from env.step(): {len(step_result)}")
            
            next_state = np.reshape(next_state, (1,) + next_state.shape + (1,))

            score += reward
            state = next_state

            env.render(screen)
            pygame.display.flip()
            clock.tick(0.1)

        print(f"Game {game + 1} finished with score: {score}\n")

    pygame.quit()

if __name__ == "__main__":
    model_path = "models\FruitCatcher100.h5"
    run_model_with_explanations(model_path)