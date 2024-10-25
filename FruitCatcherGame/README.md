# Fruit Catcher Game

## 1. About the Game

The Fruit Catcher game is an engaging and interactive reinforcement learning environment where a player controls a basket that moves horizontally at the bottom of the screen. The objective is to catch falling fruits to score points. Each successful catch increases the score, while missed fruits decrease the score or can potentially end the game. This setup provides a platform to train and evaluate reinforcement learning models, focusing on optimizing the agent's decision-making to achieve the highest score.

![image](https://github.com/user-attachments/assets/1108b0bb-e90f-48f2-9167-48611105905d)  <!-- Replace 'link_to_game_image_here' with the actual URL for the game image in your repository -->

## 2. Reinforcement Learning Model Used

### Deep Q-Network (DQN)

For this project, we employed a Deep Q-Network (DQN), which is an algorithm that combines Q-Learning with deep neural networks. The DQN algorithm trains the agent to learn an optimal policy for maximizing rewards by approximating the Q-value function. The Q-value represents the expected cumulative reward of taking a specific action in a given state and following the optimal policy thereafter.

### Training Details

The DQN model was trained for different numbers of episodes to analyze its learning process:
- **1 Episode**: A preliminary training session with limited learning.
  - [Model File - 1 Episode](FruitCatcherGame/Explanations/models/Fruit_Catcher.h5)
- **10 Episodes**: A short training period with some learning.
  - [Model File - 10 Episodes](FruitCatcherGame/Explanations/models/FruitCatcher10.h5)
- **100 Episodes**: A longer training period allowing the model to better understand the game environment.
  - [Model File - 100 Episodes](FruitCatcherGame/Explanations/models/FruitCatcher100.h5)


## 3. Explanations Used

To understand why the trained RL agent makes certain decisions, we employed two eXplainable AI techniques: SHAP and Counterfactual explanations.

### (i) SHAP (SHapley Additive exPlanations)

SHAP is a method for interpreting the output of machine learning models. It assigns importance scores to input features based on their contribution to the model's prediction. In the Fruit Catcher game, SHAP values help in identifying which aspects of the game state influence the agent's decisions.

![image](https://github.com/user-attachments/assets/e23715aa-fda7-4083-ad11-d9421551e9ef)  <!-- Replace 'link_to_shap_image_here' with the actual URL for the SHAP explanation image in your repository -->

### (ii) Counterfactual Explanations

Counterfactual explanations provide insight into the model's decisions by illustrating how altering certain inputs would change the outcome. For example, if the basket's position was different, how would the agent's action choice change?

![image](https://github.com/user-attachments/assets/84e5d016-a8d4-4e5d-8940-9c249bc704ad)  <!-- Replace 'link_to_counterfactual_image_here' with the actual URL for the Counterfactual explanation image in your repository -->
