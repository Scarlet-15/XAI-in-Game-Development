Fruit Catcher Game
1. About the Game
The Fruit Catcher game is an engaging and interactive reinforcement learning environment where a player controls a basket that moves horizontally at the bottom of the screen. The objective is to catch falling fruits to score points. Each successful catch increases the score, while missed fruits decrease the score or can potentially end the game. This setup provides a platform to train and evaluate reinforcement learning models, focusing on optimizing the agent's decision-making to achieve the highest score.

<-- [image](https://github.com/user-attachments/assets/e182e673-b06f-49fd-9bbb-1dc76cb6df8a)-->

2. Reinforcement Learning Model Used
Deep Q-Network (DQN)
For this project, we employed a Deep Q-Network (DQN), which is an algorithm that combines Q-Learning with deep neural networks. The DQN algorithm trains the agent to learn an optimal policy for maximizing rewards by approximating the Q-value function. The Q-value represents the expected cumulative reward of taking a specific action in a given state and following the optimal policy thereafter.

Training Details
The DQN model was trained for different numbers of episodes to analyze its learning process:

1 Episode: A preliminary training session with limited learning.
Model File - 1 Episode
10 Episodes: A short training period with some learning.
Model File - 10 Episodes
100 Episodes: A longer training period allowing the model to better understand the game environment.
Model File - 100 Episodes
Note: Replace link_to_1_episode_model_here, link_to_10_episode_model_here, and link_to_100_episode_model_here with the actual links to the model files in your GitHub repository.

3. Explanations Used
To understand why the trained RL agent makes certain decisions, we employed two eXplainable AI techniques: SHAP and Counterfactual explanations.

(i) SHAP (SHapley Additive exPlanations)
SHAP is a method for interpreting the output of machine learning models. It assigns importance scores to input features based on their contribution to the model's prediction. In the Fruit Catcher game, SHAP values help in identifying which aspects of the game state influence the agent's decisions.

<!-- Replace 'link_to_shap_image_here' with the actual URL for the SHAP explanation image in your repository -->

(ii) Counterfactual Explanations
Counterfactual explanations provide insight into the model's decisions by illustrating how altering certain inputs would change the outcome. For example, if the basket's position was different, how would the agent's action choice change?

<!-- Replace 'link_to_counterfactual_image_here' with the actual URL for the Counterfactual explanation image in your repository -->
