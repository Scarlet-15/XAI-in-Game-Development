# Integrating Deep Learning and Explainability in Game Development

This repository contains the code and resources for our research paper on bringing explainability to actions performed by reinforcement learning (RL) models and Deep Learning (DL) models in game environments. The research focuses on two games: Fruit Catcher and Shooter, using various eXplainable Artificial Intelligence (XAI) techniques such as SHAP, LIME, Counterfactuals, and GRAD-CAM to interpret the models' behavior.

## Table of Contents
1. [Introduction](#introduction)
2. [Folder Descriptions](#folder-descriptions)
3. [Explainability Techniques](#explainability-techniques)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Results](#results)

## Introduction

The goal of this project is to introduce explainability to reinforcement learning agents playing games and Deep Learning Games allowing for a better understanding of their decisions. Two games are studied:
- **Fruit Catcher Game**: A simple game where the user catches fruits in basket implemented using Reinforcement Learning(DQN models). Uses SHAP and Counterfactuals for generating explanations.
- **Shooter Game**: A simple shooter game that helps aim a ball implemented using a YOLOv8 model Uses LIME and GRAD-CAM for generating explanations.

We evaluate the explanations based on several metrics, including fidelity, unambiguity, and inference time.

## Folder Descriptions
- **FruitCatcherGame**: Contains code for the Fruit Catcher game, including training scripts, explanation visualizations, and models.
- **ShooterGame**: Contains code and datasets for the Shooter game, along with explanation methods and game logic.
- **Explanations**: Implementations of various XAI techniques.
- **models**: Pre-trained models used in the experiments.
- **Dataset**: Contains image datasets and annotations used in the Shooter game.

## Explainability Techniques

- **SHAP**: Provides feature importance by measuring the impact of each input feature on the model's predictions.
- **Counterfactuals**: Generates hypothetical scenarios to show how changes in the input would affect the output.
- **LIME**: Locally approximates the model's behavior to provide interpretable explanations.
- **GRAD-CAM**: Uses gradients to highlight the regions of input data influencing the model's decision.

## Evaluation Metrics

The following metrics are used to evaluate the quality of the explanations:
- **Fidelity**: Measures how well the explanation aligns with the model’s true predictions.
- **Unambiguity**: Assesses how clear and interpretable the explanation is.
- **Inference Time**: Measures the time taken to generate an explanation.

## Results

The results of our experiments show how different explanation techniques perform across the chosen metrics. Detailed results can be found in the paper.
