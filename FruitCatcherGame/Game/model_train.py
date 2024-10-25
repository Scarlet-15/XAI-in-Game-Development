#This code is essentially used to train the model by playing the game for many instanses and learning how to play it
#We also save the model trained for future references
import numpy as np
from collections import deque
import game
import dqn

LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = 0.99
EXPLORATION_DECAY = 0.95
GAMMA = 0.975
UPDATE_TARGET_EVERY = 10

BATCH_SIZE = 128
EPISODES = 100

env = game.Environment()
agent = dqn.DQN(
    state_shape=env.ENVIRONMENT_SHAPE,
    action_size=env.ACTION_SPACE_SIZE,
    batch_size=BATCH_SIZE,
    learning_rate_max=LEARNING_RATE,
    learning_rate_decay=LEARNING_RATE_DECAY,
    exploration_decay=EXPLORATION_DECAY,
    gamma=GAMMA
)
agent.save(f'models/FruitCatcher100.h5')

state = env.reset()
state = np.expand_dims(state, axis=0)

most_recent_losses = deque(maxlen=BATCH_SIZE)

log = []

# fill up memory before training starts
while agent.memory.length() < BATCH_SIZE:
    action = agent.act(state)
    next_state, reward, done, score = env.step(action)
    next_state = np.expand_dims(next_state, axis=0)
    agent.remember(state, action, reward, next_state, done)
    state = next_state

for e in range(0, EPISODES):
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    done = False
    step = 0
    ma_loss = None

    while not done:
        action = agent.act(state)
        print(action)
        next_state, reward, done, score = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        step += 1

        loss = agent.replay(episode=e)
        most_recent_losses.append(loss)
        ma_loss = np.array(most_recent_losses).mean()

        if loss != None:
            print(f"Step: {step}. Score: {score}. -- Loss: {loss}", end="          \r")

        if done:
            print(f"Episode {e}/{EPISODES-1} completed with {step} steps. Score: {score:.0f}. LR: {agent.learning_rate:.6f}. EP: {agent.exploration_rate:.2f}. MA loss: {ma_loss:.6f}")
            break

    log.append([e, step, score, agent.learning_rate, agent.exploration_rate, ma_loss])
