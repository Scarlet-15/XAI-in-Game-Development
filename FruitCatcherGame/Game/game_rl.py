#Playing the game with the RL model we trained to see it's performance
import numpy as np

import game
import dqn

model_path = r"models\-1.h5"

env = game.Environment()
agent = dqn.DQN(
    state_shape=env.ENVIRONMENT_SHAPE,
    action_size=env.ACTION_SPACE_SIZE
)
agent.load(model_path)

state = env.reset()
state = np.expand_dims(state, axis=0)

import pygame
pygame.init()
screen = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
clock = pygame.time.Clock()
running = True
score = 0

while running:
    pygame.display.set_caption(f"Score: {score}")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = agent.act(state, 0)
    state, reward, done, score = env.step(action)
    print(state,reward,action)
    state = np.expand_dims(state, axis=0)

    env.render(screen)
    pygame.display.flip()
    clock.tick(30)
    running=False

pygame.quit()