"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable
#import time


def update():
    for episode in range(100):
        # initial observation
#        time.sleep(1000)
        observation = env.reset()
        step_counter = 0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_
            
            step_counter += 1

            # break while loop when end of this episode
            if done:
                interaction = 'Episode %s: total_steps = %s: reward = %s' % (episode+1, step_counter,reward)
                print('\n{}'.format(interaction), end='')
                break

    # end of game
    print('\ngame over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
    print('\r\nQ-table:\n')
    print(RL.q_table)