#!/usr/bin/env python
from __future__ import print_function

import gzip
import os
import pickle
import time

import gym
import numpy as np

from util import DATA_DIR, DATA_FILE


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_wants_exit, human_sets_pause, acceleration
    if key == 0xff0d:  # enter
        human_wants_restart = True

    if key == 0xff1b:  # escape
        human_wants_exit = True

    if key == 0x020:  # space
        human_sets_pause = not human_sets_pause

    if key == 0xff52:  # up
        acceleration = True
        human_agent_action[1] = 1.0
        human_agent_action[2] = 0
    if key == 0xff54:  # down
        human_agent_action[2] = 1  # stronger brakes

    if key == 0xff51:  # left
        human_agent_action[0] = -1.0

        # no acceleration while turning
        human_agent_action[1] = 0.0

    if key == 0xff53:  # right
        human_agent_action[0] = +1.0

        # no acceleration when turning
        human_agent_action[1] = 0.0


def key_release(key, mod):
    global human_agent_action, acceleration
    if key == 0xff52:  # up
        acceleration = False
        human_agent_action[1] = 0.0

    if key == 0xff54:  # down
        human_agent_action[2] = 0.0

    if key == 0xff51:  # left
        human_agent_action[0] = 0

        # restore acceleration
        human_agent_action[1] = acceleration

    if key == 0xff53:  # right
        human_agent_action[0] = 0

        # restore acceleration
        human_agent_action[1] = acceleration


def rollout(env):
    global human_wants_restart, human_agent_action, human_wants_exit, human_sets_pause

    ACTIONS = env.action_space.shape[0]
    human_agent_action = np.zeros(ACTIONS, dtype=np.float32)
    human_wants_exit = False
    human_sets_pause = False

    human_wants_restart = False

    # if the file exists, append
    if os.path.exists(os.path.join(DATA_DIR, DATA_FILE)):
        with gzip.open(os.path.join(DATA_DIR, DATA_FILE), 'rb') as f:
            observations = pickle.load(f)
    else:
        observations = list()

    state = env.reset()
    total_reward = 0
    total_timesteps = 0
    episode = 1
    while 1:
        env.render()

        a = np.copy(human_agent_action)

        old_state = state

        if human_agent_action[2] != 0:
            human_agent_action[2] = 0.1

        state, r, terminal, info = env.step(human_agent_action)

        observations.append((old_state, a, state, r, terminal))

        total_reward += r

        if human_wants_exit:
            env.close()
            return

        if human_wants_restart:
            human_wants_restart = False
            state = env.reset()
            continue

        if terminal:
            if episode % 5 == 0:
                # store generated data
                data_file_path = os.path.join(DATA_DIR, DATA_FILE)
                print("Saving observations to " + data_file_path)

                if not os.path.exists(DATA_DIR):
                    os.mkdir(DATA_DIR)

                with gzip.open(data_file_path, 'wb') as f:
                    pickle.dump(observations, f)

            print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

            episode += 1

            state = env.reset()

        while human_sets_pause:
            env.render()
            time.sleep(0.1)


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    print("ACTIONS={}".format(env.action_space.shape[0]))
    print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
    print("No keys pressed is taking action 0")

    rollout(env)
