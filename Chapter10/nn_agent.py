#!/usr/bin/env python
from __future__ import print_function

import os

import gym
import numpy as np
import torch

from train \
    import \
    data_transform, \
    available_actions, \
    build_network, \
    DATA_DIR, MODEL_FILE


def nn_agent_play(model, device):
    """
    Let the agent play
    :param model: the network
    :param device: the cuda device
    """

    env = gym.make('CarRacing-v0')

    # use ESC to exit
    global human_wants_exit
    human_wants_exit = False

    def key_press(key, mod):
        """Capture ESC key"""
        global human_wants_exit
        if key == 0xff1b:  # escape
            human_wants_exit = True

    # initialize environment
    state = env.reset()
    env.unwrapped.viewer.window.on_key_press = key_press

    while 1:
        env.render()

        state = np.moveaxis(state, 2, 0)  # channel first image

        # numpy to tensor
        state = torch.from_numpy(np.flip(state, axis=0).copy())
        state = data_transform(state)  # apply transformations
        state = state.unsqueeze(0)  # add additional dimension
        state = state.to(device)  # transfer to GPU

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(state)

        normalized = torch.nn.functional.softmax(outputs, dim=1)

        # translate from net output to env action
        max_action = np.argmax(normalized.cpu().numpy()[0])
        action = available_actions[max_action]

        # adjust brake power
        if action[2] != 0:
            action[2] = 0.3

        state, _, terminal, _ = env.step(action)  # one step

        if terminal:
            state = env.reset()

        if human_wants_exit:
            env.close()
            return


if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = build_network()
    m.load_state_dict(torch.load(os.path.join(DATA_DIR, MODEL_FILE)))
    m.eval()
    m = m.to(dev)
    nn_agent_play(m, dev)
