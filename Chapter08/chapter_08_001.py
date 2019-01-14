import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')

# Build the network
input_size = env.observation_space.shape[0]

input_placeholder = tf.placeholder("float", [None, input_size])

# weights and bias of the hidden layer
weights_1 = tf.Variable(tf.truncated_normal([input_size, 20], stddev=0.01))
bias_1 = tf.Variable(tf.constant(0.0, shape=[20]))

# weights and bias of the output layer
weights_2 = tf.Variable(tf.truncated_normal([20, env.action_space.n], stddev=0.01))
bias_2 = tf.Variable(tf.constant(0.0, shape=[env.action_space.n]))

hidden_layer = tf.nn.tanh(tf.matmul(input_placeholder, weights_1) + bias_1)
output_layer = tf.matmul(hidden_layer, weights_2) + bias_2

action_placeholder = tf.placeholder("float", [None, 2])
target_placeholder = tf.placeholder("float", [None])

# network estimation
q_estimation = tf.reduce_sum(tf.multiply(output_layer, action_placeholder), reduction_indices=1)

# loss function
loss = tf.reduce_mean(tf.square(target_placeholder - q_estimation))

# Use Adam
train_operation = tf.train.AdamOptimizer().minimize(loss)

# initialize TF variables
session = tf.Session()
session.run(tf.global_variables_initializer())


def choose_next_action(state, rand_action_prob):
    """
    Simplified e-greedy policy
    :param state: current state
    :param rand_action_prob: probability to select random action
    """

    new_action = np.zeros([env.action_space.n])

    if random.random() <= rand_action_prob:
        # choose an action randomly
        action_index = random.randrange(env.action_space.n)
    else:
        # choose an action given our state
        action_values = session.run(output_layer, feed_dict={input_placeholder: [state]})[0]
        # we will take the highest value action
        action_index = np.argmax(action_values)

    new_action[action_index] = 1
    return new_action


def train(mini_batch):
    """
    Train the network on a single minibatch
    :param mini_batch: the mini-batch
    """

    last_state, last_action, reward, current_state, terminal = range(5)

    # get the batch variables
    previous_states = [d[last_state] for d in mini_batch]
    actions = [d[last_action] for d in mini_batch]
    rewards = [d[reward] for d in mini_batch]
    current_states = [d[current_state] for d in mini_batch]
    agents_expected_reward = []

    # this gives us the agents expected reward for each action we might take
    agents_reward_per_action = session.run(output_layer,
                                           feed_dict={input_placeholder: current_states})
    for i in range(len(mini_batch)):
        if mini_batch[i][terminal]:
            # this was a terminal frame so there is no future reward...
            agents_expected_reward.append(rewards[i])
        else:
            # otherwise compute expected reward
            discount_factor = 0.9
            agents_expected_reward.append(
                rewards[i] + discount_factor * np.max(agents_reward_per_action[i]))

    # learn that these actions in these states lead to this reward
    session.run(train_operation, feed_dict={
        input_placeholder: previous_states,
        action_placeholder: actions,
        target_placeholder: agents_expected_reward})


def q_learning():
    """The Q-learning method"""

    episode_lengths = list()

    # Experience replay buffer and definition
    observations = deque(maxlen=200000)

    # Set the first action to nothing
    last_action = np.zeros(env.action_space.n)
    last_action[1] = 1
    last_state = env.reset()

    total_reward = 0
    episode = 1

    time_step = 0

    # Initial chance to select random action
    rand_action_prob = 1.0

    while episode <= 400:
        # render the cart pole on the screen
        # comment this for faster execution
        # env.render()

        # select action following the policy
        last_action = choose_next_action(last_state, rand_action_prob)

        # take action and receive new state and reward
        current_state, reward, terminal, info = env.step(np.argmax(last_action))
        total_reward += reward

        if terminal:
            reward = -1.
            episode_lengths.append(time_step)

            print("Episode: %s; Steps before fail: %s; Epsilon: %.2f reward %s" %
                  (episode, time_step, rand_action_prob, total_reward))
            total_reward = 0

        # store the transition in previous_observations
        observations.append((last_state, last_action, reward, current_state, terminal))

        # only train if done observing
        min_experience_replay_size = 5000
        if len(observations) > min_experience_replay_size:
            # mini-batch of 128 from the experience replay observations
            mini_batch = random.sample(observations, 128)

            # train the network
            train(mini_batch)

            time_step += 1

        # reset the environment
        if terminal:
            last_state = env.reset()
            time_step = 0
            episode += 1
        else:
            last_state = current_state

        # gradually reduce the probability of a random action
        # starting from 1 and going to 0
        if rand_action_prob > 0 and len(observations) > min_experience_replay_size:
            rand_action_prob -= 1.0 / 15000

    # display episodes length
    plt.xlabel("Episode")
    plt.ylabel("Length (steps)")
    plt.plot(episode_lengths, label='Episode length')
    plt.show()


q_learning()
