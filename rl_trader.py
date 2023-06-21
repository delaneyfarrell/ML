# using reinforcement learning to make a stock trading algorithm
# uses data from "aapl_msi_sbox.csv", which contains timeseries data of stock prices for 3 stocks:
# Apple, Motorola, and Starbucks

# imports
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import itertools
import argparse
import re
import os
import pickle

# import data - using Apple, Motorola, and Starbucks data (aapl_msi_sbux.csv)
def get_data():
    # returns a T x 3 list of stock prices, each row is different stock
    # 0 = AAPL, 1 = MSI, and 2 = SBUX
    df = pd.read_csv('aapl_msi_sbux.csv')
    return df.values


# replay buffer
class ReplayBuffer:
    """
    replay buffer contains 3 functions:
        - constructor, which initializes buffer arrays
        - store, which stores s, a, r, s', and done in buffers
        - sample_batch, which chooses indices from 0 to size of
          the buffer, and returns a dictionary
    """
    # constructor
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)  # states
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)  # next states
        self.acts_buf = np.zeros(size, dtype=np.uint8)               # actions
        self.rews_buf = np.zeros(size, dtype=np.float32)             # rewards
        self.done_buf = np.zeros(size, dtype=np.uint8)               # done (0 or 1)
        self.ptr, self.size, self.max_size = 0, 0, size              # pointer and size

    # store
    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs               # s
        self.obs2_buf[self.ptr] = next_obs          # s'
        self.acts_buf[self.ptr] = act               # a
        self.rews_buf[self.ptr] = rew               # r
        self.done_buf[self.ptr] = done              # done
        self.ptr = (self.ptr+1) % self.max_size     # increase pointer position (circular)
        self.size = min(self.size+1, self.max_size) # memory size increase with pointer

    # sample batch
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])


# get scaler, which returns a scaler object to scale states
# need some data to scale, so play randomly and store states we encounter
def get_scaler(env):
    # note: could also populate the replay buffer here
    states = []
    for _ in range(env.n_step):
        # randomly play, and store states
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break
    # now, scaler object (sklearn)
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


# utility function, makes directory if one doesn't yet exist
def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# MLP function: creates a NN model (multi-layer perceptron)
# this NN is trained and used by agent
def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
    # input
    i = Input(shape=(input_dim,))
    x = i
    # hidden layers
    for _ in range(n_hidden_layers):
        x = Dense(hidden_dim, activation='relu')(x)
    # final layer
    x = Dense(n_action)(x)
    model = Model(i,x)
    model.compile(loss='mse', optimizer='adam')
    print((model.summary()))
    return model


# environment class
class MultiStockEnv:
    """
    a 3 stock trading environment (n = 3)
    state: vector of size 7 (2 * n_stock + 1), comprised of
        - number of shares of stock 1 owned
        - number of shares of stock 2 owned
        - number of shares of stock 3 owned
        - price of stock 1 (using daily close price)
        - price of stock 2
        - price of stock 3
        - cash owned
    action: categorical value with 3^3 (= 27) possibilities, as for
    each stock, you can
        - sell = 0
        - hold = 1
        - buy = 2
    """
    def __init__(self, data, initial_investment=20000):
        # data
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape
        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.action_space = np.arange(3**self.n_stock)
        # action permutations returns a nested list, with elements like [0,1,0]
        self.action_list = list(map(list, itertools.product([0,1,2], repeat=self.n_stock)))
        # calculate size of state
        self.state_dim = self.n_stock*2 + 1
        self.reset()  # returns initial state

    # reset function
    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()  # state vector

    # step function, preforms action in env and returns next state & reward
    def step(self, action):
        assert action in self.action_space
        prev_val = self._get_val()              # get current value before the action
        self.cur_step += 1                      # update price (step to next day)
        self.stock_price = self.stock_price_history[self.cur_step]
        self._trade(action)                     # perform the trade
        cur_val = self._get_val()               # new value after the action
        reward = cur_val-prev_val               # r = increase in portfolio value
        done = self.cur_step == self.n_step-1   # done = if we run out of data
        info = {'cur_val': cur_val}             # store current value of portfolio
        # confirm to the Gym API
        return self._get_obs(), reward, done, info

    # function to return the state (= observation for this script)
    def _get_obs(self):
        obs = np.empty(self.state_dim)
        # state vector: stocks owned, stock prices, and cash
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    # value of portfolio = stock value + cash
    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    # trade function
    # note: to simplify the problem, sell = sell ALL shares
    # buying occurs in a round-robin style until we run out of cash (see notes)
    def _trade(self, action):
        # first, get action vector (0 = sell, 1 = hold, 2 = buy)
        action_vec = self.action_list[action]
        # then determine what stocks are being bought or sold
        sell_idx = []
        buy_idx = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_idx.append(i)
            elif a == 2:
                buy_idx.append(i)
        # sell and buy
        if sell_idx:
            for i in sell_idx:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0   # sell all shares
        if buy_idx:
            can_buy = True
            while can_buy:
                for i in buy_idx:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1   # buy one share if we have cash
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False


# agent class
class DQNAgent(object):
    """
    the agent in this simulation - who takes information from past experiences,
    learning from them and maximizes future rewards (our AI!)
    """
    # constructor
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95       # discount rate (hyperparam)
        self.epsilon = 1.0      # exploration rate (hyperparam)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size)

    # update replay buffer
    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    # function to return action given a state using epsilon greedy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        # if not random, preform a "greedy" action by getting all Q values
        # and taking max over a
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    # replay function - does the learning!
    # batch size = how many samples to grab from the replay memory
    def replay(self, batch_size=32):
        # check is replay buffer contains enough data
        if self.memory.size < batch_size:
            return
        # sample a batch of data from replay memory
        minibatch = self.memory.sample_batch(batch_size)  # returns dictionary
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']
        # calculate the tentative target, Q(s',a)
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
        # value of terminal state (where data ends) is 0, so set target to = reward only
        target[done] = rewards[done]

        # note: with the Keras API, target usually must have the same shape
        # as the prediction, but we need to update the network for the actions
        # which were actually taken. we can accomplish this by setting the target
        # equal to the prediction for all values, then only change the targets
        # for the actions taken.

        # target_full: model predictions for each state & action
        target_full = self.model.predict(states)
        # now change only actions taken with targets previously calculated!
        target_full[np.arange(batch_size), actions] = target
        # run one training step (GD)
        self.model.train_on_batch(states, target_full)
        # update epsilon to reduce amount of exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # load model weights
    def load(self, name):
        self.model.load_weights(name)

    # save model weights
    def save(self, name):
        self.model.save_weights(name)


# play one episode function
def play_one_episode(agent, env, is_train):
    state = env.reset()
    state = scaler.transform([state])   # transforming puts states as 1xD
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.update_replay_memory(state, action, reward, next_state, done)
            agent.replay(batch_size)    # run one step of GD
        state = next_state
    return info['cur_val']  # return current value of portfolio


if __name__ == '__main__':
    # make directories to store models & rewards (both train and test)
    models_folder = 'rl_trader_models'
    rewards_folder = 'rl_trader_rewards'
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    # configuration variables
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000

    # create argument parser object so we can run script with command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    args = parser.parse_args()

    # get timeseries data
    data = get_data()
    n_timesteps, n_stocks = data.shape

    # split data into train and test (1/2 for each)
    n_train = n_timesteps // 2
    train_data = data[:n_train]
    test_data = data[n_train:]

    # set up environment with training data
    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # list to store final value in portfolio at end of episode
    portfolio_value = []

    # for test mode, need scaler we had in training
    if args.mode == 'test':
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # recreate the environment with test data
        env = MultiStockEnv(test_data, initial_investment)

        # ensure epsilon /= 1 (1 is pure exploration)
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f'{models_folder}/dqn.h5')

    # now, play the game num_episodes times
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, args.mode)
        dt = datetime.now() - t0
        print(f"episode: {e+1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
        portfolio_value.append(val)     # append end portfolio value

    # save weights when we are done
    if args.mode == 'train':
        agent.save(f'{models_folder}/dqn.h5')   # save the DQN
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)              # save the scaler

    # save portfolio value for each episode
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

# to run in train mode and plot train results:
# python3 rl_trader_mine.py -m train && python3 plot_rl_rewards.py -m train
# for test:
# python3 rl_trader_mine.py -m test && python3 plot_rl_rewards.py -m test







