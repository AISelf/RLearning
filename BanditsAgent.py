import numpy as np
from sklearn import logger
import random


class ActionValueAgent(object):
  def __init__(self, env, epsilon=0.1, num_actions=10):
    self.env = env
    self.epsilon = epsilon
    self.num_actions = num_actions
    self.rewards_sum = np.zeros(self.num_actions)
    self.select_num = np.zeros(self.num_actions)
    self.q_values = np.zeros(self.num_actions)

  def train(self, action, rewards):
    #logger.info("train action %d rewards %s", action, rewards)
    self.select_num[action] += 1
    self.rewards_sum[action] += rewards
    self.q_values[action] = self.rewards_sum[action] / self.select_num[action]
    #logger.info(self.q_values)

  def step(self):
    action = random.randint(0, self.num_actions - 1)
    if random.random() > self.epsilon:
      action = np.argmax(self.q_values)

    rewards = self.env.do_action(action = action)
    self.train(action, rewards)

  def done(self):
    select_num = np.sum(self.select_num)
    total_rewards = np.sum(self.rewards_sum)

    logger.info(self.q_values)
    logger.info("action num %d average rewards: %s optimal %d (%s) ", select_num, total_rewards / select_num,
                self.env.optimal_action(), self.select_num[self.env.optimal_action()] / select_num)
