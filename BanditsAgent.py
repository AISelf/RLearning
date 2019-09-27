import numpy as np
from sklearn import logger
import random


class ActionValueAgent(object):
  def __init__(self, env, epsilon=0.1, num_actions=10):
    self.env = env
    self.epsilon = epsilon
    self.num_actions = num_actions
    self.select_num = np.zeros(self.num_actions)
    self.q_values = np.random.normal(0, 1, self.num_actions)
    #self.q_values = np.zeros(self.num_actions)
    self.total_rewards = 0

    logger.info(self.q_values)

  def train(self, action, rewards):
    self.select_num[action] += 1
    self.q_values[action] = self.q_values[action] + (rewards - self.q_values[action]) / self.select_num[action]
    self.total_rewards += rewards

  def step(self):
    action = random.randint(0, self.num_actions - 1)
    if random.random() > self.epsilon:
      action = np.argmax(self.q_values)

    rewards = self.env.do_action(action = action)
    self.train(action, rewards)

  def done(self):
    select_num = np.sum(self.select_num)

    logger.info(self.q_values)
    logger.info("action num %d average rewards: %s optimal %d (%s) ", select_num, self.total_rewards / select_num,
                self.env.optimal_action(), self.select_num[self.env.optimal_action()] / select_num)
