from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random

import numpy as np

from sklearn import logger


class Testbed(object):
  def __init__(self, action_num=10, action_values=[]):
    if len(action_values) == 0:
      self.action_num = action_num
      self.action_values = np.random.normal(0, 1, self.action_num);
    else:
      self.action_num = len(action_values)
      self.action_values = action_values
    logger.info("action_values %s", self.action_values)

  def do_action(self, action):
    assert 0 <= action < self.action_num

    return np.random.normal(self.action_values[action], 1, 1)[0];

  def optimal_action(self):
    return np.argmax(self.action_values)

class NonStationaryTestbed(object):
  def __init__(self, action_num=10, action_values=[]):
    if len(action_values) == 0:
      self.action_num = action_num
      self.action_values = np.zeros(self.action_num) #np.random.normal(0, 1, self.action_num);
    else:
      self.action_num = len(action_values)
      self.action_values = action_values
    logger.info("action_values %s", self.action_values)

  def do_action(self, action):
    assert 0 <= action < self.action_num
    #adding a normally distributed incrementally with mean zero and standard deviation 0.01
    #self.action_values += np.random.normal(0, 0.01, 10)
    return np.random.normal(self.action_values[action], 1, 1)[0]

  def upgrade(self):
    self.action_values += np.random.normal(0, 0.01, 10)

  def optimal_action(self):
    return np.argmax(self.action_values)
