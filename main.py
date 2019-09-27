import tensorflow as tf
import numpy as np
from testbed import Testbed

from sklearn import logger
from BanditsAgent import *

if __name__ == '__main__':
  print(tf.__version__)
  testbed = Testbed(
    action_values=[0.18561972, 0.05399372, -0.10540842, 0.5938879, 0.38869048, 0.54136417, 0.03121748, 0.38784468,
                   0.0560689, 1.89177961])
  reward = testbed.do_action(0)
  logger.info("reward %s", reward)

  agent = ActionValueAgent(env=testbed, epsilon=0.3)
  for i in range(0, 8000000):
    agent.step()

  agent.done()
