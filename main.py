import tensorflow as tf
import numpy as np
from testbed import *

from sklearn import logger
from BanditsAgent import *

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
  print(tf.__version__)

  testbed = Testbed(
    action_values=[0.18561972, 0.05399372, -0.10540842, 0.5938879, 0.38869048, 0.54136417, 0.03121748, 0.38784468,
                   0.0560689, 1.89177961])

  testbed = NonStationaryTestbed()
  reward = testbed.do_action(0)
  logger.info("reward %s", reward)

  steps_num = 10000
  agents = [ActionValueAgent(env=testbed, epsilon=0.1), ActionValueAgent(env=testbed, epsilon=0.01), ActionValueAgent(env=testbed, epsilon=0), NonStationaryActionValueAgent(env = testbed, epsilon=0.1)]

  for i in range(0, steps_num):
    for a in agents:
      a.step()
    testbed.upgrade()

  for a in agents:
    a.done()


  idx = 1

  plt.figure(1)
  for a in agents:
    plt.subplot("22" + str(idx))
    idx+=1
    plt.plot(a.average_rewards)

  plt.savefig("average_reward.png")

  plt.close()

  plt.figure(1)
  idx = 1
  for a in agents:
    plt.subplot("22" + str(idx))
    plt.plot(a.optimal_action)
    idx+=1

  plt.savefig("optimal_action.png")