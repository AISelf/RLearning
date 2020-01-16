"""
-*- coding:utf-8 -*-
@Author  :   liaoyu
@Contact :   doitliao@126.com
@File    :   training_saving_loading.py
@Time    :   2020/1/16 00:14
@Desc    :
"""
import gym

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy


# Create environment
env = gym.make('CartPole-v1')

train = False
if train:
    # Instantiate the agent
    model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(2e4))
    # Save the agent
    model.save("dqn_cart_pole")
    del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("dqn_cart_pole")

# Evaluate the agent
mean_reward, n_steps = evaluate_policy(model, env, n_eval_episodes=10)

print(f'mean_reward {mean_reward} n_steps {n_steps}')
# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()