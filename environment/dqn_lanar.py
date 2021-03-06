"""
-*- coding:utf-8 -*-
@Author  :   liaoyu
@Contact :   doitliao@126.com
@File    :   dqn_lanar.py
@Time    :   2020/1/14 14:29
@Desc    :
"""
import gym
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

if __name__ == '__main__':
    # Create environment
    env = gym.make('CartPole-v0')

    train = False
    if train:
        # Instantiate the agent
        model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
        # Train the agent
        model.learn(total_timesteps=int(2e5))
        # Save the agent
        model.save("dqn_lunar")
        del model  # delete trained model to demonstrate loading

    # Load the trained agent
    model = DQN.load("dqn_lunar")

    # Evaluate the agent
    mean_reward, n_steps = evaluate_policy(model, env, n_eval_episodes=10)

    # Enjoy trained agent
    obs = env.reset()
    for i in range(200):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            print(i)