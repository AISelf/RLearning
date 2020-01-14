"""
-*- coding:utf-8 -*-
@Author  :   liaoyu
@Contact :   doitliao@126.com
@File    :   dqn_stock.py
@Time    :   2020/1/14 16:58
@Desc    :
"""
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN

from env.StockTradingEnv import StockTradingEnv

if __name__ == '__main__':


    import pandas as pd

    df = pd.read_csv('data/AAPL.csv')
    df = df.sort_values('Date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    train = True
    if train:
        # Instantiate the agent
        model = PPO2('MlpPolicy', env, verbose=1)
        # Train the agent
        model.learn(total_timesteps=int(2e5))
        # Save the agent
        model.save("dqn_stock")
        del model  # delete trained model to demonstrate loading

    # Load the trained agent
    model = PPO2.load("dqn_stock")

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