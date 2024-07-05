import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT
from datetime import datetime
import cv2
from tqdm import tqdm
import numpy as np
from gym.wrappers import GrayScaleObservation
import gym
import logging
import random


#renders given frames with mediapy and shows a video
def renderEnv(frames):
  import mediapy as media
  media.show_video(frames,fps=60//6)

#plot for visualizing results
def plotRewardandTime(avg_norm_reward,avg_length):
  import matplotlib.pyplot as plt
  import scipy
  from scipy.signal import savgol_filter
  x = np.linspace(0,len(avg_norm_reward),len(avg_norm_reward))

  fig, axs = plt.subplots(1, 2,figsize=(9,3))

  axs[0].plot(x, avg_norm_reward)
  axs[0].set_title("avg_norm_reward")

  axs[1].plot(x, avg_length)
  axs[1].set_title("avg_length")
  plt.show()

  

#This environment wrapper is used to stop a run if mario is stuck on a pipe
class DeadlockEnv(gym.Wrapper):
    def __init__(self, env, threshold=10):
        super().__init__(env)
        self.last_x_pos = 0
        self.count = 0
        self.threshold = threshold
        self.lifes = 3
        self.stage = 1
        self.world = 1

    def reset(self, **kwargs):
        self.last_x_pos = 0
        self.count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        #state, reward, done, info, terminated = self.env.step(action)
        #observation, reward, terminated, truncated, info =  self.env.step(action)
        observation, reward, terminated, info, truncated =  self.env.step(action)
        x_pos = info['x_pos']

        if x_pos <= self.last_x_pos:
            self.count += 1
        else:
            self.count = 0
            self.last_x_pos = x_pos

        if info['life'] != self.lifes or info["stage"] != self.stage or info["world"] != self.world:
            self.last_x_pos = x_pos
            self.count = 0
            self.lifes = info['life']
            self.stage = info["stage"]
            self.world = info["world"]

        if self.count >= self.threshold:
            reward = -15
            #done = True
            #truncated = True
            terminated = True

        #return state, reward, done, info
        #return observation, reward, truncated, info
        return observation, reward, terminated or truncated, info

#This environment wrapper is used to stop a run if mario is stuck on a pipe
class DeadlockEnv_2(gym.Wrapper):
    def __init__(self, env, threshold=10, force_progress=False):
        super().__init__(env)
        self.last_x_pos = 0
        self.count = 0
        self.threshold = threshold
        self.lifes = 3
        self.stage = 1
        self.world = 1
        self.force_progress = force_progress
        print("DeadlockEnv using force_process: ", self.force_progress)

    def reset(self, **kwargs):
        self.last_x_pos = 0
        self.count = 0
        self._done = False
        return self.env.reset(**kwargs)

    def step(self, action):
        # state, reward, done, info = self.env.step(action)
        state, reward, done, truncated, info = self.env.step(action)

        x_pos = info['x_pos']

        if self.force_progress:
            stuck_cond = x_pos <= self.last_x_pos
        else:
            stuck_cond = x_pos == self.last_x_pos

        #if x_pos <= self.last_x_pos:
        if stuck_cond:
            self.count += 1
        else:
            self.count = 0
            self.last_x_pos = x_pos

        if info['life'] != self.lifes or info["stage"] != self.stage or info["world"] != self.world:
            self.last_x_pos = x_pos
            self.count = 0
            self.lifes = info['life']
            self.stage = info["stage"]
            self.world = info["world"]

        if self.count >= self.threshold:
            reward = -10
            done = True

        return state, reward, done or truncated, info

#skipframe wrapper
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        reward_out = 0
        for i in range(self._skip):
            #obs, reward, done, info, terminated = self.env.step(action)
            observation, reward, terminated, truncated, info = self.env.step(action)
            reward_out += reward
            #if terminated:
            #print("terminated:", terminated, "truncated:", truncated)
            if terminated or truncated:
                # print("terminated:", terminated, "truncated:", truncated)
                break
        reward_out /= max(1,i+1)

        #return obs, reward_out, done, info, terminated
        return observation, reward_out, terminated, info, truncated

# 随机环境
class RandomEnvWrapper(gym.Wrapper):
    def __init__(self, env_id, stages):
        # 初始化包装器并创建所有环境
        prefix, version = env_id.split('-v')
        version = 'v' + version
        generate_env_list = lambda base, stages: [f"{base.split('-v')[0]}-{stage}-v{base.split('-v')[1]}" for stage in stages]
        env_id_list = generate_env_list(env_id, stages)

        self.env_list = [gym_super_mario_bros.make(env_id, apply_api_compatibility=True) for env_id in env_id_list]
        self.current_env = random.choice(self.env_list)
        super(RandomEnvWrapper, self).__init__(self.current_env)

    def reset(self):
        # 随机选择一个新的环境并重置
        self.current_env = random.choice(self.env_list)
        self.env = self.current_env  # 更新包装器中的当前环境
        return self.env.reset()

    def close(self):
        # 关闭所有环境
        for env in self.env_list:
            env.close()

class MarioDeathLoggerWrapper(gym.Wrapper):
    def __init__(self, env, logfile="mario_deaths.log", env_id=None, select_random_stage=None):
        super().__init__(env)
        self.env = env
        self.episode_counter = 0
        self.log_file = logfile
        self.env_id = env_id
        self.select_random_stage = select_random_stage

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Check if Mario died
        if done and 'x_pos' in info and 'y_pos' in info:
            # Log the death position and episode number
            self.log_death(info['x_pos'], info['y_pos'], info['stage'], info['world'])
        
        return obs, reward, done, info

    def reset(self):
        # Increment episode counter on reset
        self.episode_counter += 1
        return self.env.reset()

    def log_death(self, x_pos, y_pos, stage, world):
        # Log the death position and episode number to a file
        with open(self.log_file, 'a') as file:
            if self.select_random_stage is not None and self.env_id is not None:
                file.write(f"Episode: {self.episode_counter}, \tEnv-Id: {self.env_id}, \tRandom-Stages: {self.select_random_stage}, \tWorld: {world}, \tStage: {stage} \tDeath Position (x, y): ({x_pos}, {y_pos}) \t⸺ {self.env_id}, {self.episode_counter}, \t⸺ {world}, {stage}, {x_pos}\n")
            else:
                file.write(f"Episode: {self.episode_counter}, \tWorld: {world}, \tStage: {stage}, \tDeath Position (x, y): ({x_pos}, {y_pos}) \t⸺ {self.episode_counter}, \t⸺ {world}, {stage}, {x_pos}\n")


#downsample wrapper to reduce dimensionality
def Downsample(ratio,state):
  (oldh, oldw, oldc) = state.shape
  newshape = (oldh//ratio, oldw//ratio, oldc)
  frame = cv2.resize(state, (newshape[0], newshape[1]), interpolation=cv2.INTER_AREA)
  return frame

#small function to change rgb images to grayscale
def GrayScale(state):
  return cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

def move_state_channels(state):
    """
    将形状为 (height, width, channels) 的 NumPy 数组的通道数移动到第一维。

    Args:
        state (np.ndarray): 形状为 (240, 256, 3) 的 NumPy 数组。

    Returns:
        np.ndarray: 形状为 (3, 240, 256) 的 NumPy 数组。
    """
    return np.transpose(state, (2, 0, 1))


