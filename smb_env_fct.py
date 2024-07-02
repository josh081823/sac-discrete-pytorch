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

