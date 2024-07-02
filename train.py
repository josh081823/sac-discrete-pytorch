import imp
import os
import yaml
import argparse
from datetime import datetime

from sacd.env import make_pytorch_env
from sacd.agent import SacdAgent, SharedSacdAgent


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
from smb_env_fct import renderEnv, plotRewardandTime, DeadlockEnv, SkipFrame, Downsample, GrayScale, move_state_channels


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    frameskip = 8                         # the frameskip value of the environment

    # Create environments.
    # env = make_pytorch_env(args.env_id, clip_rewards=False)
    # test_env = make_pytorch_env(
    #     args.env_id, episode_life=False, clip_rewards=False)

    env = gym_super_mario_bros.make('SuperMarioBros-v1', apply_api_compatibility=True)  #the environment. v0 is with original background, v1 has the background removed
    env = JoypadSpace(env, SIMPLE_MOVEMENT)               #The Joypadspace sets the available actions. We use SIMPLE_MOVEMENT.
    env = SkipFrame(env, skip=frameskip)                  #Skipframewrapper to skip some frames
    env = DeadlockEnv(env,threshold=10)                   #Deadlock environment wrapper to stop the game if mario is stuck at a pipe

    test_env = gym_super_mario_bros.make('SuperMarioBros-v1', apply_api_compatibility=True)  
    test_env = JoypadSpace(test_env, SIMPLE_MOVEMENT)
    test_env = SkipFrame(test_env, skip=frameskip)
    test_env = DeadlockEnv(test_env,threshold=10)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    Agent = SacdAgent if not args.shared else SharedSacdAgent
    agent = Agent(
        env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
        seed=args.seed, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
