import os
import argparse
import collections

import gym
import ptan
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from rsim.rsim import RSim

import rc_gym

from client import FiraClient
from Entities import *

ENV = 'VSS3v3-v0'
DEADZONE = 0.05
SPEED_RANGE = 1.15
WHEEL_RADIUS = 0.026
# Q_SIZE = 1


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


def _actions_to_v_wheels(actions):
        left_wheel_speed = actions[0] * SPEED_RANGE
        right_wheel_speed = actions[1] * SPEED_RANGE

        # Deadzone
        if -DEADZONE < left_wheel_speed < DEADZONE:
            left_wheel_speed = 0

        if -DEADZONE < right_wheel_speed < DEADZONE:
            right_wheel_speed = 0

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -2.6, 2.6)

        return left_wheel_speed/ WHEEL_RADIUS, right_wheel_speed/ WHEEL_RADIUS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    args = parser.parse_args()

    net = DDPGActor(40, 2)
    net.load_state_dict(torch.load(args.model))

    client = FiraClient()
    # client = RSim()

    while True:
        obs_v = torch.FloatTensor([client.receive()])
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        lw, rw = _actions_to_v_wheels(action)
        command = [Robot(yellow=False, id=0, v_wheel1=lw, v_wheel2=rw)]
        client.send(command)
