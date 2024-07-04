import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            Flatten(),
        ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)


class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        if not shared:
            self.conv = DQNBase(num_channels)

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_channels, num_actions, shared, dueling_net)
        self.Q2 = QNetwork(num_channels, num_actions, shared, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False):
        super().__init__()
        if not shared:
            self.conv = DQNBase(num_channels)

        self.head = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions))

        self.shared = shared

    def act(self, states):
        if not self.shared:
            states = self.conv(states)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        if not self.shared:
            states = self.conv(states)

        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

    def soft_act(self, states, top_n=3):
        if not self.shared:
            states = self.conv(states)

        # 计算动作的概率分布
        action_probs = F.softmax(self.head(states), dim=1)

        # 找到前 top_n 个概率最大的动作
        top_probs, top_indices = torch.topk(action_probs, top_n, dim=1)

        # 重新归一化选出的 top_n 动作的概率分布
        top_probs_normalized = top_probs / top_probs.sum(dim=1, keepdim=True)

        # 创建一个新的类别分布，仅从 top_n 动作中采样
        action_dist = Categorical(top_probs_normalized)
        actions = top_indices.gather(1, action_dist.sample().view(-1, 1))

        # 计算 top_n 动作的对数概率
        z = (top_probs_normalized == 0.0).float() * 1e-8
        # log_action_probs = torch.log(top_probs_normalized + z)

        return actions
