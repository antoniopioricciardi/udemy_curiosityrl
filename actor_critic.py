import torch
import torch.distributions
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, tau=1.0):
        super(ActorCritic, self).__init__()
        self.gamma = gamma  # needed in learning function
        self.tau = tau  # can test from 0.95 to 1
        # (n_channels, out_filters, kernel_size, stride, padding)
        self.conv_1 = nn.Conv2d(input_dims[0], 32, 3, stride=2, padding=1)
        # (in_filters, out_filters ...)
        self.conv_2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.conv_shape = self.calc_conv_shape(input_dims)
        self.gru = nn.GRU(input_size=self.conv_shape, hidden_size=256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def calc_conv_shape(self, input_dims):
        # 1 dimension because we need to use batches!
        dims = torch.zeros(1, *input_dims)
        # print(dims.shape)
        dims = self.conv_1(dims)
        # print(dims.shape)
        dims = self.conv_2(dims)
        # print(dims.shape)
        dims = self.conv_3(dims)
        # print(dims.shape)
        dims = self.conv_4(dims)
        # print(dims.shape)
        return int(np.prod(dims.size()))

    def forward(self, state, hx):
        # use elu, as in the paper
        x = F.elu(self.conv_1(state))
        x = F.elu(self.conv_2(x))
        x = F.elu(self.conv_3(x))
        x = F.elu(self.conv_4(x))
        # print(x.size())
        # x.size()[0] is needed to batch data, flatten the rest
        x = x.view((x.size()[0], -1)).unsqueeze(0)
        # print(x.size())

        # hx = hx[None, :]
        # hx = hx.unsqueeze(0)
        _, hx = self.gru(x, (hx))
        pi = self.pi(hx)
        value = self.v(hx)

        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.numpy()[0], value, log_prob, hx

    def calc_R(self, done, rewards, values):
        # values is a list of tensors, need to convert to tensor
        values = torch.cat(values).squeeze()
        # handle sizes for terminal states, might have different sizes
        # 1 - int(done) is for taking care of the fact that if the state is terminal, then the value for that state
        # should be 0
        if len(values.size()) == 1:  # batch of states
            # from the appendix, Algo S3 for A3C
            R = values[-1]*(1 - int(done))  # if we are done, take terminal value for state and set it to 0
        elif len(values.size()) == 0:  # single state
            R = values*(1 - int(done))  # if we have a single state, if we are done then set it to 0 (no need to take last state, we have only 1)

        # deal with calculation of batch returns, iterate backward over rewards received
        # add reward + gamma*(R-1), then reverse and convert to tensor
        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float).reshape(values.size())
        return batch_return

    def calc_cost(self, new_state, hx, done, rewards, values, log_probs):
        returns = self.calc_R(done, rewards, values)
        # values needed to compute delta
        next_v = torch.zeros(1, 1, 1) if done else self.forward(torch.tensor([new_state], dtype=torch.float), hx)[1]
        values.append(next_v.detach())
        values = torch.cat(values).squeeze()
        log_probs = torch.cat(log_probs)
        rewards = torch.tensor(rewards)

        # next_v[1:] because it is the t+1 timestep
        # values[:-1] value function for the state at time T
        delta_t = rewards + self.gamma * next_v[1:] - values[:-1]
        n_steps = len(delta_t)

        gae = np.zeros(n_steps)

        """ Compute Generalized Advantage Estimate (GAE) """
        for t in range(n_steps):
            for k in range(0, n_steps-t):  # k is the l from the paper
                temp = (self.gamma * self.tau)**k * delta_t[t+k]
                gae[t] += temp
        gae = torch.tensor(gae, dtype=torch.float)

        actor_loss = -(log_probs * gae).sum()

        # critic loss computed for the t-1,...t_start timesteps (this is why we take values[:-1])
        # we use squeeze for critic loss, because if we have a single state then values has rank=1, returns rank=0
        # so we want ot have same shape
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)

        entropy_loss = (-log_probs * torch.exp(log_probs)).sum()

        total_loss = actor_loss + critic_loss + 0.01 * entropy_loss  # 0.01 is Beta from the paper
        return total_loss


# input_dims = np.array([4,42,42])
# input_dims = torch.tensor(input_dims)
# conv_dim = torch.tensor([128])
# net = ActorCritic(input_dims, 6)
# x = np.random.random([1, 4,42,42])
# x = torch.tensor(x, dtype=torch.float)
# print(x.shape)
# net.forward(x, torch.tensor([[2]]))


