import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return pi, v


class Agent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions, fc1_dims, fc2_dims)

        self.log_prob = None

    def choose_action(self, observation):
        """Take raw obs from gym input and convert them to pytorch tensor
        Wrap in brackets because pytorch assumes batch data"""
        # dtype=float, pytorch deals with float better wrt long, int ...
        state = torch.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        # Categorical for discrete action space (for continuous use normal distr, providing mean and stdev too)
        action_probs = torch.distributions.Categorical(probabilities)
        # torch.distributions.categorical.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob
        return action.item()

    def learn(self, state, reward, next_state, done):
        self.actor_critic.optimizer.zero_grad()

        state = torch.tensor([state], dtype=torch.float).to(self.actor_critic.device)
        next_state = torch.tensor([next_state], dtype=torch.float).to(self.actor_critic.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, next_critic_value = self.actor_critic.forward(next_state)

        # 1 - int(done) because v(s_t_+1) is 0
        delta = reward + self.gamma*next_critic_value*(1-int(done)) - critic_value

        '''actor_loss: neg log_prob of the action the agent took, multiplied
        by delta, so as the agent learns it will shift the probs in the direction
        that maximizes delta, because we're doing gradient descent
        so we have neg sign, minimizing neg quantity'''
        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()



