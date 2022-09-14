import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    def __init__(self, input_dims, n_actions=3, alpha=0.1, beta=0.2):
        super(ICM, self).__init__()
        self.alpha = alpha
        self.beta = beta

        # (n_channels, out_filters, kernel_size, stride, padding)
        self.conv_1 = nn.Conv2d(input_dims[0], 32, 3, stride=2, padding=1)
        # (in_filters, out_filters ...)
        self.conv_2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # phi is the feature representation of the state (when flattened)
        self.phi = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # inverse model
        self.inverse = nn.Linear(288*2, 256) # 288 is the output of phi, 2 because we concatenate phi(s) and phi(s')
        self.pi_logits = nn.Linear(256, n_actions) # logits for the policy. We pass this to the cross_entropy loss function, which automatically applies softmax
        
        # forward model
        self.dense_1 = nn.Linear(288+1, 256) # 288 is the output of phi, 1 is the action
        self.phi_hat_next = nn.Linear(256, 288) # 288 is the output of phi

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # write the forward function
    def forward(self, state, next_state, action):
        # use elu, as in the paper
        x = F.elu(self.conv_1(state))
        x = F.elu(self.conv_2(x))
        x = F.elu(self.conv_3(x))
        phi_s = self.phi(x)
        # convert [T, 32, 32, 3] to [T, 288] ( with [0] we say we want to preserve T, then flatten the rest by concatenating) 32x32x3 elements
        phi_s = phi_s.view((phi_s.size()[0], -1)).to(torch.float)

        x = F.elu(self.conv_1(next_state))
        x = F.elu(self.conv_2(x))
        x = F.elu(self.conv_3(x))
        phi_s_prime = self.phi(x)
        # convert [T, 32, 32, 3] to [T, 288]
        phi_s_prime = phi_s_prime.view((phi_s_prime.size()[0], -1)).to(torch.float)

        # inverse model
        phi_s_phi_s_prime = torch.cat([phi_s, phi_s_prime], dim=1)
        inverse = self.inverse(phi_s_phi_s_prime) # F.elu(self.inverse(phi_s_phi_s_prime)) # phil tabor does not use activation function here
        pi_logits = self.pi_logits(inverse)

        # forward model
        # from [T] to [T, 1]
        action = action.reshape((action.size()[0], 1))
        phi_s_action = torch.cat([phi_s, action], dim=1)
        dense = self.dense_1(phi_s_action) # F.elu(self.dense_1(phi_s_action))
        phi_hat_s_prime = self.phi_hat_next(dense)

        return pi_logits, phi_s_prime, phi_hat_s_prime

    def calc_loss(self, states, next_states, actions):
        # no need to use [] when converting to tensor because these are lists already
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        # states = torch.stack(states).squeeze()
        # states = states.float().to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)

        pi_logits, phi_s_prime, phi_hat_s_prime = self.forward(states, next_states, actions)

        """ COMPUTE THE LOSSES """
        # inverse model loss (cross entropy between the logits and the actions taken by the agent)
        inverse_loss = nn.CrossEntropyLoss()
        L_I = (1 - self.beta) * inverse_loss(pi_logits, actions.long())

        # forward loss (mse between the predicted phi(s') and the actual phi(s'))
        forward_loss = nn.MSELoss()
        L_F = self.beta * forward_loss(phi_hat_s_prime, phi_s_prime)

        # curiosity loss (intrinsic reward) (L_C)
        # intrinsic_reward = self.alpha * 0.5 * ((phi_hat_s_prime - phi_s_prime).pow(2).mean(dim=1))  # 0.5 comes from ETHA/2 in the paper. Here etha is 1
        # dim=1 because we want a reward for each state in the batch
        intrinsic_reward = self.alpha * 0.5 * ((phi_hat_s_prime - phi_s_prime).pow(2)).mean(dim=1)  # 0.5 comes from ETHA/2 in the paper. Here etha is 1 <- phil tabor version
        return intrinsic_reward, L_I, L_F

