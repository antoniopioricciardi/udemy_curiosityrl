"""shared Adam class derived from the Adam class
Constructor iterates over parameters and the various parameter groups, set the state variables and then
share the memory for exponential average and squared exponential average """
import torch


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

