import os
import torch.multiprocessing as mp
from curiosity.actor_critic import ActorCritic  # for the global agent
from shared_adam import SharedAdam
from curiosity.worker import worker
from curiosity.icm import ICM


os.environ['SET_NUM_THREAD'] = '4'


"""class ParallelEnv must have
    init:
        thread names
        list of mp.Process functions, one per thread
        start all processes
        join all processes
"""

class ParallelEnv:
    def __init__(self, env_id, global_idx, input_shape, n_actions, num_threads, icm=False):
        # needed to track thread execution
        thread_names = [str(i) for i in range(num_threads)]
        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        global_optim = SharedAdam(global_actor_critic.parameters(), lr=1e-4)

        if icm:
            global_icm = ICM(input_shape, n_actions)
            global_icm.share_memory()
            global_icm_optim = SharedAdam(global_icm.parameters(), lr=1e-4)
        else:
            global_icm = None
            global_icm_optim = None



        # target is the function we want to be called by each thread, give name end env_id as input to
        # worker function
        self.processes = [mp.Process(target=worker,
                                     args=(name, input_shape, n_actions, global_actor_critic, global_optim, env_id, num_threads, global_idx,
                                     global_icm, global_icm_optim, icm))
                          for name in thread_names]
        # start the process to start new thread
        [p.start() for p in self.processes]
        [p.join() for p in self.processes]




