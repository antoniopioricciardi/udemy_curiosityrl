import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv
os.environ['SET_NUM_THREAD'] = '4'
from actor_critic import ActorCritic

"""
def worker:
    init unique envs
    init unique agent
    play episodes
"""
# worker works as a "single threaded" function, no need to worry about multithread here

if __name__ == '__main__':
    mp.set_start_method('spawn')
    global_ep = mp.Value('i', 0)
    #mp.set_start_method('spawn') # faster but only linux/macos
    env_id = 'PongNoFrameskip-v4'
    n_threads = 4
    n_actions = 6
    input_shape = [4, 42, 42]
    env = ParallelEnv(env_id=env_id, num_threads=n_threads, n_actions=n_actions, global_idx=global_ep, input_shape=input_shape)
