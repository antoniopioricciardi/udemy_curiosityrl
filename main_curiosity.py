import os
import torch.multiprocessing as mp
from curiosity.parallel_env import ParallelEnv
from curiosity.actor_critic import ActorCritic
from miniworld import gym_miniworld
import time

os.environ['OMP_NUM_THREADS'] = '1'

"""
def worker:
    init unique envs
    init unique agent
    play episodes
"""
# worker works as a "single threaded" function, no need to worry about multithread here

if __name__ == '__main__':
    # create a folder to save the results
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('models'):
        os.makedirs('models')

    start = time.time()
    mp.set_start_method('spawn')
    global_ep = mp.Value('i', 0)
    #mp.set_start_method('spawn') # faster but only linux/macos
    # env_id = 'PongNoFrameskip-v4'
    env_id = 'MiniWorld-Hallway-v0'
    # env_id = 'MiniWorld-FourRooms-v0'
    n_threads = 4
    n_actions = 3
    input_shape = [4, 42, 42]
    env = ParallelEnv(env_id=env_id, num_threads=n_threads, n_actions=n_actions, global_idx=global_ep, input_shape=input_shape, icm=True)
    print('RUNNING TIME:', time.time() - start)
