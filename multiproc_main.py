import os
import torch.multiprocessing as mp
import gym

os.environ['SET_NUM_THREAD'] = '1'

env_name = 'CartPole-v1'
env = gym.make(env_name)

observation = env.reset()


def worker(action_space):
    """
    single arg as input, print "hello name"
    """
    return action_space.sample()


mp.set_start_method('spawn')
# pass a target function, that is what the multiprocessing package is going to call when the process start
# pass the argument, too
process = mp.Process(target=worker, args=(env,))
# start the process to start new thread
process.start()
# call join so that the main process doesn't end before the child process (avoid zombie processes)
process.join()


done = False
while not done:
    obs = agent.random_action(env.action_space)