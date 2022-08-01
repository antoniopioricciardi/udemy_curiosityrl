import os
import torch.multiprocessing as mp

os.environ['SET_NUM_THREAD'] = '1'


def worker(name):
    """
    single arg as input, print "hello name"
    """

    print('hello', name)


if __name__ == '__main__':
    """
    tell os how to create new thread, different options that tells how data is shared between the parent
    and child threads, and os we're using. 'spawn' is the most available start method
    """
    mp.set_start_method('spawn')
    # pass a target function, that is what the multiprocessing package is going to call when the process start
    # pass the argument, too
    process = mp.Process(target=worker, args=('dale',))
    # start the process to start new thread
    process.start()
    # call join so that the main process doesn't end before the child process (avoid zombie processes)
    process.join()