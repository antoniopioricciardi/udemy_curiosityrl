import gym
import numpy as np
import cv2
import collections

class RepeatAction(gym.Wrapper):
    def __init__(self, env=None, repeat=4, fire_first=False):
        # fire_first: in certain envs the agent have to fire to start the env, as in pong
        # the agent can figure it out alone sometimes
        super(RepeatAction, self).__init__(env)
        self.env = env
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.fire_first = fire_first

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.fire_first:
            # get_action_meanings returns a list of strings (['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'])
            fire_act_idx = self.env.unwrapped.get_action_meanings().index('FIRE')
            obs, _, _, _ = self.env.step(fire_act_idx)
        return obs


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.env = env
        # pytorch expects channel firsts
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # self.shape[1:] because it is 42x42 (shape[0] indicates the num_channels)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA)
        # uint8 because we have obs as ints for 0 to 255 (might not be needed), since we're converting to float
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.repeat = repeat
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0),
                                                dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        obs = self.env.reset()
        for i in range(self.repeat):
            self.stack.append(obs)
        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(42, 42, 1), repeat=4, fire_first=False):
    env = gym.make(env_name)
    env = RepeatAction(env, repeat, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    return env




