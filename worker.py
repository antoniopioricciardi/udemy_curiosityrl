import gym
import numpy as np
import torch
from actor_critic import ActorCritic
from memory import Memory
from utils import plot_learning_curve
from custom_env import make_env

"""
init unique env
init unique agent
play episodes
"""


def worker(name, input_shape, n_actions, global_agent, optimizer, env_id, n_threads, global_idx):
    T_MAX = 20
    local_agent = ActorCritic(input_shape, n_actions)

    memory = Memory()
    """ when we call make, frame buffer must have channels last, but the input_shape has channel first, so we swap """
    frame_buffer = [input_shape[1], input_shape[2], 1]  # we use 1 because we know it will be grayscale
    env = make_env(env_id, shape=frame_buffer)

    # instantiate variables to dictate we want to play and keep track of the scores
    episode, max_eps, total_timestep, scores = 0, 1000, 0, []
    # iterate over eps
    while episode < max_eps:
        obs = env.reset()
        score, done, ep_step = 0, False, 0
        hx = torch.zeros(1,1, 256)
        while not done:
            obs = torch.tensor([obs], dtype=torch.float)
            action, value, log_prob, hx = local_agent(obs, hx)
            # action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            memory.remember(reward, value, log_prob)
            score += reward
            obs = next_obs
            ep_step += 1
            total_timestep += 1
            if ep_step % T_MAX == 0 or done:
                rewards, values, log_probs = memory.sample_memory()
                loss = local_agent.calc_cost(next_obs, hx, done, rewards, values, log_probs)
                local_agent.zero_grad()
                """detach() operates on a tensor and returns the same tensor, which will be detached from the computation
                graph at this point, so that the backward pass will stop at this point.
                detach_() is the inplace operation of detach()"""
                hx = hx.detach_()  # we use detach_ because
                loss.backward()
                # 40 is max_norm
                torch.nn.utils.clip_grad_norm(local_agent.parameters(), 40)
                for local_param, global_param in zip(local_agent.parameters(), global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())
                memory.clear_memory()
        episode += 1
        # we use episode instead of global_idx.get_lock() ....
        # with global_idx.get_lock():
        #     global_idx.value += 1
        """ eval for agent n.1 (there are other means, this is the easier) """
        if name == "1":
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print('A3C episode {} thread {} of {} steps {:.2f}M score {:.2f}, avg score (100) {:.2f}'
                  .format(episode, name, n_threads, total_timestep/1e6, score, avg_score))

    # plot learning curve, only for first agent
    if name == "1":
        x = [z for z in range(episode)]
        plot_learning_curve(x, scores, "A3C_pong_final.png")
