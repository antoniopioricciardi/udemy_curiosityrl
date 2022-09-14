import os
from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

first_plot = True
n_games_to_avg = [50, 100]

def plot_learning_curve(x, scores, figure_file):
    global first_plot
    global n_games_to_avg
    for n_games in n_games_to_avg:
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i - n_games):i + 1])
            # running_avg[i] = np.mean(scores[max(0, i-n_games_to_avg):(i+1)])
        plt.plot(x, running_avg, label=str(n_games) + " games avg")
    plt.title(f"Running avg of previous games")
    if first_plot:
        plt.legend()
        first_plot = False
    plt.savefig(os.path.join('plots', figure_file))