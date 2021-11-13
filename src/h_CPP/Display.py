import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtplt
import time

from src.CPP.Display import CPPDisplay


class h_CPPDisplay(CPPDisplay):

    def __init__(self):
        super().__init__()

    def plot_map(self, state, terminal=False): # , h_target_idx):
        colors = 'white blue lime red yellow cyan'.split()
        cmap = mtplt.colors.ListedColormap(colors, name='colors', N=None)
        data = state.no_fly_zone
        target = ~data*state.target
        target = ~state.h_target*target
        lz = state.landing_zone*~state.h_target
        data = data*1
        data += lz * 5
        # data[h_target_idx[0], h_target_idx[1]] = 2
        data += state.h_target*2
        data += target * 4
        data[state.position[1], state.position[0]] = 3

        [m, n] = np.shape(data)
        plt.imshow(data, alpha=1, cmap=cmap, vmin=0, vmax=5)
        plt.xticks(np.arange(n))
        plt.yticks(np.arange(m))
        textstr = f'mb: {state.initial_movement_budget} \ncurrent mb: {state.movement_budget} \nllmb: {state.initial_ll_movement_budget} \ncurrent llmb: {state.current_ll_mb} \n'

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        plt.text(-15, 0.95, textstr, fontsize=11,
                verticalalignment='top', bbox=props) # transform=plt.transAxes,
        plt.subplots_adjust(left=0.25)
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()
        if terminal:
            plt.gcf()
            plt.close('all')



    def save_plot_map(self, trajectory): # TODO: save a plot of a whole episode in one map with trail etc..
        pass