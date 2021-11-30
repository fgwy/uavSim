import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtplt
import time
import os


from src.CPP.Display import CPPDisplay


class h_CPPDisplay(CPPDisplay):

    def __init__(self):
        super().__init__()
        self.my_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/plots/')
        try:
            os.mkdir(self.my_path)
        except:
            print("Dir already present")

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
        plt.subplots_adjust(left=0.3)
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
        if terminal:
            plt.gcf()
            plt.close('all')



    def save_plot_map(self, trajectory, episode_num, testing, name, las, cum_rew): # TODO: save a plot of a whole episode in one map with trail etc..
        state = trajectory[0]
        ending_state = trajectory[-1]
        positions = []
        if testing:
            val = 'testing'
        else:
            val = 'training'
        for i in range(len(trajectory)):
            positions.append((trajectory[i].position[1], trajectory[i].position[0]))
        # print(positions)
        colors = 'white blue green red yellow cyan'.split()
        cmap = mtplt.colors.ListedColormap(colors, name='colors', N=None)

        fig, axs = plt.subplots(1,2, tight_layout=True)
        fig.suptitle(f'Episode: {episode_num}')

        data = state.no_fly_zone
        target = ~data * state.target
        target = ~state.h_target * target
        lz = state.landing_zone * ~state.h_target
        data = data * 1
        data += lz * 5
        # data[h_target_idx[0], h_target_idx[1]] = 2
        data += state.h_target * 2
        data += target * 4
        data[state.position[1], state.position[0]] = 3

        data_end = ending_state.no_fly_zone
        target = ~data_end * ending_state.target
        target = ~ending_state.h_target * target
        lz = ending_state.landing_zone * ~ending_state.h_target
        data_end = data_end * 1
        data_end += lz * 5
        # data[h_target_idx[0], h_target_idx[1]] = 2
        # data_end += ending_state.h_target * 2
        data_end += target * 4
        for i in range(len(positions)):
            data_end[positions[i][0], positions[i][1]] = 3
        for i in range(len(trajectory)):
            data_end += trajectory[i].h_target*2

        textstr = f'Ended by landing: {ending_state.landed} \nMode: {val}\nLAS: {las}\nCumulative_reward: {cum_rew}'

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
        plt.text(-70, 60, textstr, fontsize=11,
                 verticalalignment='top', bbox=props)  # transform=plt.transAxes,
        plt.subplots_adjust(left=0.3)

        axs[0].imshow(data, alpha=1, cmap=cmap, vmin=0, vmax=5)
        axs[1].imshow(data_end, alpha=1, cmap=cmap, vmin=0, vmax=5)
        axs[0].set_title('First State')
        axs[1].set_title('Ending State')

        if not las:
            my_file = f'{name}/{val}_ep{episode_num}.png'
        else:
            my_file = f'{name}/{val}_ep{episode_num}_las_{las}.png'

        path = os.path.join(self.my_path, my_file)
        try:
            plt.savefig(path, dpi=600)
        except:
            print(f"Path dosen't exist: {self.my_path}\nCreating New Path")
            os.mkdir(self.my_path + name)
            plt.savefig(path, dpi=600)
            print(f"Created dir and saved Plots in: {path}")

        else:
            print(f"Saved Plots in: {path}")
        fig.clf()