import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtplt
import time
import os
import copy

import seaborn as sns # sns.set_theme()
from matplotlib import patches

from src.CPP.Display import CPPDisplay


class h_CPPDisplay(CPPDisplay):

    def __init__(self):
        super().__init__()

        self.my_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/plots/')
        try:
            os.mkdir(self.my_path)
        except:
            print("Dir already present")

        colors = 'white brown lime red yellow blue'.split()
        self.cmap = mtplt.colors.ListedColormap(colors, name='colors', N=None)

    def init_plt(self, state):

        self.ax = self.get_plt(state)

    def get_plt(self, state, axs=None):

        if axs is None:
            axs = plt.axes()
        else:
            pass
        area_y_max, area_x_max = state.shape
        obst = state.obstacles
        # print(area_x_max, area_y_max)
        if (area_x_max, area_y_max) == (64, 64):
            tick_labels_x = np.arange(0, area_x_max, 4)
            tick_labels_y = np.arange(0, area_y_max, 4)
            self.arrow_scale = 14
            self.marker_size = 6
        elif (area_x_max, area_y_max) == (32, 32):
            tick_labels_x = np.arange(0, area_x_max, 2)
            tick_labels_y = np.arange(0, area_y_max, 2)
            self.arrow_scale = 8
            self.marker_size = 15
        elif (area_x_max, area_y_max) == (50, 50):
            tick_labels_x = np.arange(0, area_x_max, 4)
            tick_labels_y = np.arange(0, area_y_max, 4)
            self.arrow_scale = 12
            self.marker_size = 8
        else:
            tick_labels_x = np.arange(0, area_x_max, 1)
            tick_labels_y = np.arange(0, area_y_max, 1)
            self.arrow_scale = 5
            self.marker_size = 15

        try:
            for ax in axs:
                plt.sca(ax)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.xticks(tick_labels_x)
                plt.yticks(tick_labels_y)
                plt.axis([0, area_x_max, area_y_max, 0])

                for i in range(obst.shape[0]):
                    for j in range(obst.shape[1]):
                        if obst[j, i]:
                            rect = patches.Rectangle((i-0.5, j-0.5), 1, 1, fill=None, hatch='////', edgecolor="Black")
                            # ax.add_patch(rect)
                            ax.add_patch(rect)
        except:
            plt.sca(axs)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xticks(tick_labels_x)
            plt.yticks(tick_labels_y)
            plt.axis([0, area_x_max, area_y_max, 0])

            for i in range(obst.shape[0]):
                for j in range(obst.shape[1]):
                    if obst[j, i]:
                        rect = patches.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=None, hatch='////', edgecolor="Black")
                        # ax.add_patch(rect)
                        axs.add_patch(rect)
        else:
            pass
        return axs

    def plot_map(self, state, terminal=False):  # , h_target_idx):

        data = self.get_data(state)

        fig_size = 5.5
        # fig, ax = plt.subplots(1, 1, figsize=[fig_size, fig_size])

        # ax = self.get_plt(ax, state)
        plt.imshow(data, alpha=1, cmap=self.cmap, vmin=0, vmax=5)
        ax = plt.gca()
        ax = self.get_plt(state, ax)
        plt.sca(ax)


        [m, n] = np.shape(data)
        # for dp in data:
        #     print(f'data point after: {dp}')


        # plt.xticks(np.arange(n))
        # plt.yticks(np.arange(m))
        textstr = f'mb: {state.initial_movement_budget} \ncurrent mb: {state.movement_budget} \nllmb: {state.initial_ll_movement_budget} \ncurrent llmb: {state.current_ll_mb} \n'

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        plt.text(-15, 0.95, textstr, fontsize=11,
                 verticalalignment='top', bbox=props)  # transform=plt.transAxes,
        plt.subplots_adjust(left=0.3)
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()
        if terminal:
            # print(f'Terminal Fig')
            plt.gcf()
            plt.close('all')

    def save_plot_map(self, trajectory, episode_num, testing, name, las,
                      cum_rew, hl_steps):  # TODO: save a plot of a whole episode in one map with trail etc..
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
        # colors = 'white blue green red yellow cyan'.split()
        # cmap = mtplt.colors.ListedColormap(colors, name='colors', N=None)

        fig, axs = plt.subplots(1, 2) #, tight_layout=True)
        fig.suptitle(f'Episode: {episode_num}')

        axs = self.get_plt(state, axs)

        # data = state.no_fly_zone
        # target = ~data * state.target
        # target = ~state.h_target * target
        # lz = state.landing_zone * ~state.h_target
        # data = data * 1
        # data += lz * 5
        # # data[h_target_idx[0], h_target_idx[1]] = 2
        # data += state.h_target * 2
        # data += target * 4
        # data[state.position[1], state.position[0]] = 3

        data = self.get_data(state)

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
            data_end += trajectory[i].h_target * 2

        textstr = f'Ended by landing: {ending_state.landed} \nMode: {val}\nLAS: {las}\nCumulative_reward: {cum_rew:.2f}\nTotal steps in ep: {ending_state.initial_movement_budget - ending_state.movement_budget}\nHL-Steps: {hl_steps}'

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
        plt.text(-70, 60, textstr, fontsize=11,
                 verticalalignment='top', bbox=props)  # transform=plt.transAxes,
        plt.subplots_adjust(left=0.3)

        axs[0].imshow(data, alpha=1, cmap=self.cmap, vmin=0, vmax=5)
        axs[1].imshow(data_end, alpha=1, cmap=self.cmap, vmin=0, vmax=5)
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

    def save_state_and_hm(self, state, q_vals, name, episode_num):
        grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

        f, (ax, cbar_ax, lm_ax) = plt.subplots(3, gridspec_kw=grid_kws)

        q_vals.squeeze()
        print(q_vals.shape)

        q = q_vals[:-1]
        root = int(np.sqrt(q.size))
        q_vals = q.reshape(root, root)

        ax = sns.heatmap(q_vals, ax=ax,

                         cbar_ax=cbar_ax,

                         cbar_kws={"orientation": "horizontal"})

        data = self.get_data(state)
        lm_ax = data

        my_file = f'{name}/heat_map_ep{episode_num}_figure.png'
        self.save_fig(plt, my_file, name)
        plt.show()

        plt.clf()
        plt.cla()

    def save_fig(self, plt, my_file, name):

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

    def save_lm_tm_and_q(self, state, p):
        data = self.get_data(copy.deepcopy(state))
        data_lm = self.get_data_lm(copy.deepcopy(state))

        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(data)
        fig.add_subplot(1, 3, 2)
        # plt.imshow(lm[:, :, 3])
        plt.imshow(data_lm)
        fig.add_subplot(1, 3, 3)
        plt.imshow(p, cmap='hot', interpolation='nearest')
        plt.show()

        p_val = p[:-1]

        ### put -inf on view
        p = p_val.reshape((self.params.goal_size, self.params.goal_size))

        # goal = tf.one_hot(a,
        #                   depth=self.num_actions_hl - 1).numpy().reshape(self.params.goal_size,
        #                                                                  self.params.goal_size).astype(int)
        d = 1
        goal = np.pad(goal, ((d, d), (d, d)))
        data = self.get_data(copy.deepcopy(state))
        data_lm = self.get_data_lm(copy.deepcopy(state))

        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(data)
        fig.add_subplot(1, 3, 2)
        # plt.imshow(lm[:, :, 3])
        data_lm += goal
        plt.imshow(data_lm)
        fig.add_subplot(1, 3, 3)
        plt.imshow(p, cmap='hot', interpolation='nearest')
        plt.show()


    def get_data(self, state):

        data = state.no_fly_zone.astype(bool)
        target = ~data * state.target
        # for dp in target:
        #     print(f'target point: {dp}')
        target = ~state.h_target * target
        # for dp in data:
        #     print(f'target after point: {dp}')
        lz = state.landing_zone * ~state.h_target
        data = data * 1
        # for dp in data:
        #     print(f'data point before: {dp}')
        data += lz * 5
        # data[h_target_idx[0], h_target_idx[1]] = 2
        data += state.h_target * 2
        data += target * 4
        data[state.position[1], state.position[0]] = 3

        return data

    def get_lm_data(self, state):
        lm = state.get_local_map().astype(bool)
        data = lm[:, :, 0]*1
        data += (~lm[:, :, 0].astype(bool) *1) * lm[:, :, 1] *2
        data += (~data.astype(bool) *1) *lm[:,:,2] *3
        data += (~data.astype(bool) * 1) * lm[:, :, 3] * 4
        data[state.position[1], state.position[0]] = 5
        # data += (~data.astype(bool) * 1) * lm[:, :, 4] * 5
        return data