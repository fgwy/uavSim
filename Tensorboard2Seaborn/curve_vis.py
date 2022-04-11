import math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300
from pandas.core.frame import DataFrame
from tensorboard.backend.event_processing import event_accumulator

sns.set(style="darkgrid")
sns.set_context("paper")

plt_colors=['g', 'b', 'r', 'c', 'm', 'y']

# from tensorboard.data import experimental

class CurveVis():
    sup_data_form = ['csv', 'tb']

    def __init__(
            self,
            dir,
            csv_file,
            data_form='csv',
            x_label='step(s)',
            y_label='loss',
            labels=['curve1'],
            smooth_k=5,
            dpi=200.0,
            suptitle='',
            mean=False,
            remove_zeros=False,
            legend_outside=False,
            keep_range=False,
            min_max=(0, 1)
    ) -> None:
        assert data_form in self.sup_data_form, f"Invalid data form, current support form is {self.sup_data_form}."
        self.data_form = data_form
        self.dir = dir
        self.x_label = x_label
        self.y_label = y_label
        self.smooth_k = smooth_k
        self.dpi = dpi
        self.suptitle = suptitle
        self.mean = mean
        self.remove_zeros = remove_zeros
        self.legend_outside = legend_outside
        self.keep_range = keep_range
        self.min_max = min_max

        if isinstance(csv_file, str):
            csv_file = [csv_file]
        assert len(csv_file) == len(
            labels), f"Invalid num of labels: {len(csv_file)} for csv_files but {len(labels)} for labels"
        self.i=0
        for i in range(len(csv_file)):
            self.i+=1
            # if self.data_form == self.sup_data_form[0]:
            #     data = self._load_csv_data(csv_file[i])[:]
            # elif self.data_form == self.sup_data_form[1]:
            #     data = self._load_tb_data(csv_file[i])
            data = getattr(self, f'_load_{self.data_form}_data')(f'{self.dir}{csv_file[i]}')[:]
            if self.remove_zeros:
                data = pd.read_csv(f'{self.dir}+{csv_file[i]}')
                data.drop(data[data['Value'] == 0].index, inplace=True)
                data.to_csv(f'no_zeros_{csv_file[i]}')
                data = getattr(self, f'_load_{self.data_form}_data')(f'no_zeros_{csv_file[i]}')[:]
            data = self._smooth_data(data, self.smooth_k)
            data = self._index_data(data)
            self._add_curve(data, labels[i])

    def _load_csv_data(self, csv_file):
        return pd.read_csv(csv_file)["Value"].to_numpy()
        # return pd.read_csv(f'{self.dir}{csv_file}')["Value"].to_numpy()

    def _load_tb_data(self, log_file):
        ea = event_accumulator.EventAccumulator(log_file)
        # e = experimental.ExperimentFromDev('eQvoJyXJSOe4exdGep8YNQ')
        # return e.get_scalars()

    def _load_some_other_data():
        '''
            data type must be 'numpy.array'
            data shape is raw data without index
        '''
        raise NotImplementedError

    @staticmethod
    def _smooth_data(data, k=5):
        raw_data = data.copy()
        sz = len(raw_data)

        # padding
        for i in range(int(k / 2)):
            raw_data = np.insert(raw_data, 0, raw_data[0])
            raw_data = np.insert(raw_data, -1, raw_data[-1])
        sm_data = []
        # for i in range(k-1, sz):
        #     sm_data_i = []
        #     for j in range(k-1, -1, -1):
        #         sm_data_i.append(raw_data[i-j])
        #     sm_data.append(sm_data_i)
        for i in range(int(k / 2), sz + int(k / 2)):
            sm_data_i = []
            for j in range(-int(k / 2), int(k / 2)):
                sm_data_i.append(raw_data[i + j])
            sm_data.append(sm_data_i)
        return np.array(sm_data)

    @staticmethod
    def _index_data(data):
        sz = len(data)
        id_data = []
        for i in range(sz):
            for j in data[i]:
                id_data.append([i, j])
        return id_data

    def _add_curve(self, raw_data, curve_label):
        data = DataFrame(raw_data, columns=['x', 'y'])

        ax = sns.lineplot(x='x', y='y', linestyle='solid', linewidth=1, markersize=6, data=data, label=curve_label, color=plt_colors[self.i])
        if self.mean:
            mean_y = [np.mean(data.y)] * len(data.x)
            print(mean_y[0], self.i)
            ax = sns.lineplot(x=data.x, y=mean_y, linestyle='--', linewidth=0.5, markersize=6, label=f"mean {curve_label}", color=plt_colors[self.i])

    def show(self):
        # plt.figure.set_dpi(self.dpi)
        plt.gcf().set_dpi(self.dpi)
        if self.legend_outside:
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        if self.keep_range:
            plt.ylim(self.min_max)
        plt.suptitle(self.suptitle, fontweight="bold")
        plt.xlabel(self.x_label, fontsize=14)
        plt.ylabel(self.y_label, fontsize=14)
        plt.show()