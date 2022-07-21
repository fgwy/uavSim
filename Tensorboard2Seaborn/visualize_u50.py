from curve_vis import CurveVis
import glob

import asyncio
import time

# def background(f):
#     def wrapped(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
#
#     return wrapped
#
# y_label = ['landing',
#                'CR',
#                'CRAL']
# suptitle = ['Urban50',
#                 'Multimap',
#                 'Maze80']
#
#
# @background
# def plot_csvs(i,n):
#     dir = ""
#
#     smooth_k = 100
#
#     # specifying the path to csv files
#     path = "./data/eval/"
#
#     NP = path + suptitle[n] + '/' + y_label[i]
#     print(NP)
#     csv_files = sorted(glob.glob(NP + "/*.csv"))
#     # csv_files =
#     # csv_files = ["run-training_flat_multimap_2403_training-tag-cumulative_reward.csv",
#     #              "run-training_flat_multimap_masked_0604_training-tag-cumulative_reward.csv"]
#     print(csv_files)
#     labels = ['masked Flat Agent gr.',
#               'masked Flat Agent st.',
#               'non-masked Flat Agent gr.',
#               'non-masked Flat Agent st.',
#               'H2D2 greedy',
#               'H2D2 stochastic']
#
#     # labels = [
#     #           'masked Flat Agent',
#     # 'non-masked Flat Agent',
#     #           'H2D2']
#
#     cv = CurveVis(
#         csv_file=csv_files,
#         dir=dir,
#         labels=labels,
#         x_label='episodes',
#         y_label=y_label[i],
#         smooth_k=smooth_k,
#         dpi=200.0,
#         suptitle=suptitle[n],
#         mean=True,
#         remove_zeros=False,
#         legend_outside=False,
#         keep_range=False,
#         min_max=(0.55, 0.95)
#     )
#     cv.show()

# for n in range(len(suptitle)):
#     for i in range(len(y_label)):
#         plot_csvs(i, n)

# csv_file,
#             data_form='csv',
#             x_label='step(s)',
#             y_label='loss',
#             labels=['curve1'],
#             smooth_k=5
#
# csv_file = ["data/training_flat/run-training_flat_maze80_2403_training-tag-cral.csv",
#             "data/training_flat/run-training_flat_maze80_masked_2903_training-tag-cral.csv"]
# labels = [
#     "unmasked",
#     "unmasked_2"
# ]
# csv_files = [['run-training_flat_urban50_0904_eval_greedy_training-tag-cr.csv',
#             'run-training_flat_urban50_0604_eval_stoch_training-tag-cr.csv'],[
#             'run-training_flat_urban50_0904_eval_unmskdtr_mskdeval_greedy_training-tag-cr.csv',
#             'run-training_flat_urban50_0904_eval_unmskdtr_mskdeval_stoch_training-tag-cr.csv'],[
#             'run-training_h2d2_urban50_eval_0804_greedy_training-tag-cr.csv',
#             'run-training_h2d2_urban50_eval_0804_stoch_training-tag-cr.csv'
#             ]]
# labels = [["NM greedy Flat Agent",
#           "NM stochastic Flat Agent"],[
#           "NM tr M eval gr. Flat Agent",
#           "NM tr M eval st. Flat Agent"],[
#           "greedy H2D2",
#           "stochastic H2D2"]]
#
# csv_files = ['run-training_h2d2_urban50_astar_2403_training-tag-cral.csv',
#              'run-training_flat_urban50_2403_test-tag-cral.csv',
#              'run-training_flat_urban50_masked_2903_test-tag-cral.csv']
# labels = ['H2D2', 'Flat Agent', 'masked Flat Agent']
# dir = 'data/eval/'



dir = 'data/'
dir='./data/multi/cral'
dir=""
y_label = ['landing',
    'CR',
    'CRAL']

suptitle=['Urban50',
          'Multimap',
          'Maze80']
smooth_k=100

# specifying the path to csv files
path = "./data/eval/"

# for n in range(len(suptitle)):
#     for i in range(len(y_label)):
#     csv files in the path
i=1
n=0
NP = path + suptitle[n] + '/' + y_label[i]
print(NP)
csv_files = sorted(glob.glob(NP+ "/*.csv"))
# csv_files =
# csv_files = ["run-training_flat_multimap_2403_training-tag-cumulative_reward.csv",
#              "run-training_flat_multimap_masked_0604_training-tag-cumulative_reward.csv"]
print(csv_files)
labels = [ 'masked Flat Agent gr.',
          'masked Flat Agent st.',
           'non-masked Flat Agent gr.',
           'non-masked Flat Agent st.',
          'H2D2 greedy',
          'H2D2 stochastic']

# labels = [
#           'masked Flat Agent',
# 'non-masked Flat Agent',
#           'H2D2']



cv = CurveVis(
    csv_file=csv_files,
    dir=dir,
    labels=labels,
    x_label='episodes',
    y_label=y_label[i],
    smooth_k=smooth_k,
    dpi=200.0,
    suptitle=suptitle[n],
    mean=True,
    remove_zeros=False,
    legend_outside=False,
    keep_range=False,
    min_max=(0.55, 0.95)
)
cv.show()
# for i in range(len(csv_files)):
#     cv = CurveVis(
#     csv_file=csv_files[i],
#     dir=dir,
#     labels=labels[i],
#     x_label='steps x1000',
#     y_label=y_label,
#     smooth_k=smooth_k,
#     dpi=200.0,
#     suptitle=suptitle,
#     mean=False,
#     remove_zeros=False,
#     legend_outside=False,
#     keep_range=False,
#     min_max=(0.55, 0.95)
#     )
#     cv.show()