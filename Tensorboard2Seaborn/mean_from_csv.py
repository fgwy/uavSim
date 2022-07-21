import glob
import pandas as pd
import numpy as np

# specifying the path to csv files
path = "./data/multi_last"

# csv files in the path
files = glob.glob(path + "/*.csv")

# defining an empty list to store
# content
data_frame = pd.DataFrame()
content = []

# checking all the csv files in the
# specified path
for filename in files:
    # reading content of csv file
    # content.append(filename)
    df = pd.read_csv(filename, index_col=None)
    mean_y = np.mean(df.Value)
    content.append([mean_y, filename])
    print(mean_y, filename)

# converting content to data frame
# data_frame = pd.concat(content)
# print(content)