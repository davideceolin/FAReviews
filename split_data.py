# split data file into a number of separate files
import pandas as pd
import numpy as np
import math
import os

# review file to split
file = './Data/AMAZON_FASHION.json.gz'
# assuming file extension = json.gz, get stem of filename
outfile_stem = os.path.splitext(os.path.splitext(file)[0])[0]
# number of files to split the main file into
n = 18

df = pd.read_json(file, compression='gzip', lines=True)
df['reviewText'] = df['reviewText'].fillna('')
# remove duplicate dows
df_reviews_2 = df.loc[df.astype(str).drop_duplicates().index]
df_reviews = df_reviews_2.reset_index()

# count number of reviews per product
df_count = df_reviews['asin'].value_counts()

# define numpy array
prod = np.empty(shape=(math.ceil(len(df_count)/n), n), dtype='object')

for i in range(len(prod)):
    for j in range(0, n):
        if i*n + j < len(df_count):
            prod[i, j] = df_count.index[i*n + j]


for i in range(0, n):
    df_out = df.loc[df['asin'].isin(prod[:, i])].reset_index()
    df_out.drop(['index'], axis=1)
    out_filename = outfile_stem + '_split_' + str(i) + '.json.gz'
    df_out.to_json(out_filename, compression='gzip')
