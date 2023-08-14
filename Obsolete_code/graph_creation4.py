# TODO
# 1. each review is a node
# 2. extract arguments & weights
# 3. compute semantic similarity matrix per product!
# 4. link

import ast
import multiprocessing as mp
import os
from functools import partial

import gensim
import pandas as pd
import psutil
from nltk.corpus import stopwords
from numpy import isinf, isnan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time as ttime

stop_words = stopwords.words('english')
model = gensim.models.KeyedVectors.load_word2vec_format(
            'GoogleNews-vectors-negative300.bin.gz', binary=True)


def get_matrix_and_clusters(prod, df, k=-1):
    # Get tokens
    t1 = ttime.time()
    k = [x for x in df.loc[df['asin'].str.match(prod), 'ranks']]
    tokens = [y[0].lower().split() for x in k for y in ast.literal_eval(str(x))]
    tokens = list(set([" ".join(filter(lambda y: y not in stop_words, x)) for x in tokens]))
    tokens = [x for x in tokens if x != "" and x != " "]
    # Create matrix with wmdistance values
    df_matrix = pd.DataFrame(index=tokens, columns=tokens, data=0)
    for x in range(len(tokens)):
        for y in range(x+1):
            dist = model.wmdistance(tokens[x], tokens[y])
            df_matrix.iat[x, y] = df_matrix.iat[y, x] = (100 if isinf(dist) else 0 if isnan(dist)
                                                         else dist)
    tm = ttime.time()
    # Calculate clusters
    silhouettes = []
    if not df_matrix.empty:
        for i in range(min(len(df_matrix.index), 10)):
            cluster = KMeans(n_clusters=(i + 1), random_state=10)
            cluster_labels = cluster.fit_predict(df_matrix)
            try:
                silhouette_avg = silhouette_score(df_matrix, cluster_labels)
            except Exception:
                silhouette_avg = 0
            silhouettes.append(silhouette_avg)
        try:
            if k == -1:
                optimal_clusters = 1
            else:
                optimal_clusters = silhouettes.index(max(silhouettes))
            cluster = KMeans(n_clusters=optimal_clusters + 1, random_state=10)
            cluster_labels = cluster.fit_predict(df_matrix)
        except Exception:
            cluster_labels = []
    else:
        cluster_labels = []
    tc = ttime.time()
    print(prod, len(tokens), tm-t1, tc-tm, tc-t1, mp.current_process())
    return prod, df_matrix, cluster_labels


def parallelize_get_mc(func, prods, n_cores):
    pool = mp.Pool(n_cores)
    prods_mc = pool.map(func, prods)
    df_prods_mc = pd.DataFrame(columns=['prod', 'matrix', 'clusters'], data=prods_mc)
    pool.close()
    pool.join()
    return df_prods_mc


def run_graph_creation(df_reviews, prods, nc):
    p_get_mc = partial(get_matrix_and_clusters, df=df_reviews)
    df_prods_mc = parallelize_get_mc(p_get_mc, prods, nc)
    return df_prods_mc


if __name__ == '__main__':
    file = input('Please provide the file path (csv) for the input data (reviews): ')
    file2 = input('Please provide the file path (pkl) for the input data (product list): ')
    print('Your logical CPU count is:', psutil.cpu_count(logical=True))
    nc = int(input('Define number of usable cpu (optional, default is 8): ') or 8)
    savename = str(input("Please provide the name of the (existing) output folder to which you " +
                         "want to save the output: " or ""))
    df_reviews = pd.read_csv(file, compression="gzip")
    df_prods = pd.read_pickle(file2, compression="gzip")
    prods = df_prods['prod'].to_list()
    df_prods_mc = run_graph_creation(df_reviews, prods, nc)
    try:
        bn = os.path.basename(file)
        output_path = os.path.join(savename, bn[:bn.index('_reviews.')])
        df_prods_mc.to_pickle(output_path + "_prods_mc.pkl", compression="gzip")
    except Exception:
        print('Failed to save the output of graph_creation')
