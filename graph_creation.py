# TODO
# 1. each review is a node
# 2. extract arguments & weights
# 3. compute semantic similarity matrix per product!
# 4. link

import networkx as nx
#import duckdb
import pandas as pd
import pickle
from time import time
import gensim
import logging
from nltk.corpus import stopwords
from nltk import download
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from numpy import isnan, isinf
import sys
import multiprocessing as mp
import numpy as np
from functools import partial
import ast
import psutil
from tqdm import tqdm
import os
import gzip
import json
import traceback

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)


def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def distances_matrix(prod, df):
    global model
    k = [x for x in df.loc[df['asin'].str.match(prod), 'ranks']]
    tokens = [y[0].lower().split() for x in k for y in ast.literal_eval(x)]
    # list(filter(lambda y : y not in stop_words, x)))<---------------------------------------- check
    tokens = list(set([" ".join(filter(lambda y : y not in stop_words, x)) for x in tokens]))
    # tokens = list(set([" ".join(x) for x in tokens]))
    df_matrix = pd.DataFrame(index=tokens, columns=tokens)
    for x in range(len(tokens)):
        for y in range(len(tokens)):
            if tokens[x] == " " or tokens[y] == " ":
                dist = 100
            else:
                dist = model.wmdistance(tokens[x], tokens[y])
            df_matrix.iat[x, y] = 100 if isinf(dist) else 0 if isnan(dist) else dist
    return df_matrix


def clusters(prod, df_matrix, df):
    try:
        df_matrix = df_matrix.values[0]
    except Exception as e:
        print(traceback.print_exc())
        print(len(df_matrix))
        print(df_matrix.columns)
        if len(df_matrix) == 2:
            print(df_matrix)
        df_matrix = pd.DataFrame()
    '''try:
        k = [x for x in df.loc[df['asin'].str.contains(prod), 'ranks']]
        tokens = [y[0].lower().split() for x in k for y in ast.literal_eval(x)]
        tokens = list(set([" ".join(list(filter(lambda y: y not in stop_words, x))) for x in tokens]))
    except:
        print("error")
        tokens = []
    '''
    silhouettes = []
    if not df_matrix.empty:
        l = min(len(df_matrix.index), 10)
        for i in range(l):
            cluster = KMeans(n_clusters=(i + 1), random_state=10)
            cluster_labels = cluster.fit_predict(df_matrix)
            try:
                silhouette_avg = silhouette_score(df_matrix, cluster_labels)
            except:
                silhouette_avg = 0
            silhouettes.append(silhouette_avg)
        try:
            optimal_clusters = silhouettes.index(max(silhouettes))
            cluster = KMeans(n_clusters=optimal_clusters + 1, random_state=10)
            cluster_labels = cluster.fit_predict(df_matrix)
        except:
            cluster_labels = []
    else:
        cluster_labels = []
    return cluster_labels

def get_matrix_and_clusters(prod, df):
    global model
    k = [x for x in df.loc[df['asin'].str.match(prod), 'ranks']]
    tokens = [y[0].lower().split() for x in k for y in ast.literal_eval(x)]
    tokens = list(set([" ".join(filter(lambda y : y not in stop_words, x)) for x in tokens]))
    tokens = [x for x in tokens if x != "" and x != " "]
    df_matrix = pd.DataFrame(index=tokens, columns=tokens)
    for x in range(len(tokens)):
        for y in range(len(tokens)):
            if tokens[x] == " " or tokens[y] == " ":
                dist = 100
            else:
                dist = model.wmdistance(tokens[x], tokens[y])
            df_matrix.iat[x, y] = 100 if isinf(dist) else 0 if isnan(dist) else dist
    silhouettes = []
    if not df_matrix.empty:
        for i in range(min(len(df_matrix.index), 10)):
            cluster = KMeans(n_clusters=(i + 1), random_state=10)
            cluster_labels = cluster.fit_predict(df_matrix)
            try:
                silhouette_avg = silhouette_score(df_matrix, cluster_labels)
            except:
                silhouette_avg = 0
            silhouettes.append(silhouette_avg)
        try:
            optimal_clusters = silhouettes.index(max(silhouettes))
            cluster = KMeans(n_clusters=optimal_clusters + 1, random_state=10)
            cluster_labels = cluster.fit_predict(df_matrix)
        except:
            cluster_labels = []
    else:
        cluster_labels = []
    return (df_matrix, cluster_labels)

def add_features(df, ncores, df_reviews):
    psutil.Process().cpu_affinity([ncores])
    print(ncores)
    #matrix = df['prod'].apply(lambda x: distances_matrix(x, df_reviews))
    df['matrix','clusters'] = df['prod'].apply(lambda x: get_matrix_and_clusters(x, df_reviews))  #df['prod'].apply(lambda x: distances_matrix(x, df_reviews))
    print(df)
    #if matrix.empty:
    #    df['clusters'] = []
    #else:
    #    df['clusters'] = df['prod'].apply(lambda x: clusters(x, matrix, df_reviews))
    return df


def parallelize_dataframe(df_to_par, func, n_cores=4):
    print("started")
    df_split = np.array_split(df_to_par, n_cores)
    pool = mp.Pool(n_cores)
    df_res = pd.concat(pool.starmap(func, zip(df_split, range(n_cores))))
    print(df_res)
    print("concatenated")
    pool.close()
    pool.join()
    print("everything closed")
    return df_res


if __name__ == '__main__':
    num_cpus = psutil.cpu_count(logical=False)
    stop_words = stopwords.words('english')

    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    model.init_sims(replace=True)

    files = [dirpath+"/"+file for dirpath, dirnames, filename in os.walk(".") for file in filename if file.endswith("_5_reviews.csv")]

    for file in files:
        print(file)
        df = get_df(file.replace("_5_reviews.csv", "_5.json.gz"))
        df_reviews = pd.read_csv(file,compression="gzip")
        file_prods = file.replace("_5_reviews.csv", "_5_prods.pkl")
        df_prods = pd.read_pickle(file_prods, compression="gzip")
        add_features_df = partial(add_features, df_reviews=df_reviews)
        df_prods = parallelize_dataframe(df_prods, func=add_features_df, n_cores=num_cpus)
        df_prods.to_pickle(file_prods.replace("_5_prods.pkl", "_5_prods_mc.pkl"))
        print("DONE")
