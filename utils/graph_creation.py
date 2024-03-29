import ast
import itertools
import multiprocessing as mp
import os
import platform
from functools import partial

import gensim
import numpy as np
import pandas as pd
import psutil
from nltk.corpus import stopwords
from numpy import isinf, isnan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_model():
    """Load the pre-trained word2vec model and stop words list."""
    stop_words = stopwords.words('english')
    model = gensim.models.KeyedVectors.load_word2vec_format(
                'GoogleNews-vectors-negative300.bin.gz', binary=True)
    model.init_sims(replace=True)
    return model, stop_words


def get_matrix_and_clusters(prod, df, model, stop_words, k=-1):
    """
    Compute the distance matrix (wmdistance between all tokens of one product)
    and clusters (kmeans cluster, clustering tokens based on the wm distance)
    for a given product.

    Args:
        prod (str): The df with product asin.
        df (pandas.DataFrame): The reviews dataframe.
        model: The pre-trained word2vec model.
        stop_words (list): List of stop words.

    Returns:
        tuple: A tuple (df_matrix, cluster_labels) containing the computed distance matrix and
        cluster labels.
    """
    # Get tokens
    k = [x for x in df.loc[df['asin'].str.match(prod), 'ranks']]
    tokens = [y[0].lower().split() for x in k for y in ast.literal_eval(str(x))]
    tokens = list(set([" ".join(filter(lambda y: y not in stop_words, x)) for x in tokens]))
    tokens = [x for x in tokens if x != "" and x != " "]

    # Create matrix with wmdistance values
    combis = list(itertools.combinations(tokens, 2))
    dists = list(itertools.starmap(model.wmdistance, combis))
    dists = [100 if isinf(dist) else 0 if isnan(dist) else dist for dist in dists]
    data = np.zeros((len(tokens), len(tokens)))
    data[np.triu_indices(len(tokens), 1)] = dists
    data = data + data.T
    df_matrix = pd.DataFrame(data, index=tokens, columns=tokens)

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
    return df_matrix, cluster_labels


def add_features(df, ncores, df_reviews, model, stop_words):
    """Apply get_matrix_and_clusters to the dataframe with all reviews (df_reviews), and add the
    results to the columnsm 'matrix' and 'clusters' respectively in the product dataframe (df).

    Args:
        df (pandas.DataFrame): The dataframe with products
        ncores (int): The number of cores to use.
        df_reviews (pandas.DataFrame): The datframe with reviews.
        model: The pre-trained word2vec model.
        stop_words (list): List of stop words.

    Returns:
        df: The product dataframe with added features.
    """
    if platform.system() != 'Darwin':
        psutil.Process().cpu_affinity([ncores])
    df['matrix'], df['clusters'] = zip(*df['prod'].apply(lambda x:
                                                         get_matrix_and_clusters(x,
                                                                                 df_reviews,
                                                                                 model,
                                                                                 stop_words,
                                                                                 k=-1)))
    return df


def parallelize_dataframe(df_to_par, func, n_cores=4):
    """Parallelize a function across a DataFrame using multiprocessing.

    Args:
        df_to_par (pandas.DataFrame): The DataFrame to parallelize.
        func: The function to apply to each chunk.
        n_cores (int): The number of cores to use.

    Returns:
        df_res: The combined result df."""
    df_split = np.array_split(df_to_par, n_cores)
    pool = mp.Pool(n_cores)
    df_res = pd.concat(pool.starmap(func, zip(df_split, range(n_cores))))
    pool.close()
    pool.join()
    return df_res


def run_graph_creation(df_reviews, df_prods, nc):
    """Calculate the matrices and clusters for each product using parallel processing.

    Args:
        df_reviews (pandas.DataFrame): DataFrame with all reviews.
        df_prods (pandas.DataFrame): DataFrame with product data.
        nc (int): Number of cores to use.

    Returns:
        df_prods_mc: product dataframe with added matrices and clusters.

    """
    model, stop_words = load_model()
    add_features_df = partial(add_features, df_reviews=df_reviews,
                              model=model, stop_words=stop_words)
    df_prods_mc = parallelize_dataframe(df_prods, func=add_features_df, n_cores=nc)
    return df_prods_mc


if __name__ == '__main__':
    file = input('Please provide the file path (csv) for the input data (reviews): ')
    file2 = input('Please provide the file path (pkl) for the input data (product list): ')
    print('Your logical CPU count is:', psutil.cpu_count(logical=True))
    nc = int(input('Define number of usable cpu (optional, default is 8): ') or 8)
    savename = str(input("Please provide the name of the (existing) output folder to which you " +
                         "want to save the output: " or ""))
    df_reviews_in = pd.read_csv(file, compression="gzip")
    # remove duplicate dows
    df_reviews_2 = df_reviews_in.loc[df_reviews_in.astype(str).drop_duplicates().index]
    df_reviews = df_reviews_2.reset_index()
    df_prods = pd.read_pickle(file2, compression="gzip")
    df_prods_mc = run_graph_creation(df_reviews, df_prods, nc)
    try:
        bn = os.path.basename(file)
        output_path = os.path.join(savename, bn[:bn.index('_reviews.')])
        df_prods_mc.to_pickle(output_path + "_prods_mc.pkl", compression="gzip")
    except Exception:
        print('Failed to save the output of graph_creation')
