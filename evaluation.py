from sklearn.cluster import AgglomerativeClustering
import gzip
import json
import pandas as pd
import networkx as nx
from spacy_readability import Readability
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
# cluster.fit_predict(X)


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


df = get_df('/home/davide/Downloads/AMAZON_FASHION_5.json.gz')

sns.distplot(df['vote'].value_counts())

plt.show()