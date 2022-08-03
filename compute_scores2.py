import gzip
import json
import pandas as pd
import spacy
from spacy_readability import Readability
from spacy.language import Language
import pytextrank  # noqa: F401
import sys

nlp = spacy.load('en_core_web_md')


@Language.component("readability")
def readability(doc):
    read = Readability()
    doc = read(doc)
    return doc


nlp.add_pipe("textrank", last=True)
nlp.add_pipe("readability", last=True)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def parse(path):
    g = gzip.open(path, 'rb')
    for ll in g:
        yield json.loads(ll)


def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def get_readability():
    global df
    df['readability'] = ''
    df['readability'] = pd.read_csv(
                                    "AMAZON_FASHION_5_readability.csv"
                                    )['Automated Readability Index']


def get_entities():
    entities = pd.read_csv("AMAZON_FASHION_5_entities.csv")
    global df
    df['entities'] = entities['entities']


def get_relevance():
    global df
    df['coref_clusters'] = ''
    for index, review in df.iterrows():
        try:
            doc = nlp(review['reviewText'])
        except Exception:
            pass
        df.at[index, 'coref_clusters'] = doc._.coref_clusters


def get_simiarity_matrix(reviews):
    global df
    for index, review in df.iterrows():
        entities = get_entities(reviews)
    return entities


def spl(x):
    y = x.split("_")
    if len(y) > 1:
        return x.split("_")[2]
    else:
        return 0


def get_ranks():
    global df
    df['ranks'] = ''
    for index, review in df.iterrows():
        try:
            doc = nlp(review['reviewText'])
        except Exception:
            pass
        df.at[index, 'ranks'] = [(x.text, x.rank) for x in doc._.phrases]


files = ["AMAZON_FASHION_5_reviews.csv"]
for file in files:
    print(file)
    df = pd.read_csv(file, compression='gzip',  keep_default_na=False)
    df['ranks'] = ''
    df['readability'] = ''
    index = 0
    for doc in nlp.pipe(df['reviewText']):
        try:
            df.at[index, 'ranks'] = [(x.text, x.rank) for x in doc._.phrases]
            df.at[index, 'readability'] = doc._.flesch_kincaid_reading_ease
        except Exception:
            print("Unexpected error:", sys.exc_info()[0])
            df.at[index, 'ranks'] = []
            df.at[index, 'readability'] = 0
        index += 1
    print(df)
    df_prods = pd.DataFrame()
    df_prods['prod'] = df['asin'].unique()
    df_prods.to_pickle("1_prods.pkl", compression="gzip")
    df.to_csv("1_reviews.csv", compression="gzip")


class Node:
    def __init__(self, topic, rating, weight):
        self.topic = topic
        self.rating = rating
        self.weight = weight
