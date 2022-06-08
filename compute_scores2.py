from sklearn.cluster import AgglomerativeClustering
import gzip
import json
import pandas as pd
import spacy
#import neuralcoref
import pytextrank
import networkx as nx
from spacy_readability import Readability
import os

nlp = spacy.load('en_core_web_md')
#neuralcoref.add_to_pipe(nlp)

#tr = pytextrank.TextRank()
#nlp.add_pipe(tr.PipelineComponent, name="textrank")
from spacy.language import Language
#@Language.component("textrank")
#def textrank(doc):
#    tr = pytextrank.TextRank()
#    doc = tr.PipelineComponent(doc)
#    return doc

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
  for l in g:
    yield json.loads(l)


def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


#df = get_df('../AMAZON_FASHION_5.json.gz')



def get_readability():
    global df
    df['readability'] = ''
    #for index, review in df.iterrows():
     #   try:
     #       doc = nlp(review['reviewText'])
     #   except:
     #       pass
     #   df.at[index, 'readability'] = d

    df['readability'] = pd.read_csv("AMAZON_FASHION_5_readability.csv")['Automated Readability Index']

#get_readability()
# get review entities

def get_entities():
    entities = pd.read_csv("AMAZON_FASHION_5_entities.csv")
    global df
    df['entities'] = entities['entities']
    #for index, review in df.iterrows():
    #    df.at[index,'entities'] = entities[entities[review_id == review["reviewerID"]+"_"+review["asin"]]]


#get_entities()

def get_relevance():
    global df
    df['coref_clusters'] = ''
    for index, review in df.iterrows():
        try:
            doc = nlp(review['reviewText'])
        except:
            pass
        df.at[index, 'coref_clusters'] = doc._.coref_clusters

#get_relevance()

# compute semantic similarity matrix

def get_simiarity_matrix(reviews):
    global df
    for index, review in df.iterrows():

        entities = get_entities(reviews)

# compute hierarchical clustering

# get entities with importance

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
        except:
            pass
        df.at[index, 'ranks'] = [(x.text, x.rank) for x in doc._.phrases]


files = ["AMAZON_FASHION_5_reviews.csv"]
for file in files:
    print(file)
    #df = pd.read_csv(file,encoding='latin1',keep_default_na=False)
    df = pd.read_csv(file, compression='gzip',  keep_default_na=False)
    #print(df)
    #df['doc_review_text'] = df['doc_review_text'].astype('unicode').values
    df['ranks'] = ''
    df['readability'] = ''
    index = 0
    for doc in nlp.pipe(df['reviewText']):
        try:
            df.at[index, 'ranks'] = [(x.text, x.rank) for x in doc._.phrases]
            df.at[index, 'readability'] = doc._.flesch_kincaid_reading_ease  #forcast #flesch_kincaid_reading_ease #flesch_kinkaid_grade_level # dale_chall #coleman_liau_index _smog automated_readability_index
        except:
            print("Unexpected error:", sys.exc_info()[0])
            df.at[index, 'ranks'] = []
            df.at[index, 'readability'] = 0
        index += 1
    print(df)
    df_prods = pd.DataFrame()
    df_prods['prod'] = df['asin'].unique()
    df_prods.to_pickle("1_prods.pkl", compression="gzip")
    df.to_csv("1_reviews.csv", compression="gzip")

#get_ranks

#df.to_csv("reviews.csv")
#df.to_pickle("df.pkl")
#df_prods = pd.DataFrame()

class Node:
     def __init__(self, topic, rating, weight):
          self.topic = topic
          self.rating = rating
          self.weight = weight



#df_prods.to_pickle("prods.pkl")
#df_prods.to_csv("prods.csv")

#print(df)

