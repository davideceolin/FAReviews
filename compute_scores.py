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
#    nlp.add_pipe("textrank")
#    doc = nlp(text)
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


def get_ranks():
    global df
    df['ranks'] = ''
    for index, review in df.iterrows():
        try:
            doc = nlp(review['reviewText'])
        except:
            pass
        df.at[index, 'ranks'] = [(x.text, x.rank) for x in doc._.phrases]


files = [dirpath+"/"+file for dirpath, dirnames, filename in os.walk("..") for file in filename if file.endswith("_5.json.gz")]
#files = ["workers_data_merged.csv"]
for file in files:
    print(file)
    df = get_df(file)
    #read = Readability(nlp)
    #nlp.add_pipe(read, last=True)
    df['reviewText'] = df['reviewText'].astype('unicode').values
    df['ranks'] = ''
    df['readability'] = ''
    index = 0
    for doc in nlp.pipe(df['reviewText']):
        try:
            df.at[index, 'ranks'] = [(x.text, x.rank) for x in doc._.phrases]
            df.at[index, 'readability'] = doc._.automated_readability_index  #forcast #flesch_kincaid_reading_ease #flesch_kinkaid_grade_level # dale_chall #coleman_liau_index _smog automated_readability_index
        except:
            print("Unexpected error:", sys.exc_info()[0])
            df.at[index, 'ranks'] = []
            df.at[index, 'readability'] = 0
        index += 1

    #for index, review in df.iterrows():
    #    print(index)
    ##    try:
    #        doc = nlp(review['reviewText'])
    #        df.at[index, 'ranks'] = [(x.text, x.rank) for x in doc._.phrases]
    #        df.at[index, 'readability'] = doc._.flesch_kincaid_reading_ease
    #    except:
    #        df.at[index, 'ranks'] = []
    #        df.at[index, 'readability'] = 0
    print(df)
    df_prods = pd.DataFrame()
    df_prods['prod'] = df['asin'].unique()
    #df_prods['nodes'] = [[] for x in df['asin'].unique()]
    df_prods.to_pickle(file.split("/")[1].split(".")[0]+"_prods.pkl", compression="gzip")
    df.to_csv(file.split("/")[1].split(".")[0]+"_reviews.csv",compression="gzip")
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


#df_prods['prod'] = df['asin'].unique()
#df_prods['nodes'] = [[] for x in df['asin'].unique()]

#nodes = []

#for index_prod, prod in df_prods.iterrows():
#    for index, review in df.loc[df['asin'] == prod['prod']].iterrows():
#        try:
#            doc = nlp(review['reviewText'])
#            for token in review['ranks']:
#                weight = token[1] * doc._.dale_chall
#                node = Node(token[0], 1 if review['overall'] in [1, 2] else (-1 if review['overall'] in [4, 5] else 0),
#                        weight)
#                df_prods.at[index_prod, 'nodes'].append(node)
#                nodes.append(node)
#        except:
#            pass

#df_prods.to_pickle("prods.pkl")
#df_prods.to_csv("prods.csv")

#print(df)

