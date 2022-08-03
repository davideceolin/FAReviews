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
