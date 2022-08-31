import os
import time

import pandas as pd
import pytextrank  # noqa: F401
import spacy
from joblib import Parallel, delayed
from spacy.language import Language
from spacy_readability import Readability


# Load nlp model
@Language.component("readability")
def readability(doc):
    read = Readability()
    doc = read(doc)
    return doc


nlp = spacy.load('en_core_web_md')
nlp.add_pipe("textrank", last=True)
nlp.add_pipe("readability", last=True)


def apply_ranking(doc, trt):
    return [(str(x.text), x.rank) for x in doc._.phrases if x.rank >= trt]


def apply_readability(doc):
    return doc._.flesch_kincaid_reading_ease


def chunker(iterable, total_length, chunksize):
    x = [iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize)]
    return x


def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]


def process_chunk(texts, bs, trt):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=bs):
        preproc_pipe.append([apply_ranking(doc, trt), [apply_readability(doc)]])
    return preproc_pipe


def preprocess_parallel(texts, n_jobs, chunksize, batchsize, trt):
    executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk, batchsize, trt) for chunk in chunker(texts, len(texts), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)


def compute_scores(file, nc=8, cs=100, bs=20, trt=0.0, savename=""):
    start_time = time.time()
    if not os.path.exists(file):
        raise ValueError('Cannot find file:', file)
    else:
        print('Computing scores for', file, 'with nc:', nc, 'cs:', cs, 'bs:', bs, 'trt:', trt)
    df = pd.read_json(file, compression='gzip', lines=True)
    df['reviewText'] = df['reviewText'].fillna('')
    results = preprocess_parallel(df['reviewText'].astype(str), nc, cs, bs, trt)
    df['ranks'] = [x[0] for x in results]
    df['readability'] = [x[1][0] for x in results]
    df_prods = pd.DataFrame()
    df_prods['prod'] = df['asin'].unique()
    try:
        bn = os.path.basename(file)
        output_path = os.path.join(savename, bn[:bn.index('.')])
        df_prods.to_pickle(output_path + "_prods.pkl", compression="gzip")
        df.to_csv(output_path + "_reviews.csv", compression="gzip")
    except Exception:
        print('Failed to save the output of compute_scores')
    print("--- %s seconds ---" % (time.time() - start_time))
    return df_prods, df


if __name__ == "__main__":
    file = str(input("Please provide path to data file: "))
    nc = int(input("Define number of jobs: "))
    cs = int(input("Define number of chunks: "))
    bs = int(input("Define batch size: "))
    trt = float(input("Define threshold for textrank token collection: "))
    compute_scores(file, nc, cs, bs, trt)
