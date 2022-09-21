import os
import time

import pandas as pd
import pytextrank  # noqa: F401
import spacy
from joblib import Parallel, delayed
from spacy.language import Language
from spacy_readability import Readability
import psutil


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
    df['n_tokens'] = [len(x[0]) for x in results]
    df['readability'] = [x[1][0] for x in results]
    # create df_prods with prods sorted for load balancing over nc
    # based on total number of tokens per product
    df_prod = pd.DataFrame(df[['asin', 'n_tokens']].groupby(['asin']).sum()).reset_index()
    df_prod = df_prod.rename(columns={"asin": "prod"})
    indices = list(df_prod.sort_values(by=['n_tokens'], ascending=False).index.values)
    ni = [indices[i:][::nc] for i in range(nc)]
    nj = [item for i in ni for item in i]
    df_prod['newi'] = None
    for indx, value in enumerate(nj):
        df_prod.at[value, 'newi'] = indx
    df_prods = pd.DataFrame(df_prod.sort_values(by=['newi']).reset_index()['prod'])
    print("--- %s seconds ---" % (time.time() - start_time))
    return df_prods, df


if __name__ == "__main__":
    file = str(input("Please provide path to data file (json(.gz)): "))
    print('Your logical CPU count is:', psutil.cpu_count(logical=True))
    nc = int(input("Define number of jobs (optional, default is 8): ") or 8)
    cs = int(input("Define number of chunks (optional, default is 100): ") or 100)
    bs = int(input("Define batch size (optional, default is 20): ") or 20)
    trt = float(input("Define threshold for textrank token collection "
                "(optional, default is 0.0): ") or 0.0)
    savename = str(input("Please provide the name of the (existing) output folder to which you " +
                         "want to save the output: " or ""))
    df_prods, df = compute_scores(file, nc, cs, bs, trt)
    try:
        bn = os.path.basename(file)
        output_path = os.path.join(savename, bn[:bn.index('.')])
        df_prods.to_pickle(output_path + "_prods.pkl", compression="gzip")
        df.to_csv(output_path + "_reviews.csv", compression="gzip")
    except Exception:
        print('Failed to save the output of compute_scores')
