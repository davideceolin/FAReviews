import os

import pandas as pd
import psutil
import pytextrank  # noqa: F401
import spacy
from joblib import Parallel, delayed
from spacy.language import Language
from spacy_readability import Readability


# Load nlp model
@Language.component("readability")
def readability(doc):
    """
    Set up the readability component for the nlp pipeline.
    """
    read = Readability()
    doc = read(doc)
    return doc

# set up the nlp pipeline
nlp = spacy.load('en_core_web_md')
nlp.add_pipe("textrank", last=True)
nlp.add_pipe("readability", last=True)


def apply_ranking(doc, trt):
    """Return a list with only the tokens that have a textrankscore above the treshold."""
    return [(str(x.text), x.rank) for x in doc._.phrases if x.rank >= trt]


def apply_readability(doc):
    "Determine the Flesch-Kincaid readability score for a given text."
    return doc._.flesch_kincaid_reading_ease


def chunker(iterable, total_length, chunksize):
    "Split the data into chunck of size chunksize."
    x = [iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize)]
    return x


def flatten(list_of_lists):
    "Flatten a list of lists."
    return [item for sublist in list_of_lists for item in sublist]


def process_chunk(texts, bs, trt):
    """
    Process a given chuck of data. For each text element (product review) in the chuck,
    the readability score and ranked tokens are calculated.
    """
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=bs):
        preproc_pipe.append([apply_ranking(doc, trt), [apply_readability(doc)]])
    return preproc_pipe


def preprocess_parallel(texts, n_jobs, chunksize, batchsize, trt):
    """
    Setup the parallel processing: the analysis of all product reviews is split into chuncks,
    and the different chunks are processed in parallel.
    """
    executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk, batchsize, trt) for chunk in chunker(texts, len(texts), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)


def compute_scores(file, nc=8, cs=100, bs=20, trt=0.0):
    """
    Compute scores and preprocess review data from a given file.

    This function reads review data from a JSON file, preprocesses the data in parallel
    using the specified number of clusters, chunck size, and batch size, calculates for each review:
    - ranks
    - number of tokens
    - readability score
    Then aggregates per product number (asin), the total number of tokens along all reviews of that product

    Args:
        file (str): Path to the JSON file containing review data.
        nc (int, optional): Number of cores for parallel processing (default is 8).
        cs (int, optional): Chunk size for parallel processing (default is 100).
        bs (int, optional): Batch size for processing (default is 20).
        trt (float, optional): Minimum textrank score threshold for the tokens to be used (default is 0.0).

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - df_reviews: reviews dr with processed review data with for each review the
            calculated readability score and extracted text tokens (on which a trt selecting is applied).
            - df_prods: product number with per product the total number of tokens.
    """

    if not os.path.exists(file):
        raise ValueError('Cannot find file:', file)
    else:
        print('Computing scores for', file, 'with nc:', nc, 'cs:', cs, 'bs:', bs, 'trt:', trt)
    # try to read json file, lines True/False depends on the orient
    # (Indication of expected JSON string format)
    try:
        df = pd.read_json(file, compression='gzip', lines=False)
    except Exception:
        df = pd.read_json(file, compression='gzip', lines=True)
    df['reviewText'] = df['reviewText'].fillna('')
    # remove duplicate dows
    df_reviews_2 = df.loc[df.astype(str).drop_duplicates().index]
    df_reviews = df_reviews_2.reset_index()
    results = preprocess_parallel(df_reviews['reviewText'].astype(str), nc, cs, bs, trt)
    df_reviews['ranks'] = [x[0] for x in results]
    df_reviews['n_tokens'] = [len(x[0]) for x in results]
    df_reviews['readability'] = [x[1][0] for x in results]
    df_prod = pd.DataFrame(df_reviews[['asin', 'n_tokens']].groupby(['asin']).sum()).reset_index()
    df_prod = df_prod.rename(columns={"asin": "prod"})
    # Sort based on ntokens for better distribution over cores
    df_prod = df_prod.sort_values(by=['n_tokens'], ascending=False)
    indices = list(df_prod.index.values)
    ni = [indices[i:][::nc] for i in range(nc)]
    for i in range(1, len(ni), 2):
        ni[i] = ni[i][::-1]
    nj = [item for i in ni for item in i]
    df_prod['newi'] = None
    for indx, value in enumerate(nj):
        df_prod.at[value, 'newi'] = indx
        df_prods = pd.DataFrame(df_prod.sort_values(by=['newi']).reset_index()[['prod',
                                                                                'n_tokens']])
    return df_reviews, df_prods


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
    df, df_prods = compute_scores(file, nc, cs, bs, trt)
    try:
        bn = os.path.basename(file)
        output_path = os.path.join(savename, bn[:bn.index('.')])
        df_prods.to_pickle(output_path + "_prods.pkl", compression="gzip")
        df.to_csv(output_path + "_reviews.csv", compression="gzip")
    except Exception:
        print('Failed to save the output of compute_scores')
