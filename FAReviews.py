import argparse
import os
import time
from argparse import RawTextHelpFormatter
from datetime import datetime

from compute_scores2 import compute_scores


def main():
    desc = """
    FARreviews performs feature extraction and creates the matrix with distance metrics for a
    given data set of product reviews.

    Run with -f to provide the input data file. -nr to define the number of cores,
    -cs for chuncksize, -bs for batch size, and
    -trt for the textrank threshold. """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f', '--file',
                        help='Specific json.gz data file with product reviews to be processed.')
    parser.add_argument('-nc', '--num_cores', type=int, default=8,
                        help='Number of cores to use for the various processes that are ran in '
                        'parallel.')
    parser.add_argument('-cs', '--chunk_size', type=int, default=100,
                        help='Chunk size used in compute scores')
    parser.add_argument('-bs', '--batch_size', type=int, default=20,
                        help='Batch size used in compute scores')
    parser.add_argument('-trt', '--textrank_threshold', type=float, default=0.0,
                        help="Minimum textrank score threshold for the tokens to be used. Tokens "
                        "with a textrank score below the threshold are not used.")
    args = parser.parse_args()

    if not (args.file):
        raise Exception('No input provided. Run with -h for help on arguments to be provided.')
    if not os.path.exists('Output'):
        os.mkdir("Output")
    return args.file, args.num_cores, args.chunk_size, args.batch_size, args.textrank_threshold


def run_compute_scores(infile, nc, cs, bs, trt):
    # compute scores
    print('starting computing scores')
    prods, reviews = compute_scores(infile, nc, cs, bs, trt)
    print('Finished computing scores')
    # create and solve argumentation graphs
    # print('Finished creating and solving argumentation graphs')
    return reviews, prods


def run_graph(reviews, prods, num_cores):
    from graph_creation3 import run_graph_creation
    # calculate matrix and clusters
    tt = time.time()
    df_prods_mc = run_graph_creation(reviews, prods, num_cores)
    try:
        bn = os.path.basename(file)
        output_path = os.path.join("Output", bn[:bn.index('.')])
        df_prods_mc.to_pickle(output_path + "_prods_mc.pkl", compression="gzip")
    except Exception:
        print('Failed to save the output of graph_creation')
    print('Finished calculating matrix and clusters', time.time()-tt)
    return df_prods_mc


def run_prolog_solver(df_prods_mc, reviews):
    import graph_creation_3
    tt = time.time()
    graph_creation_3.run_solver(df_prods_mc, reviews)
    print('Finished solving', time.time()-tt)
    return


if __name__ == "__main__":
    # start time
    st = time.time()
    start_time = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    print('Start time:', start_time)
    file, num_cores, chunk_size, batch_size, trt = main()
    if file is not None:
        reviews, prods = run_compute_scores(file, num_cores, chunk_size, batch_size, trt)
        df_prods_mc = run_graph(reviews, prods, num_cores)
        run_prolog_solver(df_prods_mc, reviews)
    end_time = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    print('End time FAReviews:', end_time, 'duration:', time.time()-st)
