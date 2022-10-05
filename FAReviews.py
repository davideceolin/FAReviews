import argparse
import os
import time as ttime
from argparse import RawTextHelpFormatter
from datetime import datetime
import pandas as pd

from compute_scores2 import compute_scores


def main():
    desc = """
    FARreviews performs score computation, calculates the matrices and clusters for a
    given data set of product reviews, and finally solves the argumentation graph per product.
    This last step requires the prolog server to be running.

    Run with -f to provide the input data file. -nr to define the number of cores,
    -cs for chuncksize, -bs for batch size,
    -trt for the textrank threshold, and -sn for the name of the output folder. """

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
    parser.add_argument('-sn', '---savename', type=str, default="Output", help="Name of the output"
                        " folder (within the current folder) where you want to save the output. "
                        "If it does not yet exist, it will be created.")
    parser.add_argument('-si', '--save_intermediate', type=bool, default=False, help="If true, also"
                        " save the outputof compute scores and run_graph")
    parser.add_argument('-sf', '--savefigs', type=bool, default=False, help="Option to save the "
                        "constructed graphs to png per product")
    args = parser.parse_args()

    if not (args.file):
        raise Exception('No input provided. Run with -h for help on arguments to be provided.')
    if not os.path.exists(args.savename):
        os.mkdir(args.savename)
    return (args.file, args.num_cores, args.chunk_size, args.batch_size, args.textrank_threshold,
            args.savename, args.save_intermediate, args.savefigs)


def run_compute_scores(infile, nc, cs, bs, trt, si, savename):
    print('Starting computing scores')
    prods, reviews = compute_scores(infile, nc, cs, bs, trt, savename)
    if si:
        df_prods = pd.DataFrame({'prod': prods})
        try:
            bn = os.path.basename(file)
            output_path = os.path.join(savename, bn[:bn.index('.')])
            df_prods.to_pickle(output_path + "_prods.pkl", compression="gzip")
            reviews.to_csv(output_path + "_reviews.csv", compression="gzip")
        except Exception:
            print('Failed to save the output of compute_scores')
    print('Finished computing scores')
    return reviews, prods


def run_graph(reviews, prods, num_cores, si, savename):
    from graph_creation3 import run_graph_creation
    print('Start calculating matrices and clusters')
    tt = ttime.time()
    df_prods_mc = run_graph_creation(reviews, prods, num_cores)
    if si:
        try:
            bn = os.path.basename(file)
            output_path = os.path.join(savename, bn[:bn.index('.')])
            df_prods_mc.to_pickle(output_path + "_prods_mc.pkl", compression="gzip")
        except Exception:
            print('Failed to save the output of graph_creation')
    print('Finished calculating matrix and clusters', ttime.time()-tt)
    return df_prods_mc


def run_prolog_solver(df_prods_mc, reviews, nc, savename, savefigs):
    import graph_creation_3
    tt = ttime.time()
    print('Start solving graphs')
    df_results = graph_creation_3.run_graph_solver(df_prods_mc, reviews, nc,
                                                   savename, savefigs)
    try:
        bn = os.path.basename(file)
        output_path = os.path.join(savename, bn[:bn.index('.')])
        df_results.to_csv(output_path + "_reviews_results.csv", compression='gzip')
    except Exception:
        print('Failed to save the output of graph_creation')
    print('Finished solving', ttime.time()-tt)
    return


if __name__ == "__main__":
    # start time
    st = ttime.time()
    start_time = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    print('Start time:', start_time)
    file, num_cores, chunk_size, batch_size, trt, savename, save_intermediate, savefigs = main()
    if file is not None:
        reviews, prods = run_compute_scores(file, num_cores, chunk_size, batch_size, trt,
                                            save_intermediate, savename)
        df_prods_mc = run_graph(reviews, prods, num_cores, save_intermediate, savename)
        run_prolog_solver(df_prods_mc, reviews, num_cores, savename, savefigs)
    end_time = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    print('End time FAReviews:', end_time, 'duration:', ttime.time()-st)
