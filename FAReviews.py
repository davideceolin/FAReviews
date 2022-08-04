import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import os
from compute_scores2 import compute_scores


def main():
    desc = """
    FARreviews performs feature extraction and creates the matrix with distance metrics for a
    given data set of product reviews.

    Run with -f to provide the input data file. -nr to define the number of cores,
    -cs for chuncksize, -bs for batch size, and
    -trt for the textrank threshold. """

    # start time
    start_time = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    print('Start time:', start_time)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f', '--file',
                        help='Specific json.gz data file with product reviews to be processed.')
    parser.add_argument('-nc', '--num_cores', type=int, default=8,
                        help='Number of cores to use for the various processes that are ran in '
                        'parallel.')
    parser.add_argument('-cs', '--chunck_size', type=int, default=100,
                        help='Chunk size used in compute scores')
    parser.add_argument('-bs', '--batch_size', type=int, default=2000,
                        help='Batch size used in compute scores')
    parser.add_argument('-trt', '--textrank_treshold',
                        help="Minimum textrank score threshold for the tokens to be used. Tokens "
                        "with a textrank score below the threshold are not used.")
    args = parser.parse_args()

    if not (args.file):
        raise Exception('No input provided. Run with -h for help on arguments to be provided.')
    if not os.path.exists('Output'):
        os.mkdir("Output")
    if args.file is not None:
        run_FAReviews(args.file, args.num_cores, args.chunck_size, args.batch_size)

    end_time = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    print('End time:', end_time)


def run_FAReviews(infile, nc, cs, bs):
    # compute scores
    prods, reviews = compute_scores(infile, nc, cs, bs)
    print('Finished computing scores')
    # calculate matrix and clusters
    print('Finished calculating matrix and clusters')
    # create and solve argumentation graphs
    print('Finished creating and solving argumentation graphs')


if __name__ == "__main__":
    main()
