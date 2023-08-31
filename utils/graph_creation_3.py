import ast
import itertools
import math
import multiprocessing as mp
import os
import time as ttime
from functools import partial
from operator import itemgetter

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import requests
from networkx.readwrite import json_graph
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


def color_node(solution):
    """Define the color of the node based on the given solution."""
    if solution == 'undec':
        return 'grey'
    elif solution == 'in':
        return 'green'
    else:
        return 'red'


def draw_graph(g, model, df, savename):
    pos = nx.layout.spring_layout(g)
    node_sizes = [i[1]['weight']*2 for i in g.nodes.data()]
    m = max(g.edges.data(), key=itemgetter(1))[1]  # noqa: F841
    edge_colors = np.argsort([x[2]['weight'] for x in g.edges.data()])  # range(2, m + 2)
    sol = [df.loc[(df['asin'] == x[0].split("_")[0]) & (df['reviewerID'] == x[0].split("_")[1]),
           'solutions'].apply(color_node).values.tolist() for x in g.nodes.data()]
    sol = [y[0] for y in sol]
    nodes = nx.draw_networkx_nodes(g, pos, node_size=node_sizes, node_color=sol)  # noqa: F841
    edges = nx.draw_networkx_edges(g, pos, node_size=node_sizes, arrowstyle='->',
                                   arrowsize=10, edge_color=edge_colors,
                                   edge_cmap=plt.cm.Blues, width=2)
    ax = plt.gca()
    ax.set_axis_off()
    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    prod_id = list(g.nodes())[0].split("_")[0]
    plt.savefig(os.path.join(savename, prod_id+'.png'), bbox_inches='tight')
    nx.draw_networkx_labels(g, pos, g.nodes, font_size=8)
    p = plt.savefig(os.path.join(savename, prod_id+'_labels.png'), bbox_inches='tight')
    plt.clf()
    plt.close(p)


def solve_argumentation_graph_json(g):
    """Solve the argumentation graph using the prolog solver."""
    data = json_graph.node_link_data(g)
    r = requests.post('http://localhost:3333/argue', json=data)
    return r.json()


def run_solver(prod, mc, df, savename, savefigs):
    """
    Run the solver on the given product.

    Args:
        prod (str): product asin.
        mc (tuple): Tuple containing matrix and clusters for each product.
        df (pandas.DataFrame): df with all reviews.
        savename (str): Path to save the output.
        savefigs (bool): flag to create save graphs.

    Returns:
        prod_reviews: df with per product the graph solution
    """
    matrix = mc[0]
    cluster_prod = mc[1]
    # lists that will collect all tokens & info of one product
    review_id = []
    score = []
    tokens = []
    importance = []
    readability = []
    clusters = []
    g = nx.DiGraph()
    # Get all info of prod and convert data into lists
    prod_reviews = pd.DataFrame(df[df['asin'] == prod])
    ranks_prod = [ast.literal_eval(str(rank)) for rank in prod_reviews['ranks'].to_numpy()]
    readability_prod = prod_reviews['readability'].to_list()
    score_prod = prod_reviews['overall'].to_list()
    review_id_prod = (prod_reviews['asin'] + '_' + prod_reviews['reviewerID'] + '_' +
                      prod_reviews['unixReviewTime'].map(str)).to_numpy(dtype=str)
    # Determine per review the individual tokens and accompanying info, and append those
    # to the corresponding lists that are collecting all tokens & info of one product
    for r, ranks in enumerate(ranks_prod):
        tokens_imp = [[" ".join([x for x in token[0].lower().split() if x not in stop_words]),
                      token[1]] for token in ranks]
        tokens_imp = [i for i in tokens_imp if i[0] not in ("", " ")]
        toks = [i[0] for i in tokens_imp]
        imps = [i[1] for i in tokens_imp]
        clusters_toks = [cluster_prod[matrix.columns.get_loc(token)]
                         if token in list(matrix.columns) else 0 for token in toks]
        review_id.extend([review_id_prod[r]]*len(toks))
        score.extend([score_prod[r]]*len(toks))
        tokens.extend(toks)
        importance.extend(imps)
        readability.extend([readability_prod[r]]*len(toks))
        clusters.extend(clusters_toks)
    for i, j in itertools.product(list(range(len(tokens))), repeat=2):
        if (review_id[i] != review_id[j] and
            score[i] != score[j] and
                clusters[i] == clusters[j]):
            if not g.has_node(review_id[i]):
                g.add_node(review_id[i], weight=readability[i])
            if not g.has_node(review_id[j]):
                g.add_node(review_id[j], weight=readability[j])
            try:
                w1 = readability[i] * importance[i]
            except Exception:
                w1 = 0
            try:
                w2 = readability[j] * importance[j]
            except Exception:
                w2 = 0
            w1 = 0 if math.isnan(w1) else w1
            w2 = 0 if math.isnan(w2) else w2
            try:
                sim_t1_t2 = matrix.at[tokens[i], tokens[j]]
            except Exception:
                sim_t1_t2 = 0
            weight = 0
            if w1 > w2:
                if g.has_edge(review_id[i], review_id[j]):
                    weight = int(g.get_edge_data(review_id[i], review_id[j],
                                 'weight')['weight'])
                    g.remove_edge(review_id[i], review_id[j])
                    g.add_edge(review_id[i], review_id[j],
                               weight=weight + (w1 - w2) * sim_t1_t2)
                elif g.has_edge(review_id[j], review_id[i]):
                    weight = int(g.get_edge_data(review_id[j], review_id[i],
                                 'weight')['weight']) * -1
                    weight = weight + (w1 - w2) * sim_t1_t2
                    if weight > 0:
                        g.remove_edge(review_id[j], review_id[i])
                        g.add_edge(review_id[i], review_id[j], weight=weight)
                    else:
                        if g.has_edge(review_id[i], review_id[j]):
                            g.remove_edge(review_id[i], review_id[j])
                        g.add_edge(review_id[j], review_id[i],
                                   weight=weight * -1)
                else:
                    g.add_edge(review_id[i], review_id[j],
                               weight=weight + (w1 - w2) * sim_t1_t2)
            elif w2 > w1:
                if g.has_edge(review_id[j], review_id[i]):
                    weight = int(g.get_edge_data(review_id[j], review_id[i],
                                 'weight')['weight'])
                    g.remove_edge(review_id[j], review_id[i])
                    g.add_edge(review_id[j], review_id[i],
                               weight=weight + (w2 - w1) * sim_t1_t2)
                elif g.has_edge(review_id[i], review_id[j]):
                    weight = int(g.get_edge_data(review_id[i], review_id[j],
                                 'weight')['weight']) * -1
                    weight = weight + (w2 - w1) * sim_t1_t2
                    if weight > 0:
                        g.remove_edge(review_id[i], review_id[j])
                        g.add_edge(review_id[j], review_id[i], weight=weight)
                    else:
                        if g.has_edge(review_id[j], review_id[i]):
                            g.remove_edge(review_id[j], review_id[i])
                        g.add_edge(review_id[i], review_id[j],
                                   weight=weight * -1)
                else:
                    g.add_edge(review_id[j], review_id[i],
                               weight=weight + (w2 - w1) * sim_t1_t2)
    r = solve_argumentation_graph_json(g)
    if 'models' in r and len(r['models']) > 0:
        models = r['models']
        weights = [0 for x in r['models']]
        for i in range(len(models)):
            for node in models[i]['nodes']:
                if node['state'] == 'in':
                    prod_id, reviewer_id, time = node['id'].upper().split("_")
                    weights[i] = weights[i] + node['weight']
        model = models[weights.index(max(weights))]
        for node in model['nodes']:
            reviewer_id = node['id'].upper().split("_")[1]
            prod_reviews.loc[prod_reviews['reviewerID'] == reviewer_id, 'solutions'] = node['state']
        if savefigs:
            draw_graph(g, model, prod_reviews, savename)
    return prod_reviews


def parallelize_run_solver(p_run_solver, prods, mc_m, mc_c, n_cores):
    """Parallelize the solver function across products using multiprocessing.

    Args:
        p_run_solver (function): Function to parallelize.
        prods: List of products.
        mc_m: List of matrices.
        mc_c: List of clusters.
        n_cores (int): Number of cores to use.

    Returns:
        df_results: df with graph resoluts.
    """
    pool = mp.Pool(n_cores)
    mc = zip(mc_m, mc_c)
    df_results = pd.concat(pool.starmap(p_run_solver, zip(prods, mc)))
    pool.close()
    pool.join()
    return df_results


def run_graph_solver(df_prods, df_reviews, nc, savename, savefigs):
    """
    Solve the argumentation graphs.

    Args:
        df_prods (pandas.DataFrame): df with products.
        df_reviews (pandas.DataFrame): df with all reviews.
        nc (int): Number of cores to use.
        savename (str): Path to save output.
        savefigs (bool): flag to create save graphs.

    Returns:
        df_results: df with solved graohs per product, sorted by index.
    """
    df_reviews['solutions'] = "undec"
    p_run_solver = partial(run_solver, df=df_reviews, savename=savename, savefigs=savefigs)
    prods = df_prods['prod'].to_list()
    mc_matrix = df_prods['matrix'].to_list()
    mc_cluster = df_prods['clusters'].to_list()
    df_results = parallelize_run_solver(p_run_solver, prods, mc_matrix, mc_cluster, nc)
    return df_results.sort_index()


if __name__ == "__main__":
    t1 = ttime.time()
    file = input('Please provide the file path (csv) for the input data (reviews): ')
    file2 = input('Please provide the file path (pkl) for the input data (mc product list): ')
    print('Your logical CPU count is:', psutil.cpu_count(logical=True))
    nc = int(input('Define number of usable cpu (optional, default is 8): ') or 8)
    savename = str(input("Please provide the name of the (existing) output folder to which you " +
                         "want to save the output: " or ""))
    savefigs = bool(input("Do you want to save the graphs to png per product? "
                          "(Tru/False, default is False)" or False))
    df_prods = pd.read_pickle(file2, compression='gzip')
    df_reviews_in = pd.read_csv(file, compression='gzip')
    # remove duplicate dows
    df_reviews_2 = df_reviews_in.loc[df_reviews_in.astype(str).drop_duplicates().index]
    df_reviews = df_reviews_2.reset_index()
    df_results = run_graph_solver(df_prods, df_reviews, nc, savename, savefigs)
    try:
        bn = os.path.basename(file)
        output_path = os.path.join(savename, bn[:bn.index('_reviews.')])
        df_results.to_csv(output_path + "_reviews_results.csv", compression='gzip')
    except Exception:
        print('Failed to save the output of graph_creation')
    print('Duration:', ttime.time()-t1)
