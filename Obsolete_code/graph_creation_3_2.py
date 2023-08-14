import pandas as pd
import time as ttime
import networkx as nx
from networkx.readwrite import json_graph
import ast
import math
from nltk.corpus import stopwords
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
import os
import itertools
import psutil
import multiprocessing as mp
from functools import partial


stop_words = stopwords.words('english')


def color_node(solution):
    if solution == 'undec':
        return 'grey'
    elif solution == 'in':
        return 'green'
    else:
        return 'red'


def draw_graph(G, model, df, savename):
    pos = nx.layout.spring_layout(G)
    node_sizes = [i[1]['weight']*2 for i in G.nodes.data()]
    M = max(G.edges.data(), key=itemgetter(1))[1]  # noqa: F841
    edge_colors = np.argsort([x[2]['weight'] for x in G.edges.data()])  # range(2, M + 2)
    sol = [df.loc[(df['asin'] == x[0].split("_")[0]) & (df['reviewerID'] == x[0].split("_")[1]),
           'solutions'].apply(color_node).values.tolist() for x in G.nodes.data()]
    sol = [y[0] for y in sol]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=sol)  # noqa: F841
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                   arrowsize=10, edge_color=edge_colors,
                                   edge_cmap=plt.cm.Blues, width=2)
    ax = plt.gca()
    ax.set_axis_off()
    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    prod_id = list(G.nodes())[0].split("_")[0]
    plt.savefig(os.path.join(savename, prod_id+'.png'), bbox_inches='tight')
    nx.draw_networkx_labels(G, pos, G.nodes, font_size=8)
    p = plt.savefig(os.path.join(savename, prod_id+'_labels.png'), bbox_inches='tight')
    plt.clf()
    plt.close(p)


def solve_argumentation_graph_json(G):
    tt = ttime.time()
    data = json_graph.node_link_data(G)
    r = requests.post('http://localhost:3333/argue', json=data)
    return r.json(), ttime.time()-tt


def run_solver(prod, mc, df, savename, savefigs):
    t1 = ttime.time()
    matrix = mc[0]
    cluster_prod = mc[1]
    # lists that will collect all tokens & info of one product
    reviewID = []
    score = []
    tokens = []
    importance = []
    readability = []
    clusters = []
    G = nx.DiGraph()
    # Get all info of prod and convert data into lists
    prod_reviews = pd.DataFrame(df[df['asin'] == prod])
    ranks_prod = [ast.literal_eval(str(rank)) for rank in prod_reviews['ranks'].to_numpy()]
    readability_prod = prod_reviews['readability'].to_list()
    score_prod = prod_reviews['overall'].to_list()
    reviewID_prod = (prod_reviews['asin'] + '_' + prod_reviews['reviewerID'] + '_' +
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
        reviewID.extend([reviewID_prod[r]]*len(toks))
        score.extend([score_prod[r]]*len(toks))
        tokens.extend(toks)
        importance.extend(imps)
        readability.extend([readability_prod[r]]*len(toks))
        clusters.extend(clusters_toks)
    for i, j in itertools.product(list(range(len(tokens))), repeat=2):
        if (reviewID[i] != reviewID[j] and
            score[i] != score[j] and
                clusters[i] == clusters[j]):
            if not G.has_node(reviewID[i]):
                G.add_node(reviewID[i], weight=readability[i])
            if not G.has_node(reviewID[j]):
                G.add_node(reviewID[j], weight=readability[j])
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
                if G.has_edge(reviewID[i], reviewID[j]):
                    weight = int(G.get_edge_data(reviewID[i], reviewID[j],
                                 'weight')['weight'])
                    G.remove_edge(reviewID[i], reviewID[j])
                    G.add_edge(reviewID[i], reviewID[j],
                               weight=weight + (w1 - w2) * sim_t1_t2)
                elif G.has_edge(reviewID[j], reviewID[i]):
                    weight = int(G.get_edge_data(reviewID[j], reviewID[i],
                                 'weight')['weight']) * -1
                    weight = weight + (w1 - w2) * sim_t1_t2
                    if weight > 0:
                        G.remove_edge(reviewID[j], reviewID[i])
                        G.add_edge(reviewID[i], reviewID[j], weight=weight)
                    else:
                        if G.has_edge(reviewID[i], reviewID[j]):
                            G.remove_edge(reviewID[i], reviewID[j])
                        G.add_edge(reviewID[j], reviewID[i],
                                   weight=weight * -1)
                else:
                    G.add_edge(reviewID[i], reviewID[j],
                               weight=weight + (w1 - w2) * sim_t1_t2)
            elif w2 > w1:
                if G.has_edge(reviewID[j], reviewID[i]):
                    weight = int(G.get_edge_data(reviewID[j], reviewID[i],
                                 'weight')['weight'])
                    G.remove_edge(reviewID[j], reviewID[i])
                    G.add_edge(reviewID[j], reviewID[i],
                               weight=weight + (w2 - w1) * sim_t1_t2)
                elif G.has_edge(reviewID[i], reviewID[j]):
                    weight = int(G.get_edge_data(reviewID[i], reviewID[j],
                                 'weight')['weight']) * -1
                    weight = weight + (w2 - w1) * sim_t1_t2
                    if weight > 0:
                        G.remove_edge(reviewID[i], reviewID[j])
                        G.add_edge(reviewID[j], reviewID[i], weight=weight)
                    else:
                        if G.has_edge(reviewID[j], reviewID[i]):
                            G.remove_edge(reviewID[j], reviewID[i])
                        G.add_edge(reviewID[i], reviewID[j],
                                   weight=weight * -1)
                else:
                    G.add_edge(reviewID[j], reviewID[i],
                               weight=weight + (w2 - w1) * sim_t1_t2)
    r, tgs = solve_argumentation_graph_json(G)
    if 'models' in r and len(r['models']) > 0:
        models = r['models']
        weights = [0 for x in r['models']]
        for i in range(len(models)):
            for node in models[i]['nodes']:
                if node['state'] == 'in':
                    prodID, reviewerID, time = node['id'].upper().split("_")
                    weights[i] = weights[i] + node['weight']
        model = models[weights.index(max(weights))]
        for node in model['nodes']:
            reviewerID = node['id'].upper().split("_")[1]
            prod_reviews.loc[prod_reviews['reviewerID'] == reviewerID, 'solutions'] = node['state']
        if savefigs:
            draw_graph(G, model, prod_reviews, savename)
    t2 = ttime.time()-t1
    print(prod, len(tokens), tgs, t2-tgs, t2, mp.current_process())
    return prod_reviews


def parallelize_run_solver(p_run_solver, prods, mc_m, mc_c, n_cores):
    pool = mp.Pool(n_cores)
    mc = zip(mc_m, mc_c)
    df_results = pd.concat(pool.starmap(p_run_solver, zip(prods, mc)))
    pool.close()
    pool.join()
    return df_results


def run_graph_solver(df_prods, df_reviews, nc, savename, savefigs):
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
    df = pd.read_csv(file, compression='gzip')
    df_results = run_graph_solver(df_prods, nc, df, savename, savefigs)
    try:
        bn = os.path.basename(file)
        output_path = os.path.join(savename, bn[:bn.index('_reviews.')])
        df_results.to_csv(output_path + "_reviews_results.csv", compression='gzip')
    except Exception:
        print('Failed to save the output of graph_creation')
    print('Duration:', ttime.time()-t1)
