# from compute_scores import Node
# import pickle5 as pickle
# import pickle
import pandas as pd
import time as ttime
import networkx as nx
from networkx.readwrite import json_graph
import ast
# from pengines.Builder import PengineBuilder
# from pengines.Pengine import Pengine
# from prologterms import TermGenerator, PrologRenderer, Program, Var, Rule, Term, Const
# from prologterms import Const
import math
from nltk.corpus import stopwords
# import sys
# import json
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
import os


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
    # node_sizes = [3 + 10 * i for i in range(len(G))]
    M = max(G.edges.data(), key=itemgetter(1))[1]  # noqa: F841

    # M = G.number_of_edges()
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
    data = json_graph.node_link_data(G)
    r = requests.post('http://localhost:3333/argue', json=data)
    return r.json()


def run_solver(df_prods, df, savename):
    stop_words = stopwords.words('english')

    df['solutions'] = "undec"
    df_results = pd.DataFrame(columns=["prodID", "solutions"])

    for index, row in df_prods.iterrows():
        df_reviews = pd.DataFrame(columns=['reviewID', 'score', 'token', 'importance',
                                           'readability', 'cluster'])
        G = nx.DiGraph()
        for id, review in df.loc[df['asin'] == row['prod']].iterrows():
            ranks = ast.literal_eval(str(review['ranks']))
            readability = review['readability']
            score = review['overall']
            reviewID = (review['asin'] + "_" + review['reviewerID'] + "_" +
                        str(review['unixReviewTime']))
            for token in ranks:
                tok = " ".join([x for x in token[0].lower().split() if x not in stop_words])
                if tok == "" or tok == " ":
                    continue
                try:
                    cluster = row[('matrix', 'clusters')][1][row[('matrix', 'clusters')
                                                                 ][0].columns.get_loc(tok)]
                except Exception:
                    cluster = 0
                to_append = pd.DataFrame(
                    {'reviewID': reviewID, 'score': score, 'token': tok, 'importance': token[1],
                        'readability': readability, 'cluster': cluster}, index=[0])
                df_reviews = pd.concat([df_reviews, to_append], ignore_index=True)
        for id1, token1 in df_reviews.iterrows():
            for id2, token2 in df_reviews.iterrows():
                if (token1['reviewID'] != token2['reviewID'] and
                    token1['score'] != token2['score'] and
                        token1['cluster'] == token2['cluster']):
                    if not G.has_node(token1['reviewID']):
                        G.add_node(token1['reviewID'], weight=token1['readability'])
                    if not G.has_node(token2['reviewID']):
                        G.add_node(token2['reviewID'], weight=token2['readability'])
                    try:
                        w1 = token1['readability'] * token1['importance']
                    except Exception:
                        w1 = 0
                    try:
                        w2 = token2['readability'] * token2['importance']
                    except Exception:
                        w2 = 0
                    w1 = 0 if math.isnan(w1) else w1
                    w2 = 0 if math.isnan(w2) else w2
                    try:
                        sim_t1_t2 = row['matrix'].at[token1['token'], token2['token']]
                    except Exception:
                        sim_t1_t2 = 0
                    weight = 0
                    if w1 > w2:
                        if G.has_edge(token1['reviewID'], token2['reviewID']):
                            weight = int(G.get_edge_data(token1['reviewID'], token2['reviewID'],
                                         'weight')['weight'])
                            G.remove_edge(token1['reviewID'], token2['reviewID'])
                            G.add_edge(token1['reviewID'], token2['reviewID'],
                                       weight=weight + (w1 - w2) * sim_t1_t2)
                        elif G.has_edge(token2['reviewID'], token1['reviewID']):
                            weight = int(G.get_edge_data(token2['reviewID'], token1['reviewID'],
                                         'weight')['weight']) * -1
                            weight = weight + (w1 - w2) * sim_t1_t2
                            if weight > 0:
                                G.remove_edge(token2['reviewID'], token1['reviewID'])
                                G.add_edge(token1['reviewID'], token2['reviewID'], weight=weight)
                            else:
                                if G.has_edge(token1['reviewID'], token2['reviewID']):
                                    G.remove_edge(token1['reviewID'], token2['reviewID'])
                                G.add_edge(token2['reviewID'], token1['reviewID'],
                                           weight=weight * -1)
                        else:
                            G.add_edge(token1['reviewID'], token2['reviewID'],
                                       weight=weight + (w1 - w2) * sim_t1_t2)
                    elif w2 > w1:
                        if G.has_edge(token2['reviewID'], token1['reviewID']):
                            weight = int(G.get_edge_data(token2['reviewID'], token1['reviewID'],
                                         'weight')['weight'])
                            G.remove_edge(token2['reviewID'], token1['reviewID'])
                            G.add_edge(token2['reviewID'], token1['reviewID'],
                                       weight=weight + (w2 - w1) * sim_t1_t2)
                        elif G.has_edge(token1['reviewID'], token2['reviewID']):
                            weight = int(G.get_edge_data(token1['reviewID'], token2['reviewID'],
                                         'weight')['weight']) * -1
                            weight = weight + (w2 - w1) * sim_t1_t2
                            if weight > 0:
                                G.remove_edge(token1['reviewID'], token2['reviewID'])
                                G.add_edge(token2['reviewID'], token1['reviewID'], weight=weight)
                            else:
                                if G.has_edge(token2['reviewID'], token1['reviewID']):
                                    G.remove_edge(token2['reviewID'], token1['reviewID'])
                                G.add_edge(token1['reviewID'], token2['reviewID'],
                                           weight=weight * -1)
                        else:
                            G.add_edge(token2['reviewID'], token1['reviewID'],
                                       weight=weight + (w2 - w1) * sim_t1_t2)
        r = solve_argumentation_graph_json(G)
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
                prodID, reviewerID, time = node['id'].upper().split("_")
                df.loc[(df['asin'] == prodID) & (df['reviewerID'] == reviewerID),
                       'solutions'] = node['state']
            draw_graph(G, model, df, savename)
    df_results.to_pickle(os.path.join(savename, "results.pkl"))
    df.to_csv(os.path.join(savename, "reviews_res.csv"))


if __name__ == "__main__":
    t1 = ttime.time()
    file = input('Please provide the file path (csv) for the input data (reviews): ')
    file2 = input('Please provide the file path (pkl) for the input data (mc product list): ')
    df_prods = pd.read_pickle(file2, compression='gzip')
    df = pd.read_csv(file, compression='gzip')
    run_solver(df_prods, df, "")
    print('Duration:', ttime.time()-t1)
