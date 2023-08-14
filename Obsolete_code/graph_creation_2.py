# from compute_scores import Node
# import pickle5 as pickle
# import pickle
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
import ast
# from pengines.Builder import PengineBuilder
# from pengines.Pengine import Pengine
# from prologterms import TermGenerator, PrologRenderer, Program, Var, Rule, Term, Const
from prologterms import Const
import math
from nltk.corpus import stopwords
# import sys
# import json
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np

# P = TermGenerator()
# Model = Var('Model')

# pickle_off = open("prods_mc.pkl", "rb")
# df_prods = pickle.load(pickle_off)
df_prods = pd.read_pickle("AMAZON_FASHION_5_prods_mc.pkl")
# df_prods = df_prods.drop(columns=['nodes'])
# df_prods.to_pickle("prods_mc.pkl")
df = pd.read_csv("AMAZON_FASHION_5_reviews.csv", compression='gzip')
stop_words = stopwords.words('english')

df['solutions'] = "undec"


def color_node(solution):
    if solution == 'undec':
        return 'grey'
    elif solution == 'in':
        return 'green'
    else:
        return 'red'


def draw_graph(G, model):
    print(G.nodes.data())
    print(G.edges.data())
    pos = nx.layout.spring_layout(G)
    node_sizes = [i[1]['weight']*2 for i in G.nodes.data()]
    # node_sizes = [3 + 10 * i for i in range(len(G))]
    M = max(G.edges.data(), key=itemgetter(1))[1]  # noqa: F841

    # M = G.number_of_edges()
    edge_colors = np.argsort([x[2]['weight'] for x in G.edges.data()])  # range(2, M + 2)
    sol = [df.loc[(df['asin'] == x[0].split("_")[0]) & (df['reviewerID'] == x[0].split("_")[1]),
           'solutions'].apply(color_node).values.tolist() for x in G.nodes.data()]
    sol = [y[0] for y in sol]
    print(sol)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=sol)  # noqa: F841
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                   arrowsize=10, edge_color=edge_colors,
                                   edge_cmap=plt.cm.Blues, width=2)
    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    ax = plt.gca()
    ax.set_axis_off()
    prod_id = list(G.nodes())[0].split("_")[0]
    plt.savefig(prod_id+'.png', bbox_inches='tight')
    nx.draw_networkx_labels(G, pos, G.nodes, font_size=8)
    p = plt.savefig(prod_id+'_labels.png', bbox_inches='tight')
    plt.clf()
    plt.close(p)


def solve_argumentation_graph(G):
    prog = []
    edges = [x for x in G.edges()]
    print("edges:")
    print(edges)
    if isinstance(G, nx.DiGraph):
        for pair in edges:
            prog.append(Const(pair[0].lower()) >> Const(pair[1].lower()))
    elif isinstance(G, nx.Graph):
        for line in edges:
            for pair in line:
                prog.append(Const(pair[0].lower()) >> Const(pair[1].lower()))
    return prog


def solve_argumentation_graph_json(G):
    data = json_graph.node_link_data(G)
    r = requests.post('http://localhost:3333/argue', json=data)
    return r.json()


df_results = pd.DataFrame(columns=["prodID", "solutions"])

for index, row in df_prods.iterrows():
    df_reviews = pd.DataFrame(columns=['reviewID', 'score', 'token', 'importance', 'readability',
                              'cluster'])
    G = nx.DiGraph()
    # R = PrologRenderer()
    # p = Program(
    #    Rule(r'', body=P.include('argue'))
    # )
    # for cluster in list(set(row['clusters'])):
    # indexes = [i for i, x in enumerate(row['clusters']) if x == cluster]
    # tok = " ".join([filter(lambda y: y not in stop_words, x) for x in z.lower().split()
    #                for z in row['matrix']
    # .columns[indexes]])
    for id, review in df.loc[df['asin'] == row['prod']].iterrows():
        ranks = ast.literal_eval(review['ranks'])
        readability = review['readability']
        score = review['overall']
        reviewID = review['asin'] + "_" + review['reviewerID'] + "_" + str(review['unixReviewTime'])
        for token in ranks:
            tok = " ".join([x for x in token[0].lower().split() if x not in stop_words])
            if tok == "" or tok == " ":
                continue
            cluster = row[('matrix', 'clusters')][1][row[('matrix',
                                                         'clusters')][0].columns.get_loc(tok)]
            df_reviews = df_reviews.append(
                {'reviewID': reviewID, 'score': score, 'token': tok, 'importance': token[1],
                 'readability': readability, 'cluster': cluster}, ignore_index=True)
    # print("df_reviews")
    # print(df_reviews)
    for id1, token1 in df_reviews.iterrows():
        for id2, token2 in df_reviews.iterrows():
            if (token1['reviewID'] != token2['reviewID'] and token1['score'] != token2['score'] and
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
                            G.add_edge(token2['reviewID'], token1['reviewID'], weight=weight * -1)
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
                            G.add_edge(token1['reviewID'], token2['reviewID'], weight=weight * -1)
                    else:
                        G.add_edge(token2['reviewID'], token1['reviewID'],
                                   weight=weight + (w2 - w1) * sim_t1_t2)

            # else:
            #     if G.has_edge(token1['reviewID'],token2['reviewID']):
            #         weight = int(G.get_edge_data(token1['reviewID'], token2['reviewID'],
            #                      'weight')['weight'])
            #     G.add_edge(token1['reviewID'], token2['reviewID'],
            #                weight= weight + w1 * sim_t1_t2)
            #     if G.has_edge(token1['reviewID'],token2['reviewID']):
            #         weight = int(G.get_edge_data(token1['reviewID'], token2['reviewID'],
            #                      'weight')['weight'])
            #     G.add_edge(token1['reviewID'], token2['reviewID'],weight= weight + w2 * sim_t1_t2)

    # prog = solve_argumentation_graph(G)
    # print("prog:")
    # print(prog)
    # q = P.edges_model(prog, Model)
    '''
    factory = PengineBuilder(urlserver="https://swish.swi-prolog.org/", srctext=R.render(p),
                             ask=R.render(q), application='swish')
    try:
        pengine = Pengine(builder=factory, debug=True)

        while pengine.currentQuery.hasMore:
            pengine.doNext(pengine.currentQuery)
        for p in pengine.currentQuery.availProofs:
            df_results = df_results.append({'prodID': row['prod'], ' solutions': p[Model.name]},
                                           ignore_index=True)
            for res in p[Model.name]:
                #print(str(res))
                res = json.loads(str(res).replace("\'", "\""))
                prodID = res['args'][0].split("_")[0].upper()
                reviewerID = res['args'][0].split("_")[1].upper()
                df.loc[(df['asin'] >= prodID) & (df['reviewerID'] <= reviewerID)] = res['args'][1]
            print('{}'.format(p[Model.name]))
    except:
        print(sys.exc_info())
        df_results = df_results.append({'prodID': row['prod'], 'solutions': 'Error'},
                                       ignore_index=True)
        print('Error')
    '''
    r = solve_argumentation_graph_json(G)
    if 'models' in r and len(r['models']) > 0:
        models = r['models']
        weights = [0 for x in r['models']]
        for i in range(len(models)):
            for node in models[i]['nodes']:
                if node['state'] == 'in':
                    prodID, reviewerID, time = node['id'].upper().split("_")
                    weights[i] = weights[i] + node['weight']
                    # df.loc[(df['asin'] == prodID) & (df['reviewerID'] == reviewerID),
                    #        'readability'].iloc[0]
                    # print(df.loc[(df['asin'] == prodID) & (df['reviewerID'] == reviewerID),
                    #       'readability'])
        model = models[weights.index(max(weights))]
        for node in model['nodes']:
            prodID, reviewerID, time = node['id'].upper().split("_")
            df.loc[(df['asin'] == prodID) & (df['reviewerID'] == reviewerID),
                   'solutions'] = node['state']
        draw_graph(G, model)
        # df_results.to_pickle("results.pkl")
        # df.to_csv("reviews_res.csv")

df_results.to_pickle("results.pkl")
df.to_csv("reviews_res.csv")
# import sys

# import matplotlib.pyplot as plt
# import networkx as nx

# G = nx.DiGraph()
# G.add_edge(1, 2)
# G.add_edge(3, 2)

# prog = []
