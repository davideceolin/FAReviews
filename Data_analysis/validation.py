from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.svm import SVC
import pandas as pd
import spacy
from spacy_readability import Readability
import pytextrank  # noqa: F401
# from operator import itemgetter
import numpy as np
from sklearn.model_selection import cross_val_score
from spacy.language import Language

nlp = spacy.load('en_core_web_md')

# @Language.component("textrank")
# def textrank(doc):
#    tr = pytextrank.TextRank()
#    doc = tr.PipelineComponent(doc)
#   return doc


@Language.component("readability")
def readability(doc):
    read = Readability()
    doc = read(doc)
    return doc


nlp.add_pipe("textrank", last=True)
nlp.add_pipe("readability", last=True)


def enrich_row(row):
    try:
        doc = nlp(row['reviewText'])
        row['num_token'] = len(doc)
        row['readability'] = doc._.flesch_kincaid_reading_ease
    except Exception:
        row['num_token'] = 0
        row['readability'] = 0
    try:
        row['length'] = len(row['reviewText'])
        row['num_words'] = len(row['reviewText'].split())
    except Exception:
        row['length'] = 0
        row['num_words'] = 0
    return row[['num_token', 'readability', 'length', 'num_words']]


df = pd.read_csv('reviews_res.csv')
df.loc[df['solutions'] == 'undec', 'solutions'] = 'in'
df.drop_duplicates(subset=["reviewerID", "reviewText", "asin", "vote"], keep=False, inplace=True)
df = df.drop_duplicates()
df = df.fillna(0)
# df = df.loc[df['vote']>0, ]
df = df.reset_index()
# print(df['vote'].value_counts())

df_features = df.apply(enrich_row, axis=1)
# print(df_features)
# df['weight'] = df['unixReviewTime'].apply(lambda x: ((np.log10(x)-9)*10)-1)
# df['weight'] = (df['unixReviewTime']-df['unixReviewTime'].min())/(df['unixReviewTime'].max()-
#                 df['unixReviewTime'].min())
# df['weighted_vote'] = df['weight']*df['vote']
cluster = KMeans(n_clusters=2, random_state=10)
df.loc[:, 'cluster_labels'] = cluster.fit_predict(df_features)
'''

print(df[['cluster_labels', 'vote']])
print(pd.pivot_table(df[['cluster_labels', 'vote']], aggfunc=np.mean, index='cluster_labels',
                     values='vote'))
print(pd.pivot_table(df[['cluster_labels', 'vote']], aggfunc=np.sum, index='cluster_labels',
                     values='vote'))
print(pd.pivot_table(df[['cluster_labels', 'vote']], aggfunc=np.std, index='cluster_labels',
                     values='vote'))
'''

# print(pd.pivot_table(df[['cluster_labels', 'weighted_vote']], aggfunc=np.mean,
#                      index='cluster_labels', values='weighted_vote'))
# print(pd.pivot_table(df[['cluster_labels', 'weighted_vote']], aggfunc=np.sum,
#                      index='cluster_labels', values='weighted_vote'))

'''
print("Precision @1 K-means:")
print(len(df.loc[(df['vote'] >= 1) & (df['cluster_labels'] == 1),
      'cluster_labels'])/len(df.loc[df['cluster_labels'] == 1, 'vote']))
print("Precision @5 K-means:")
print(len(df.loc[(df['vote'] >= 5) & (df['cluster_labels'] == 1),
      'cluster_labels'])/len(df.loc[df['cluster_labels'] == 1, 'vote']))
print("Precision @10 K-means:")
print(len(df.loc[(df['vote'] >= 10) & (df['cluster_labels'] == 1),
      'cluster_labels'])/len(df.loc[df['cluster_labels'] == 1, 'vote']))

print("Recall @1 K-means:")
print(len(df.loc[(df['vote'] >= 1) & (df['cluster_labels'] == 1),
      'cluster_labels'])/len(df.loc[df['vote'] >= 1, 'vote']))
print("Recall @5 K-means:")
print(len(df.loc[(df['vote'] >= 5) & (df['cluster_labels'] == 1),
      'cluster_labels'])/len(df.loc[df['vote'] >= 5, 'vote']))
print("Recall @10 K-means:")
print(len(df.loc[(df['vote'] >= 10) & (df['cluster_labels'] == 1),
      'cluster_labels'])/len(df.loc[df['vote'] >= 10, 'vote']))

print("PrecisionW @1 K-means:")
print(np.sum(df.loc[(df['vote'] >= 1) & (df['cluster_labels'] == 1),
      'cluster_labels'])/np.sum(df.loc[df['cluster_labels'] == 1, 'vote']))
print("PrecisionW @5 K-means:")
print(np.sum(df.loc[(df['vote'] >= 5) & (df['cluster_labels'] == 1),
      'cluster_labels'])/np.sum(df.loc[df['cluster_labels'] == 1, 'vote']))
print("PrecisionW @10 K-means:")
print(np.sum(df.loc[(df['vote'] >= 10) & (df['cluster_labels'] == 1),
      'cluster_labels'])/np.sum(df.loc[df['cluster_labels'] == 1, 'vote']))

print("RecallW @1 K-means:")
print(np.sum(df.loc[(df['vote'] >= 1) & (df['cluster_labels'] == 1),
      'cluster_labels'])/np.sum(df.loc[df['vote'] >= 1, 'vote']))
print("RecallW @5 K-means:")
print(np.sum(df.loc[(df['vote'] >= 5) & (df['cluster_labels'] == 1),
      'cluster_labels'])/np.sum(df.loc[df['vote'] >= 5, 'vote']))
print("RecallW @10 K-means:")
print(np.sum(df.loc[(df['vote'] >= 10) & (df['cluster_labels'] == 1),
      'cluster_labels'])/np.sum(df.loc[df['vote'] >= 10, 'vote']))
'''
df_features['vote'] = df['vote']
pd.set_option('display.max_rows', None)

print("vote:")
# print(df_features['cluster'].unique())
# df_features['weight'] = (df['unixReviewTime']-df['unixReviewTime'].min())/(
#                          df['unixReviewTime'].max()-df['unixReviewTime'].min())
# #df['unixReviewTime'].apply(lambda x: ((np.log10(x)-9)*10)-1)
df_features['cluster'] = df['vote'].apply(lambda x: 0 if x < 10 else 1)
# df_features['weighted_vote'].apply(round)

n_train = round(len(df_features)*0.3)
trainingset = df_features.loc[:n_train, ]
testset = df_features.loc[n_train+1:len(df_features), ]
print(len(trainingset))
print(len(testset))
# print(df_features)
# print(testset)
# print(trainingset)
# print(n_train)
clf = SVC(gamma='auto')
clf.fit(trainingset.drop(columns=['cluster', 'vote']), trainingset['cluster'])
testset.loc[:, 'cluster'] = clf.predict(testset.drop(columns=['cluster', 'vote']))

print(pd.pivot_table(testset[['cluster', 'vote']], aggfunc=np.mean, index='cluster', values='vote'))
print(pd.pivot_table(testset[['cluster', 'vote']], aggfunc=np.sum, index='cluster', values='vote'))

print("Precision @1 SVC@10:")
print(len(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1),
      'cluster'])/len(testset.loc[testset['cluster'] == 1, 'vote']))
print("Precision @5 SVC@10:")
print(len(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1),
      'cluster'])/len(testset.loc[testset['cluster'] == 1, 'vote']))
print("Precision @10 SVC@10:")
print(len(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1),
      'cluster'])/len(testset.loc[testset['cluster'] == 1, 'vote']))

print("Recall @1 SVC@10:")
print(len(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1),
      'cluster'])/len(testset.loc[testset['vote'] >= 1, 'vote']))
print("Recall @5 SVC@10:")
print(len(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1),
      'cluster'])/len(testset.loc[testset['vote'] >= 5, 'vote']))
print("Recall @10 SVC@10:")
print(len(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1),
      'cluster'])/len(testset.loc[testset['vote'] >= 10, 'vote']))
'''
print("PrecisionW @1 SVC@10:")
print(np.sum(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1),
      'cluster'])/np.sum(testset.loc[testset['cluster'] == 1, 'vote']))
print("PrecisionW @5 SVC@10:")
print(np.sum(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1),
      'cluster'])/np.sum(testset.loc[testset['cluster'] == 1, 'vote']))
print("PrecisionW @10 SVC@10:")
print(np.sum(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1),
      'cluster'])/np.sum(testset.loc[testset['cluster'] == 1, 'vote']))

print("RecallW @1 SVC@10:")
print(np.sum(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1),
      'cluster'])/np.sum(testset.loc[testset['vote'] >= 1, 'vote']))
print("RecallW @5 SVC@10:")
print(np.sum(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1),
      'cluster'])/np.sum(testset.loc[testset['vote'] >= 5, 'vote']))
print("RecallW @10 SVC@10:")
print(np.sum(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1),
      'cluster'])/np.sum(testset.loc[testset['vote'] >= 10, 'vote']))
scores = cross_val_score(clf, df_features.drop('cluster', axis=1).drop('vote', axis=1),
                         df_features['cluster'], cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''

df_features['vote'] = df['vote']
# df_features['weighted_vote'] = df['weight']*df['vote'].apply(lambda x: 0 if x < 5 else 1)
df_features['cluster'] = df['vote'].apply(lambda x: 0 if x < 5 else 1)
# df_features['weighted_vote'].apply(round)

n_train = round(len(df_features)*0.3)
trainingset = df_features.loc[:n_train, ]
testset = df_features.loc[n_train+1:len(df_features), ]

clf = SVC(gamma='auto')
clf.fit(trainingset.drop('cluster', axis=1).drop('vote', axis=1), trainingset['cluster'])
SVC(gamma='auto')
testset.loc[:, 'cluster'] = clf.predict(testset.drop('cluster', axis=1).drop('vote', axis=1))

# print(pd.pivot_table(testset[['cluster', 'vote']],aggfunc=np.mean,index='cluster', values='vote'))
# print(pd.pivot_table(testset[['cluster', 'vote']],aggfunc=np.sum,index='cluster', values='vote'))
# print(pd.pivot_table(testset[['cluster','vote']],aggfunc=np.std,index='cluster', values='vote'))
# print(pd.pivot_table(testset[['cluster', 'weighted_vote']], aggfunc=np.mean, index='cluster',
# values='weighted_vote'))
# print(pd.pivot_table(testset[['cluster', 'weighted_vote']], aggfunc=np.sum, index='cluster',
# values='weighted_vote'))

print("Precision @1 SVC@5:")
print(len(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['cluster'] == 1, 'vote']))
print("Precision @5 SVC@5:")
print(len(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['cluster'] == 1, 'vote']))
print("Precision @10 SVC@5:")
print(len(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['cluster'] == 1, 'vote']))

print("Recall @1 SVC@5:")
print(len(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['vote'] >= 1, 'vote']))
print("Recall @5 SVC@5:")
print(len(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['vote'] >= 5, 'vote']))
print("Recall @10 SVC@5:")
print(len(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['vote'] >= 10, 'vote']))

print("PrecisionW @1 SVC@5:")
print(np.sum(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['cluster'] == 1, 'vote']))
print("PrecisionW @5 SVC@5:")
print(np.sum(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['cluster'] == 1, 'vote']))
print("PrecisionW @10 SVC@5:")
print(np.sum(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['cluster'] == 1, 'vote']))

print("RecallW @1 SVC@5:")
print(np.sum(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['vote'] >= 1, 'vote']))
print("RecallW @5 SVC@5:")
print(np.sum(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['vote'] >= 5, 'vote']))
print("RecallW @10 SVC@5:")
print(np.sum(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['vote'] >= 10, 'vote']))

scores = cross_val_score(clf, df_features.drop('cluster', axis=1).drop('vote', axis=1),
                         df_features['cluster'], cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# df_features['vote'] = df['vote']
# df_features['weighted_vote'] = df['weight']*df['vote'].apply(lambda x: 0 if x < 10 else 1)
df_features['cluster'] = df['vote'].apply(lambda x: 0 if x < 1 else 1)

n_train = round(len(df_features)*0.3)
trainingset = df_features.loc[:n_train, ]
testset = df_features.loc[n_train+1:len(df_features), ]

clf = SVC(gamma='auto')
clf.fit(trainingset.drop('cluster', axis=1).drop('vote', axis=1), trainingset['cluster'])
# SVC(gamma='auto')
testset.loc[:, 'cluster'] = clf.predict(testset.drop('cluster', axis=1).drop('vote', axis=1))

# print(pd.pivot_table(testset[['cluster','vote']],aggfunc=np.mean,index='cluster', values='vote'))
# print(pd.pivot_table(testset[['cluster','vote']],aggfunc=np.sum,index='cluster', values='vote'))
# print(pd.pivot_table(testset[['cluster','vote']],aggfunc=np.std,index='cluster', values='vote'))
# print(pd.pivot_table(testset[['cluster', 'weighted_vote']], aggfunc=np.mean, index='cluster',
# values='weighted_vote'))
# print(pd.pivot_table(testset[['cluster', 'weighted_vote']], aggfunc=np.sum, index='cluster',
# values='weighted_vote'))

print("Precision @1 SVC@1:")
print(len(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['cluster'] == 1, 'vote']))
print("Precision @5 SVC@1:")
print(len(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['cluster'] == 1, 'vote']))
print("Precision @10 SVC@1:")
print(len(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['cluster'] == 1, 'vote']))

print("Recall @1 SVC@1:")
print(len(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['vote'] >= 1, 'vote']))
print("Recall @5 SVC@1:")
print(len(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['vote'] >= 5, 'vote']))
print("Recall @10 SVC@1:")
print(len(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1), 'cluster'])/len(
      testset.loc[testset['vote'] >= 10, 'vote']))

print("PrecisionW @1 SVC@1:")
print(np.sum(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['cluster'] == 1, 'vote']))
print("PrecisionW @5 SVC@1:")
print(np.sum(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['cluster'] == 1, 'vote']))
print("PrecisionW @10 SVC@1:")
print(np.sum(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['cluster'] == 1, 'vote']))

print("RecallW @1 SVC@1:")
print(np.sum(testset.loc[(testset['vote'] >= 1) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['vote'] >= 1, 'vote']))
print("RecallW @5 SVC@1:")
print(np.sum(testset.loc[(testset['vote'] >= 5) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['vote'] >= 5, 'vote']))
print("RecallW @10 SVC@1:")
print(np.sum(testset.loc[(testset['vote'] >= 10) & (testset['cluster'] == 1), 'cluster'])/np.sum(
      testset.loc[testset['vote'] >= 10, 'vote']))
scores = cross_val_score(clf, df_features.drop('cluster', axis=1).drop('vote', axis=1),
                         df_features['cluster'], cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# df_features['vote'] = df['vote']
df_features['weighted_vote'] = df['vote']
df_features['cluster'] = df_features['weighted_vote'].apply(round)

n_train = round(len(df_features)*0.3)
trainingset = df_features.loc[:n_train, ]
testset = df_features.loc[(n_train+1):len(df_features), ]


clf = SVC(gamma='auto')
clf.fit(trainingset.drop(columns=['cluster', 'weighted_vote', 'vote']), trainingset['vote'])
testset.loc[:, 'cluster'] = clf.predict(testset.drop(columns=['cluster', 'weighted_vote', 'vote']))

# print(pd.pivot_table(testset[['cluster','vote']],aggfunc=np.mean,index='cluster', values='vote'))
# print(pd.pivot_table(testset[['cluster','vote']],aggfunc=np.sum,index='cluster', values='vote'))
# print(pd.pivot_table(testset[['cluster','vote']],aggfunc=np.std,index='cluster', values='vote'))
print(pd.pivot_table(testset[['cluster', 'weighted_vote']], aggfunc=np.mean, index='cluster',
      values='weighted_vote'))
pd.set_option('display.max_rows', None)
print('cluster1')
testset['cluster1'] = testset['cluster'].apply(lambda x: 0 if x < 1 else 1)
print(testset.loc[testset['cluster1'] == 1, 'weighted_vote'].mean())
print(testset.loc[testset['cluster1'] == 0, 'weighted_vote'].mean())
print(testset.loc[testset['cluster1'] == 1, 'weighted_vote'].sum())
print(testset.loc[testset['cluster1'] == 0, 'weighted_vote'].sum())
print('cluster2')
testset['cluster2'] = testset['cluster'].apply(lambda x: 0 if x < 5 else 1)
print(testset.loc[testset['cluster2'] == 1, 'weighted_vote'].mean())
print(testset.loc[testset['cluster2'] == 0, 'weighted_vote'].mean())
print(testset.loc[testset['cluster2'] == 1, 'weighted_vote'].sum())
print(testset.loc[testset['cluster2'] == 0, 'weighted_vote'].sum())
print('cluster3')
testset['cluster3'] = testset['cluster'].apply(lambda x: 0 if x < 10 else 1)
print(testset.loc[testset['cluster3'] == 1, 'weighted_vote'].mean())
print(testset.loc[testset['cluster3'] == 0, 'weighted_vote'].mean())
print(testset.loc[testset['cluster3'] == 1, 'weighted_vote'].sum())
print(testset.loc[testset['cluster3'] == 0, 'weighted_vote'].sum())
med = testset['cluster'].mean()
print("cluster4")
print(med)
testset['cluster4'] = testset['cluster'].apply(lambda x: 0 if x < med else 1)
print(testset.loc[testset['cluster4'] == 1, 'weighted_vote'].mean())
print(testset.loc[testset['cluster4'] == 0, 'weighted_vote'].mean())
print(testset.loc[testset['cluster4'] == 1, 'weighted_vote'].sum())
print(testset.loc[testset['cluster4'] == 0, 'weighted_vote'].sum())
testset['cluster_err'] = testset['cluster'] - testset['weighted_vote']
print(testset['cluster_err'].mean())


print(len(testset.loc[testset['cluster'] > 0, ])/len(testset.loc[testset['weighted_vote'] > 0, ]))
print(len(testset.loc[testset['cluster'] > 4, ])/len(testset.loc[testset['weighted_vote'] > 4, ]))
print(len(testset.loc[testset['cluster'] > 9, ])/len(testset.loc[testset['weighted_vote'] > 9, ]))
scores = cross_val_score(clf, X=df_features.drop(columns=['cluster', 'vote', 'weighted_vote']),
                         y=df_features['vote'], cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
