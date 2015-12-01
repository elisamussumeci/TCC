from bson import json_util
from gensim.models.word2vec import Word2Vec
from operator import itemgetter
from nltk.corpus import stopwords
from gensim import corpora, models
from string import punctuation
from urllib.parse import urlparse
import numpy as np
import nltk
import pandas as pd
import networkx as nx


def doc_gen(data):
    stopw = stopwords.words('portuguese') +['']
    for doc in data:
        try:
            list_words = nltk.tokenize.wordpunct_tokenize(doc['cleaned_text'])
        except KeyError:
            continue
        important_words = [word.lower().strip().strip(punctuation) for word in list_words if word.lower().strip().strip(punctuation) not in stopw]
        yield important_words

# create_model_matrix columns have the same order of the ids list:
#
#        ids[0] ids[1] ids[2] ...
# ids[0]
# ids[1]
# ids[2]
# ...


def create_model_matrix(data, model):

    # corpora with all words from data, dictionary[id] returns the word linked to id
    dictionary = corpora.Dictionary(doc_gen(data))

    # list of documents. corpus[0] is a list of tuples where each tuple contains the
    # a word id and the number of occurrences of that word in the document
    corpus = [dictionary.doc2bow(text) for text in doc_gen(data)]

    # tfidf[corpus[0]] is a list of tuples that contains the id of the word and the
    # tfidf of that word in that document.
    tfidf = models.TfidfModel(corpus)

    docvs = np.zeros((len(corpus), model.syn0.shape[1]), dtype=float)
    for n, doc in enumerate(corpus):
        ti = dict(tfidf[doc])
        for word in doc:
            w = dictionary[word[0]]
            try:
                word_vector = model[w]
                try:
                    assert np.isfinite(word_vector).all()
                except AssertionError:
                    print(w)
            except KeyError:
                word_vector = np.zeros((1,model.syn0.shape[1]))
                #             print('pulei ', w)
                #         word_vector.shape = 1, model.syn0.shape[1]

            docvs[n, :] = docvs[n, :] + word_vector*ti[word[0]]
    return docvs


def create_ids_list(data):
    ids = []
    stopw = stopwords.words('portuguese') +['']
    for doc in data:
        try:
            list_words = nltk.tokenize.wordpunct_tokenize(doc['cleaned_text'])
            ids.append(doc['_id'])
        except KeyError:
            continue
    return ids

# ##########Functions for graph creation ##############


def get_pos(data, pub_i, column_list, time_max, sim_min, outs):
    sim = max(column_list)
    if sim < sim_min:
        return None

    pos = column_list.index(sim)
    time_dif = (data[pos]['published'] - pub_i).total_seconds()/3600
    if pos in outs:
        column_list[pos] = 0
        get_pos(data, pub_i, column_list, time_max, sim_min, outs)
    elif time_dif > time_max:
        column_list[pos] = 0
        get_pos(data, pub_i, column_list, time_max, sim_min, outs)
    else:
        return pos


def create_graph(dists_triu, data, time_max=164, sim_min=0.8):
    size = dists_triu.shape[0]
    G = nx.DiGraph()
    G.add_node(0, step=0, date=data[0]['published'], domain=urlparse(data[0]['link']).netloc, _id=data[0]['_id'],
              children=[])
    outs = []
    for i in range(1,size):
        pub_i = data[i]['published']
        column = list(dists_triu[:,i])
        pos = get_pos(data, pub_i, column, time_max, sim_min, outs)

        if pos != None:
            if pos not in G.nodes():
                domain_1 = urlparse(data[pos]['link']).netloc
                G.add_node(pos, date=data[pos]['published'], domain=domain_1,
                           _id=data[pos]['_id'], children=[])
            if i not in G.nodes():
                domain_2 = urlparse(data[i]['link']).netloc
                G.add_node(i, date=pub_i, domain=domain_2, _id=data[i]['_id'], children=[])

            G.add_edge(pos, i)
        else:
            outs.append(i)
    return G

# domain_graph create a graph where the nodes are the website from the article


def create_date(pub1, pub2, s):
    dif = (pub2-pub1).total_seconds()/3600
    return round((dif/s))


def create_graphml(dists_triu, data, time_max=164, sim_min=0.8):
    size = dists_triu.shape[0]
    G = nx.DiGraph()
    G.add_node(0, step=0, date=0,domain=urlparse(data[0]['link']).netloc)
    date_init = data[0]['published']
    outs = []
    for i in range(1, size):
        pub_i = data[i]['published']
        column = list(dists_triu[:,i])
        pos = get_pos(data, pub_i, column, time_max, sim_min, outs)

        if pos != None:
            if pos not in G.nodes():
                domain_1 = urlparse(data[pos]['link']).netloc
                date_1 = create_date(date_init, data[pos]['published'], 5)
                G.add_node(pos, date=date_1, domain=domain_1)
            if i not in G.nodes():
                domain_2 = urlparse(data[i]['link']).netloc
                date_2 = create_date(date_init, pub_i, 5)
                G.add_node(i, date=date_2, domain=domain_2)

            G.add_edge(pos, i)
        else:
            outs.append(i)
    return G


def cria_atribs_step(G,list):
    for i in list:
        for node in G.successors(i):
            G.node[node]['step'] = G.node[i]['step'] + 15
        cria_atribs_step(G,G.successors(i))
    return G


def domain_graph(H):
    G = H.copy()
    one_week = 168

    # renomear variavel
    for indice in G.nodes():
        preds = G.predecessors(indice)
        if preds != []:
            pred = G.node[preds[0]]
            node = G.node[indice]
            time_dif = round((node['date']-pred['date']).total_seconds()/3600)
            if node['domain'] == pred['domain'] and time_dif < one_week:
                for suc in G.successors(indice):
                    G.add_edge(preds[0], suc)
                G.node[preds[0]]['children'].append(node['_id'])
                G.remove_node(indice)
    return G


def create_matrix_domain(graph):
    ## cria lista de domínios
    domain_list = []
    for pos in graph.nodes():
        node = graph.node[pos]
        d = node['domain']
        if d not in domain_list:
            domain_list.append(d)

    df = pd.DataFrame(0, index = domain_list, columns = domain_list)

    for pos in graph.nodes():
        node = graph.node[pos]
        d = node['domain']
        successors = graph.successors(pos)
        for suc in successors:
            df[d][graph.node[suc]['domain']] += 1

    return [domain_list, df]

## matriz foi criada de forma em que as colunas são os influenciadores e as linhas os influenciados!


def create_complete_adjacency(graph, matrix):
    df = pd.DataFrame(0, index=graph.nodes(), columns=graph.nodes())
    for column in graph.nodes():
        i_domains_column = matrix[graph.node[column]['domain']]
        for row in graph.nodes():
            prob = i_domains_column[graph.node[row]['domain']]
            df[column][row] = prob

    return df


# ############################ Load data ################################

with open('data/articles_data/charliehebdo.json') as f:
    a = f.read()
    list_of_dict = json_util.loads(a)

# Fixing some data issues
del_pos = []
for pos,i in enumerate(list_of_dict):
    if 'published' not in i:
        i['published'] = i['updated']
    if i['published'].year < 2015:
        del_pos.append(i)

for i in del_pos:
    list_of_dict.remove(i)

# sort the list from the first published article to the last one.
# data[0].published < data[1].published
articles = sorted(list_of_dict, key=itemgetter('published'))
del articles[0]
print('articles_imported')

# ############## Create Word2Vec Model Matrix #########################

# create ids list of articles in the corpus, corpus[0] is the ids[0] article
list_ids = create_ids_list(articles)

# load model that contains all words from media cloud articles database
modelMC = Word2Vec.load('Models/MediaCloud_w2v')

docvs = create_model_matrix(articles, modelMC)


# ############# Create Similarity Matrix ###############################

# normaliza toda a matriz docvs
docvs_norm = (docvs / np.sqrt((docvs ** 2).sum(-1))[..., np.newaxis]).astype('float')

# calcula distancia de todos os os documentos com todos os documentos
dists = np.dot(docvs_norm, docvs_norm.T)
dists_triu = np.triu(dists, k=1)

# ################ Create Graphs ###############################

G = create_graph(dists_triu, articles)
nx.write_gpickle(G, '/home/elisa/Documents/Projetos/TCC/data/charlie/original_graph.gpickle')

original_nodes = G.nodes()
np.savetxt('/home/elisa/Documents/Projetos/TCC/data/charlie/original_graph_nodes.csv', original_nodes, delimiter=',')

all_nodes_domains = []
for i in original_nodes:
    all_nodes_domains.append(G.node[i]['domain'])

f = open('/home/elisa/Documents/Projetos/TCC/data/charlie/graph_original_domains_each_node.txt','w')
for item in all_nodes_domains:
    f.write("%s\n" % item)

##########################################################################
domain_list, domain_matrix = create_matrix_domain(G)

graph_complete = create_complete_adjacency(G, domain_matrix)
as_numpy = graph_complete.as_matrix()
np.fill_diagonal(as_numpy, 0)
np.savetxt('/home/elisa/Documents/Projetos/TCC/data/charlie/graph_complete.csv', as_numpy, delimiter=',')





