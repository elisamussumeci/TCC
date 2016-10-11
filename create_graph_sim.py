import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy


def graph_simu(infects, G):
    s = infects.shape[1]
    graph_sim = np.zeros((s,s))

    #I_previous é o vetor de indices dos nos infectados no tempo 0
    I_previous = np.where(infects[0]==1)[0]

    #done é a lista com os indices dos nos que ja descobrimos o pai
    done = [I_previous[0]]

    for status in infects[1:]:
        I = list(np.where(status==1)[0])
        I_current = I.copy()
        for i in done:
            if i in I_current:
                I_current.remove(i)

        if len(I_current) == 0:
            continue

        #se temos mais de uma possibilidade de pai
        if len(I_previous) == 1:
            for node in I_current:
                graph_sim[node][I_previous[0]] = 1
                done.append(node)

        else:
            #calcular pai para cada no que foi infectado
            for node in I_current:
                #pegando a probabilidade de cada no ja infectado ter infectado o no node
                influences = G[node]
                probs = np.zeros(s)

                for i in I_previous:
                    probs[i] = influences[i]
                if sum(probs) != 0 :
                    probs = probs/sum(probs)
                pos = np.random.choice(range(s),p=probs,size=1)
                graph_sim[node][pos] = 1
                done.append(node)
                if pos == 0:
                    print(probs)
        I_previous = I

    return graph_sim


def graph_simu_nx(Infects, G):
    s = Infects.shape[1]
    graph_sim = nx.DiGraph()

    #I_previous é o vetor de indices dos nos infectados no tempo 0
    I_previous = np.where(Infects[0]==1)[0]

    #done é a lista com os indices dos nos que ja descobrimos o pai
    done = [I_previous[0]]

    for status in Infects[1:]:
        I = list(np.where(status==1)[0])
        I_current = I.copy()
        for i in done:
            if i in I_current:
                I_current.remove(i)

        if len(I_current) == 0:
            continue

        #se temos mais de uma possibilidade de pai
        if len(I_previous) == 1:
            for node in I_current:
                graph_sim.add_edge(I_previous[0],node)
                done.append(node)

        else:
            #calcular pai para cada no que foi infectado
            for node in I_current:
                #pegando a probabilidade de cada no ja infectado ter infectado o no node
                influences = G[node]
                probs = np.zeros(s)

                for i in I_previous:
                    probs[i] = influences[i]
                if sum(probs) != 0 :
                    probs = probs/sum(probs)
                pos = np.random.choice(range(s), p=probs, size=1)
                graph_sim.add_edge(pos[0], node)
                done.append(node)
                if pos == 0:
                    print(probs)
        I_previous = I

    return graph_sim


Infects = np.loadtxt(open('/home/elisa/Projetos/TCC/charlie_results/Infects.csv'), delimiter=",")
G = np.loadtxt(open('/home/elisa/Projetos/TCC/charlie_results/graph_complete.csv'), delimiter=",")
original_graph = nx.read_gpickle('/home/elisa/Projetos/TCC/charlie_results/original_graph.gpickle')
#
ori_outs = [i[1] for i in original_graph.out_degree_iter()]
g_ori = gaussian_kde(ori_outs)
#
graph_simulated = graph_simu_nx(Infects, G)
nx.write_gpickle(graph_simulated, '/home/elisa/Projetos/TCC/charlie_results/graph_simulated_nx.gpickle')
sim_outs = [j[1] for j in graph_simulated.out_degree_iter()]
g_sim = gaussian_kde(sim_outs)

plt.hist(sim_outs, bins=70, color='r', alpha=0.3, normed=True)
plt.hist(ori_outs, bins=30, color='b', alpha=0.3, normed=True)
plt.plot(range(0, 20), g_sim.evaluate(range(20)), color='r')
plt.plot(range(0, 20), g_ori.evaluate(range(20)), color='b')
plt.xlim(0, 20)
plt.savefig('graph_validation.png')
plt.show()


# for i in [0.017, 0.02]:
#     Infects = np.loadtxt(open('/home/elisa/Documents/Projetos/TCC/data/charlie/Infects_%s.csv' % i), delimiter=",")
#     graph_simulated = graph_simu_nx(Infects, G)
#     nx.write_gpickle(graph_simulated, '/home/elisa/Documents/Projetos/TCC/data/charlie/graph_simulated_nx%s.gpickle' % i)
#     sim_outs = [j[1] for j in graph_simulated.out_degree_iter()]
#     g_sim = gaussian_kde(sim_outs)
#
#     plt.hist(sim_outs, bins=20, color='r', alpha=0.3, normed=True)
#     plt.hist(ori_outs, bins=25, color='b', alpha=0.3, normed=True)
#     plt.title(i)
#     plt.plot(range(0, 20), g_sim.evaluate(range(20)), color='r')
#     plt.plot(range(0, 20), g_ori.evaluate(range(20)), color='b')
#     plt.xlim(0, 20)
#     plt.savefig('data/charlie/graph_sim_%s.png' % i)
#     plt.close()
#
#     e = entropy(g_sim.evaluate(range(20)), g_ori.evaluate(range(20)))
#     print('lambda:', i, e)
