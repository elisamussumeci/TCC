import numpy as np
import csv
from scipy.stats import bernoulli

G = np.loadtxt(open('/home/elisa/Documents/Projetos/TCC/data/charlie/graph_complete.csv'), delimiter=",")
nodes = np.loadtxt(open('/home/elisa/Documents/Projetos/TCC/data/charlie/original_graph_nodes.csv'), delimiter=",")

# load list of domains in data
with open('/home/elisa/Documents/Projetos/TCC/data/charlie/original_graph_domains_each_node.txt') as f:
    domains = f.read().splitlines()

# create dictionary, one colour to each domain
total_d = list(set(domains))
cols = colors.keys()
color_dict = {'a': 1}

for pos, i in enumerate(total_d):
    color_dict[i] = cols[pos]


#initial conditions
eig = np.linalg.eig(G)[0].max()

i0 = np.zeros(G.shape[0])
i0[0] = 1
s0 = 1-i0


def fun(t, y, pars):
    y = np.array(y)
    i,s = y[:1786],y[1786:]
    A, lamb = pars

    M =  (i * A).sum(axis=1)
    N = lamb * s
    Q = N*M

    dI = -i + Q
    dS = -Q

    return np.hstack((dI,dS))




def plot_sol(sol, color_dict, domains):
    plots = list_plot([(j[0],j[1][0]) for j in sol[:1786]], color=color_dict[domains[0]], plotjoined=True, alpha=.8, gridlines=true)
    for i in range(500):
        co = color_dict[domains[i]]
        plots += list_plot([(j[0], j[1][i]) for j in sol[:1786]], color=co, plotjoined=True, alpha=.2, gridlines=true)
    plots.save('/home/elisa/Documents/Projetos/TCC/data/charlie/simulation.png')



## dI- matrix com a probabilidade do artigo ser infectado no tempo t. dI[0] - artigos infectados no tempo 0
## Infects - matrix boolean com os infectados no tempo t.
## recebe T.solution

def create_dI(sol):
    s = len(T.solution[0][1])/2
    dI = np.zeros((len(T.solution), s))
    c = 0
    for i,v in sol:
        dI[c:] =v[:s]
        c+=1
    return dI


def create_infects(dI):
    B = lambda p: bernoulli.rvs(p, size=1)[0]
    Infects = np.vectorize(B)(dI)

    for pos in range(Infects.shape[1]):
        vector = Infects[:,pos]
        indexes = np.where(vector == 1)[0]
        if indexes != []:
            i_0,i_n = min(indexes), max(indexes)
            if i_n-i_0+1 != len(indexes):
                Infects[:,pos][i_0:i_n] = 1

    return Infects


def create_infected_matrix(la, Ti):
    T.ode_solve(t_span=[0, 5], y_0=list(i0)+list(s0), num_points=3000, params=[G, la])
    plot_sol(T.solution, color_dict, domains)

    dI = create_dI(T.solution)
    Infects = create_infects(dI)

    return dI, Infects


T = ode_solver()
T.algorithm = "rkf45"
T.function = fun
l = 1/eig + 0.015


dI, Infects = create_infected_matrix(l, T)
np.savetxt('/home/elisa/Documents/Projetos/TCC/data/charlie/dI.csv', dI, delimiter=',')
np.savetxt('/home/elisa/Documents/Projetos/TCC/data/charlie/Infects.csv', Infects, delimiter=',')


#for i in [0.017, 0.02]:
#    dI, Infects = create_infected_matrix(1/eig+i, T)
#    np.savetxt('/home/elisa/Documents/Projetos/TCC/data/charlie/dI_%s.csv' % i, dI, delimiter=',')
#    np.savetxt('/home/elisa/Documents/Projetos/TCC/data/charlie/Infects_%s.csv' % i, Infects, delimiter=',')
