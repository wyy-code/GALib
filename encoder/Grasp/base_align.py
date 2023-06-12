import scipy.optimize as opt
import autograd.numpy as np
from . import Grasp as fu
import sys
import matplotlib.pyplot as plt

import torch
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

D1 = None
D2 = None
V1 = None
V2 = None
Cor1 = None
Cor2 = None
k_ = None


def cost(X):
    mu = 0.132

    global D2
    global V1
    global V2
    global Cor1
    global Cor2
    global k_
    coup = (np.linalg.norm(Cor1.T@V1[:, 0:k_]-Cor2.T@V2[:, 0:k_]@X, 'fro'))**2

    res = (X.T @ np.diag(D2[0:k_]) @ X) ** 2
    diag_res = np.diagonal(res, offset=0, axis1=-1, axis2=-2)

    diag_res = np.sum(diag_res)

    sumres = np.sum(res)

    val = sumres - diag_res
    #val=np.linalg.norm(X.T @ diag2 @ X - diag2, 'fro') ** 2

    # print(coup)
    res = val+mu*coup
    return res


def optimize_AB(Cor11, Cor21, n, V11, V21, D11, D21, k):

    global D2
    global V1
    global V2
    global Cor1
    global Cor2
    global k_

    D2 = D21
    V1 = V11
    V2 = V21
    Cor1 = Cor11
    Cor2 = Cor21
    k_ = k

    manifold = Stiefel(k, k)
    x0 = init_x0(Cor1, Cor2, n, V1, V2, D1, D2, k)
    # x0=np.load('zwischenspeicher/B.npy')
    problem = Problem(manifold=manifold, cost=cost)

    # (3) Instantiate a Pymanopt solver

    #solver = pymanopt.solvers.conjugate_gradient.ConjugateGradient(maxtime=10000, maxiter=10000)
    solver = pymanopt.solvers.trust_regions.TrustRegions()  # maxtime=float('inf'))

    # let Pymanopt do the rest
    B = solver.solve(problem, x=x0)
    # print(B)
    # print(np.reshape(res.x[0:k*k_],(k_,k)).T@np.reshape(res.x[0:k*k_],(k_,k)))

    return B


def init_x0(Cor1, Cor2, n, V1, V2, D1, D2, k):

    B = np.identity(k)

    for i in range(0, k):
        thing1 = np.linalg.norm(Cor1.T@V1[:, i]-Cor2.T@V2[:, i])
        thing2 = np.linalg.norm(Cor1.T@V1[:, i]+Cor2.T@V2[:, i])
        #print('thing1: %f, thing2: %f' %(thing1,thing2))
        if(thing1 > thing2):
            B[i, i] = -1

    return B
