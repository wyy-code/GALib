#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# From Fasttext project: https://github.com/facebookresearch/fastText

import codecs
import sys
import time
import math
import argparse
import ot
import numpy as np
from scipy import sparse
from . import utils


def objective(X, Y, R, n=5):
    Xn, Yn = X[:n], Y[:n]
    C = -np.dot(np.dot(Xn, R), Yn.T)
    P = ot.sinkhorn(np.ones(n), np.ones(n), C, 0.025, stopThr=1e-3)
    return 1000 * np.linalg.norm(np.dot(Xn, R) - np.dot(P, Yn)) / n


def sqrt_eig(x):
    U, s, VT = np.linalg.svd(x, full_matrices=False)
    return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))


def align(X, Y, R, lr=1.0, bsz=10, nepoch=5, niter=10,
          nmax=10, reg=0.05, verbose=True, project_every=True):
    for epoch in range(1, nepoch + 1):
        for _it in range(1, niter + 1):
            # sample mini-batch
            xt = X[np.random.permutation(nmax)[:bsz], :]
            yt = Y[np.random.permutation(nmax)[:bsz], :]
            # compute OT on minibatch
            C = -np.dot(np.dot(xt, R), yt.T)
            # print bsz, C.shape
            P = ot.sinkhorn(np.ones(bsz), np.ones(bsz), C, reg, stopThr=1e-3)
            # print P.shape, C.shape
            # compute gradient
            # print "random values from embeddings:", xt, yt
            # print "sinkhorn", np.isnan(P).any(), np.isinf(P).any()
            #Pyt = np.dot(P, yt)
            # print "Pyt", np.isnan(Pyt).any(), np.isinf(Pyt).any()
            G = - np.dot(xt.T, np.dot(P, yt))
            # print "G", np.isnan(G).any(), np.isinf(G).any()
            update = lr / bsz * G
            print(("Update: %.3f (norm G %.3f)" %
                   (np.linalg.norm(update), np.linalg.norm(G))))
            R -= update

            # project on orthogonal matrices
            if project_every:
                U, s, VT = np.linalg.svd(R)
                R = np.dot(U, VT)
        niter //= 4
        if verbose:
            print(("epoch: %d  obj: %.3f" % (epoch, objective(X, Y, R))))
    if not project_every:
        U, s, VT = np.linalg.svd(R)
        R = np.dot(U, VT)
    return R, P


def convex_init(X, Y, niter=10, reg=1.0, apply_sqrt=False):
    n, d = X.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    K_X, K_Y = np.dot(X, X.T), np.dot(Y, Y.T)
    K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
    K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
    P = np.ones([n, n]) / float(n)
    for it in range(1, niter + 1):
        G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
        q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3)
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
    # print(obj)
    return utils.procrustes(np.dot(P, X), Y).T, P


def convex_init_sparse(X, Y, K_X=None, K_Y=None, niter=10, reg=1.0, apply_sqrt=False, P=None):
    if P is not None:  # already given initial correspondence--then just procrustes
        return utils.procrustes(P.dot(X), Y).T, P
    n, d = X.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    if K_X is None:
        K_X = np.dot(X, X.T)
    if K_Y is None:
        K_Y = np.dot(Y, Y.T)
    # print((type(K_X), K_X.shape))
    # print((type(K_Y), K_Y.shape))
    K_Y = sparse.linalg.norm(K_X) / sparse.linalg.norm(K_Y) * K_Y  # CHANGED
    K2_X, K2_Y = K_X.dot(K_X), K_Y.dot(K_Y)
    # print K_X, K_Y, K2_X, K2_Y
    K_X, K_Y, K2_X, K2_Y = K_X.toarray(), K_Y.toarray(), K2_X.toarray(), K2_Y.toarray()
    P = np.ones([n, n]) / float(n)

    for it in range(1, niter + 1):
        # if it % 10 == 0:
        #     print(it)
        G = P.dot(K2_X) + K2_Y.dot(P) - 2 * K_Y.dot(P.dot(K_X))
        # G = G.todense() #TODO how to get around this??
        q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3)
        q = sparse.csr_matrix(q)
        # print q.shape
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    obj = np.linalg.norm(P.dot(K_X) - K_Y.dot(P))
    # print(obj)
    return utils.procrustes(P.dot(X), Y).T, P
