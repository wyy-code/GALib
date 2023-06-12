
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.linalg._expm_frechet import vec
import sys
import math


def newbound_rounding_lowrank_evaluation_relaxed(U, V, bmatch):
    # np.set_printoptions(threshold=sys.maxsize)
    U_sortperm = np.argsort(U*-1, 0, 'stable')
    # U_sortperm[265][6]=887
    #U_sortperm[266][6] = 885
    #U_sortperm[843][2] = 887
    # U_sortperm[844][2]=885

    V_sortperm = np.argsort(V*-1, 0, 'stable')
    #V_sortperm[852][2] = 872
    #V_sortperm[853][2] = 184
    nU = np.shape(U)[0]
    nV = np.shape(V)[0]
    r = np.shape(U)[1]
    assert (r == np.shape(V)[1])
    d = min(np.shape(U_sortperm)[0], np.shape(V_sortperm)[0])
    U_weights = np.sort(U*-1, 0, 'stable')
    V_weights = np.sort(V*-1, 0, 'stable')
    U_weights = U_weights*-1
    V_weights = V_weights * -1
    #   U_weights = U_weights[1:d,:]
    #   V_weights = V_weights[1:d,:]
    #
    #   U_sortperm = U_sortperm[1:d,:]
    #   V_sortperm = V_sortperm[1:d,:]

    #   P = spzeros(nU,nV)
    U1 = []
    V1 = []
    allrecoveries = np.zeros(np.shape(U)[1])
    for i in range(np.shape(U)[1]):  # 1
        ui = U_weights[:, i]
        vi = V_weights[:, i]
        lastid_ui = None
        lastid_vi = None
        lastid_ui = next((x for x in ui if x < 0), None)
        lastid_vi = next((x for x in vi if x < 0), None)
       # print("testing")

        if lastid_ui == None and lastid_vi == None:

            lastidpos = d
            lneg = -1
        elif lastid_vi == None:
            lastid_ui = np.where(ui == lastid_ui)[0][0]
            lastidpos = min(d, lastid_ui-1)+1
            lneg = -1

        elif lastid_ui == None:
            lastid_vi = np.where(vi == lastid_vi)[0][0]
            lastidpos = min(d, lastid_vi-1)+1
            lneg = -1

        else:
            lastid_vi1 = np.where(vi == lastid_vi)[0][0]+1
            lastid_ui1 = np.where(ui == lastid_ui)[0][0]+1
            lastidpos = min(lastid_ui1, lastid_vi1)-1
            lneg = min(nU - lastid_ui1, nV - lastid_vi1)

        # print(lastidpos)
        # print(lneg)
        ei1 = U_sortperm[0:lastidpos, i]
        ej1 = V_sortperm[0:lastidpos, i]
        ei2 = U_sortperm[nU - lneg-1:nU, i]
        ej2 = V_sortperm[nV - lneg-1:nV, i]
        ei = np.hstack((ei1, ei2))
        ej = np.hstack((ej1, ej2))

        # allrecoveries[i] = evaluate_erdosreyni_experiment(A,B,ei,ej)
        #     P = P + sparse(ei,ej,1,nU,nV)

        # P = P + sparse(ei,ej,1,nU,nV) #+ generate_b_match_overlapping(ei,ej,2,nU,nV)
        # with bmatching:
        U1.extend(ei)
        V1.extend(ej)

        #bmatchval = 6
        for bm in range(0, bmatch):
            if not ej.size == 0:
                ej = np.delete(ej, 0)
                eilen = len(ei)-1
                # ej.pop(0)
                # P = P + sparse(ei[1:end-bm],ej,1,nU,nV)
                U1.extend(ei[0:eilen - bm])
                V1.extend(ej)
                # print(len(U1))
                # print(len(V1))
    # print("len")
    U1len = len(U1)
    V1len = len(V1)
    # print(U1len)
    # print(V1len)
    U1 = np.reshape(U1, (U1len, 1))
    V1 = np.reshape(V1, (V1len, 1))
    all_matches = np.hstack((U1, V1))
    y = np.ascontiguousarray(all_matches).view(
        np.dtype((np.void, all_matches.dtype.itemsize * all_matches.shape[1])))
    _, idx = np.unique(y, return_index=True)
    #unique_result = all_matches[(idx)]

    unique_result = all_matches[np.sort(idx)]
    U1unique = unique_result[:, 0]
    V1unique = unique_result[:, 1]
    U1unique1 = np.array(U1)
    V1unique1 = np.array(V1)
    # hi=np.loadtxt("U1unique.txt")
    # hi=np.array(hi,float)

    uo = U[U1unique, :]
    vo = V[V1unique, :]
    weights = vec(np.sum(np.multiply(uo, vo), 1))

    #X = scipy.sparse.csc_matrix((weights,(U1unique, V1unique)),shape=(nU, nV)).toarray()
    X = scipy.sparse.csc_matrix(
        (weights, (U1unique, V1unique)), shape=(nU, nV))
    return X, U1unique, V1unique, weights
    # return U1unique,V1unique,weights #alternative


def newbound_rounding_lowrank_evaluation_relaxed1(U, V, bmatch):
    # np.set_printoptions(threshold=sys.maxsize)
    U_sortperm = np.argsort(U*-1, 0, 'mergesort')
    V_sortperm = np.argsort(V*-1, 0, 'mergesort')
    nU = np.shape(U)[0]
    nV = np.shape(V)[0]
    r = np.shape(U)[1]
    assert (r == np.shape(V)[1])

    d = min(np.shape(U_sortperm)[0], np.shape(V_sortperm)[0])-1
    U_weights = np.sort(U*-1, 0)
    V_weights = np.sort(V*-1, 0)
    U_weights = U_weights*-1
    V_weights = V_weights * -1
    #   U_weights = U_weights[1:d,:]
    #   V_weights = V_weights[1:d,:]
    #
    #   U_sortperm = U_sortperm[1:d,:]
    #   V_sortperm = V_sortperm[1:d,:]

    #   P = spzeros(nU,nV)
    U1 = []
    V1 = []
    allrecoveries = np.zeros(np.shape(U)[1])
    for i in range(np.shape(U)[1]):  # 1
        ui = U_weights[:, i]
        vi = V_weights[:, i]
        lastid_ui = None
        lastid_vi = None
        lastid_ui = next((x for x in ui if x < 0), None)
        lastid_vi = next((x for x in vi if x < 0), None)
        if lastid_ui == None and lastid_vi == None:
            lastidpos = d
            lneg = -1
        elif lastid_vi == None:
            lastid_ui = np.where(ui == lastid_ui)[0][0]
            lastidpos = min(d, lastid_ui)
            lneg = -1
        elif lastid_ui == None:
            lastid_vi = np.where(vi == lastid_vi)[0][0]
            lastidpos = min(d, lastid_vi)
            lneg = -1
        else:
            lastid_vi1 = np.where(vi == lastid_vi)[0][0]
            lastid_ui1 = np.where(ui == lastid_ui)[0][0]
            lastidpos = min(lastid_ui1, lastid_vi1)
            lneg = min(nU - lastid_ui1, nV - lastid_vi1)-1
        ei1 = U_sortperm[0:lastidpos, i]
        ej1 = V_sortperm[0:lastidpos, i]
        ei2 = U_sortperm[nU - lneg-1:nU, i]
        ej2 = V_sortperm[nV - lneg-1:nV, i]
        ei = np.hstack((ei1, ei2))
        ej = np.hstack((ej1, ej2))

        # allrecoveries[i] = evaluate_erdosreyni_experiment(A,B,ei,ej)
        #     P = P + sparse(ei,ej,1,nU,nV)

        # P = P + sparse(ei,ej,1,nU,nV) #+ generate_b_match_overlapping(ei,ej,2,nU,nV)
        # with bmatching:
        U1.extend(ei)
        V1.extend(ej)
        #bmatchval = 6
        for bm in range(0, bmatch):
            if not ej.size == 0:
                ej = np.delete(ej, 0)
                eilen = len(ei)-1
                # ej.pop(0)
                # P = P + sparse(ei[1:end-bm],ej,1,nU,nV)
                U1.extend(ei[0:eilen - bm])
                V1.extend(ej)
    U1len = len(U1)
    V1len = len(V1)
    U1 = np.reshape(U1, (U1len, 1))
    V1 = np.reshape(V1, (V1len, 1))
    all_matches = np.hstack((U1, V1))
    y = np.ascontiguousarray(all_matches).view(
        np.dtype((np.void, all_matches.dtype.itemsize * all_matches.shape[1])))
    _, idx = np.unique(y, return_index=True)
    unique_result = all_matches[np.sort(idx)]

    U1unique = unique_result[:, 0]
    V1unique = unique_result[:, 1]
    uo = U[U1unique, :]
    vo = V[V1unique, :]
    weights = vec(np.sum((uo * vo), 1))

    #X = scipy.sparse.csc_matrix((weights,(U1unique, V1unique)),shape=(nU, nV)).toarray()
    # return X
    return U1unique, V1unique, weights  # alternative
