import numpy as np
from numpy.core._multiarray_umath import ndarray


def decomposeX_balance_allfactors(A, B, k, c1, c2, c3):
    nA = len(A[0])
    nB = len(B[0])
    u = np.divide(np.ones((nA)), nA)
    v = np.divide(np.ones((nB)), nB)
    du = np.zeros(k)
    dv = np.zeros(k)
    du[-1] = 1
    dv[-1] = 1
    eA = np.ones((nA))
    eB = np.ones((nB))
    U = np.zeros((nA, k))
    U[:, -1] = eA
    for i in range(k-2, 0, -1):
        U[:, i] = np.dot(A, U[:, i + 1])
        mui = np.amax(U[:, i])
        U[:, i] = np.divide(U[:, i], mui)
        du[i] = np.dot(du[i + 1], mui)

    U[:, 0] = u
    rksums = np.zeros(k-1)
    du[0] = 1
    for i in range(k-1):
        rksums[i] = np.dot(du[0], sum(U[:, 0]))
        U[:, 0] = np.dot(A, U[:, 0])
        mui = np.amax(U[:, 0])
        U[:, 0] = np.divide(U[:, 0], mui)
        du[0] = np.dot(du[0], mui)

    usums = sum(U, 0)
    usums = usums * du

    V = np.zeros((nB, k))
    V[:, -1] = eB
    for i in range(k-2, 0, -1):
        V[:, i] = np.dot(B, V[:, i + 1])
        mvi = np.amax(V[:, i])
        V[:, i] = np.divide(V[:, i], mvi)
        dv[i] = np.dot(dv[i + 1], mvi)

    V[:, 0] = v
    hksums = np.zeros(k - 1)
    dv[0] = 1
    for i in range(k-1):
        hksums[i] = dv[0] * sum(V[:, 0])
        V[:, 0] = np.dot(B, V[:, 0])
        mvi = np.amax(V[:, 0])
        V[:, 0] = np.divide(V[:, 0], mvi)
        dv[0] = np.dot(dv[0], mvi)
    vsums = sum(V, 0)
    vsums = vsums * dv

    rhos = np.copy(du)
    gams = np.copy(dv)

    W_prev = 1 * rhos[0] * gams[0]
    W1 = 1
    W2 = 1

    scalerow = 1
    scalecol = 1

    for i in range(1, k):
        localk = i-1
        rk = np.zeros(i)
        rk[0] = rksums[localk]
        rk[1:i] = usums[len(usums)-localk:]

        hk = np.zeros((i))
        hk[0] = hksums[localk]
        hk[1:i] = vsums[len(vsums) - localk:]
        Du = np.diag(np.divide(rhos[i], rhos[0:i]))
        Dv = np.diag(np.divide(gams[i], gams[0:i]))
        v1 = np.dot(Dv, hk)
        v2 = np.dot(Du, rk)
        temp1 = c1 * W_prev
        temp2 = np.dot(np.dot(scalecol * c2, W_prev), v1)
        temp3 = np.dot(np.dot(np.dot(scalerow, c2), v2.transpose()), W_prev)
        temp4 = np.dot(
            np.dot(np.dot(np.dot(scalerow * scalecol, c3), v2.transpose()), W_prev), v1)
        if (i > 1):
            temp12 = np.hstack(
                (temp1, np.reshape(temp2, (np.shape(temp2)[0], 1))))
        else:
            temp12 = np.hstack((temp1, temp2))
        temp34 = np.hstack((temp3, temp4))
        W_curr = np.vstack((temp12, temp34))
        if i == 1:
            temp12a = np.hstack((W_prev, np.zeros((i))))
            temp34a = np.hstack((np.zeros((i)), np.conj(rk)))
        else:
            temp12a = np.hstack((W_prev, np.zeros((i, i))))
            temp34a = np.hstack((np.zeros((i)), np.conj(rk)))
        W1 = np.vstack((temp12a, temp34a))
        if (i == 1):
            temp1b = c1 * np.ones(i)
            temp2b = np.multiply(c2, hk)
        else:
            temp1b = c1*np.eye(i)
            temp2b = np.dot(c2, np.reshape(hk, (np.shape(hk)[0], 1)))
        temp12b = np.hstack((temp1b, temp2b))
        temp3b = np.dot(c2, W_prev)
        if(i == 1):
            temp4b = np.dot(np.dot(c3, W_prev), hk)
        else:
            temp4b = np.dot(np.dot(c3, W_prev), hk)
            temp4b = np.reshape(temp4b, (i, 1))
        temp34b = np.hstack((temp3b, temp4b))
        W2 = np.vstack((temp12b, temp34b))
        W_curr = W_curr / W_curr[-1][-1]
        W_prev = W_curr
    return U, V, W_prev, W1, W2
