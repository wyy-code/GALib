import numpy as np

import scipy as sci

import networkx as nx

import time

import scipy.sparse as sps

try:
    import lapjv
except:
    pass
import os

import fast_pagerank

import contextlib

from sklearn.preprocessing import normalize

import argparse

from scipy.sparse import csr_matrix, coo_matrix

from sklearn.neighbors import KDTree

import math

import matplotlib.pyplot as plt

#import base_align as ba

#import munkres

from . import base_align_pymanopt as ba

#import base_align as ba

from sklearn.neighbors import NearestNeighbors
def sort_greedy_voting(match_freq):
    dist_platt=np.ndarray.flatten(match_freq)
    idx = np.argsort(dist_platt)#
    n = match_freq.shape[0]
    k=idx//n
    r=idx%n
    idx_matr=np.c_[k,r]
# print(idx_matr)
    G1_elements=set()
    G2_elements=set()
    i= n**2 - 1
    j= 0
    matching=np.ones([n,2])*(n+1)
    while(len(G1_elements)<n):
        if (not idx_matr[i,0] in G1_elements) and (not idx_matr[i,1] in G2_elements):
            #print(idx_matr[i,:])
            matching[j,:]=idx_matr[i,:]

            G1_elements.add(idx_matr[i,0])
            G2_elements.add(idx_matr[i,1])
            j+=1
            #print(len(G1_elements))


        i-=1

    # print(idx)
    #matching = np.c_[matching[:,0], matching[:,1]]
    #real_matching = dict(matching[matching[:, 0].argsort()])
    #return real_matching
    return matching[:,0],matching[:,1]
def parse_args():

    parser=argparse.ArgumentParser(description= "RUN GASP")

    parser.add_argument('--graph',nargs='?', default='arenas')

    # 1:nn 2:sortgreedy 3: jv 

    

    parser.add_argument('--cor_func', type=int, default= 3, help='Corresponding Function. 1=Heat Kernel,2=PageRank, 3=Personalized PageRank')

    



    parser.add_argument('--k_span', type=int, default=40)

    parser.add_argument('--voting', type=int, default=1, help='1= SortGreedy, 2=Hungarian')



    parser.add_argument('--laa', type=int, default= 3, help='Linear assignment algorithm. 1=nn,2=sortgreedy, 3=jv')

    parser.add_argument('--icp', type=bool, default=True)

    parser.add_argument('--ba', type=bool, default=True, help='Base alignment')

    parser.add_argument('--icp_its', type=int, default=3, help= 'how many iterations of iterative closest point')

    parser.add_argument('--q', type=int, default=20)

    parser.add_argument('--k', type= int, default=20)

    parser.add_argument('--lower_t', type=float,default=0.1, help='smallest timestep for corresponding functions')

    parser.add_argument('--upper_t', type=float,default=50.0, help='biggest timestep for corresponding functions')

    parser.add_argument('--linsteps', type=float, default=True, help='scaling of time steps of corresponding functions, logarithmically or linearly')

    parser.add_argument('--reps', type=int,default=5, help='number of repetitions per noise level')

    parser.add_argument('--noise_levels', type=list,default=[0,2,4,6,8,10])#,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

    return parser.parse_args()



def align_voting_heuristic(A1, A2,q,k,laa,icp, icp_its, lower_t, upper_t, linsteps, corr_func, ba_,k_span):

    k_list = list(range(max(2,int(k-k_span/2)),int(k+k_span/2)))

    matchings = [0]*len(k_list)

    # voting - 1:greedy, 2:jv

    # corr_func - 1:heat kern  2:pagerank 3: mix

    # 1:nn 2:sortgreedy 3: jv

    match = laa

    print("q,k,laa,icp, icp_its, lower_t, upper_t, linsteps, corr_func, ba_,k_span")

    print(q,k,laa,icp, icp_its, lower_t, upper_t, linsteps, corr_func, ba_,k_span)

    t = np.linspace(lower_t, upper_t, q)

    if (not linsteps):

        t = np.logspace(math.log(lower_t,10), math.log(upper_t,10), q)

    n = np.shape(A1)[0]

    start = time.time()

    D1, V1 = decompose_laplacian(A1)

    # # # #

    D2, V2 = decompose_laplacian(A2)



    if corr_func==2:

        Cor1 = calc_pagerank_corr_func(q, A1)

        Cor2 = calc_pagerank_corr_func(q, A2)

    elif corr_func==3:

        Cor1 = calc_personalized_pagerank_corr_func(q, A1)

        Cor2 = calc_personalized_pagerank_corr_func(q, A2)

    elif corr_func==1:

        Cor1 = calc_corresponding_functions(n, q, t, D1, V1)

        #print(Cor1)

        Cor2 = calc_corresponding_functions(n, q, t, D2, V2)

    



    

    print('Base Align')

    if ba_:

        B = ba.optimize_AB(Cor1, Cor2, n, V1, V2, D1, D2, k_list[-1]) #base alignment, manifold optimization

    print('Voting')

    for i in (range(len(k_list))):        

        #print('Base Align')

        if ba_:

            B_ = B[:k_list[i], :k_list[i]]

            V1_rot=V1[:,0:k_list[i]] # pick k eigenvectors

            V2_rot = V2[:, 0:k_list[i]] @ B_ # pick k eigenvectors and rotate via B

        else:

            V1_rot = V1

            V2_rot = V2



        

        #print('Calculate C')

        C = calc_C_as_in_quasiharmonicpaper(Cor1, Cor2,V1_rot,V2_rot,k_list[i],q)

        







        G1_emb =  V1_rot.T#[:, 0: k].T;



        G2_emb =C @ V2_rot.T#[:, 0: k].T;

        matching = []




        if (icp):

            matching = iterative_closest_point(V1_rot, V2_rot, C, icp_its, k_list[i], match,Cor1, Cor2,q)

        else:

            if match == 1:

                matching = greedyNN(G1_emb, G2_emb)

            if match == 2:

                matching = sort_greedy(G1_emb, G2_emb)

            if match == 3:

                matching = hungarian_matching(G1_emb, G2_emb)

            if match == 4:

                matching = top_x_matching(G1_emb, G2_emb,10)

            if match == 5:

                matching = nearest_neighbor_matching(G1_emb, G2_emb)

            if match == 6:

                matching = kd_align(G1_emb, G2_emb)


        
        end=time.time()

    #  np.savetxt('/home/au640507/spectral-graph-alignment/permutations_no_subset/arenas/noise_level_1/matching_' + str(

    #     i) + '.txt', matching, fmt="%d")

        if not icp: 

            matching = dict(matching.astype(int))

        #matching=matching.astype(int)

        matchings[i] = matching



    match_freq = np.zeros((n,n))

    for i in range(len(matchings)):

        for j in range(n):

            m = matchings[i][j]

            match_freq[j][m] += 1
    print(sort_greedy_voting(match_freq))
    print("stop")
    return None,(-match_freq + np.amax(match_freq))
    #return None,(-match_freq + np.amax(match_freq).T)



def greedyNN(G1_emb, G2_emb):

    #print('greedyNN: calculating distance matrix')



    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)

    n = np.shape(dist)[0]

    # print(np.shape(dist))

    #print('greedyNN: calculating matching')

    idx = np.argsort(dist, axis=0)

    matching=np.ones([n,1])*(n+1)

    for i in range(0,n):

        matched=False

        cur_idx=0

        while(not matched):

           #print([cur_idx,i])

           if(not idx[cur_idx,i] in matching):

               matching[i,0]=idx[cur_idx,i]



               matched=True

           else:

               cur_idx += 1

               #print(cur_idx)



    matching = np.c_[np.linspace(0, n-1, n).astype(int),matching]

    return matching.astype(int)



def sort_greedy(G1_emb, G2_emb):

    #print('sortGreedy: calculating distance matrix')



    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)

    n = np.shape(dist)[0]

    # print(np.shape(dist))

    #print('sortGreedy: calculating matching')

    dist_platt=np.ndarray.flatten(dist)

    idx = np.argsort(dist_platt)#

    k=idx//n

    r=idx%n

    idx_matr=np.c_[k,r]

   # print(idx_matr)

    G1_elements=set()

    G2_elements=set()

    i=0

    j=0

    matching=np.ones([n,2])*(n+1)

    while(len(G1_elements)<n):

        if (not idx_matr[i,0] in G1_elements) and (not idx_matr[i,1] in G2_elements):

            #print(idx_matr[i,:])

            matching[j,:]=idx_matr[i,:]



            G1_elements.add(idx_matr[i,0])

            G2_elements.add(idx_matr[i,1])

            j+=1

            #print(len(G1_elements))





        i+=1



   # print(idx)

    matching = np.c_[matching[:,1], matching[:,0]]

    matching = matching[matching[:, 0].argsort()]

    return matching.astype(int)



def top_x_matching(G1_emb, G2_emb,x):

    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)

    n = np.shape(dist)[0]



    idx = np.argsort(dist, axis=0)



    matches=idx[0:x,:]





    matching = np.c_[np.linspace(0, n-1, n).astype(int), matches.T]

    #matching = matching[matching[:, 0].argsort()]

    return matching.astype(int)



def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=10):

    kd_tree = KDTree(emb2, metric=distance_metric)

    if num_top > emb1.shape[0]:

        num_top = emb1.shape[0]

    row = np.array([])

    col = np.array([])

    data = np.array([])

    

    dist, ind = kd_tree.query(emb1, k=num_top)

    print("queried alignments")

    row = np.array([])

    for i in range(emb1.shape[0]):

        row = np.concatenate((row, np.ones(num_top) * i))

    col = ind.flatten()

    data = np.exp(-dist).flatten()

    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))

    # mat = sparse_align_matrix.tocsr().toarray()

    alignment_matrix = sparse_align_matrix.tocsr()

    n_nodes = alignment_matrix.shape[0]

    nodes_aligned = []

    counterpart_dict = {}



    if not sps.issparse(alignment_matrix):

        sorted_indices = np.argsort(alignment_matrix)



    for node_index in range(n_nodes):

        if sps.issparse(alignment_matrix):

            row, possible_alignments, possible_values = sps.find(alignment_matrix[node_index])

            node_sorted_indices = possible_alignments[possible_values.argsort()]

        else:

            node_sorted_indices = sorted_indices[node_index]

        

        for i in range(num_top):

            possible_node = (node_sorted_indices[-(i+1)])

            if not possible_node in nodes_aligned:

                counterpart = possible_node

                counterpart_dict[node_index] = counterpart

                nodes_aligned.append(counterpart)

                found_node = True

                break  

        if not found_node:

            for possible_node in range(n_nodes):

                if not possible_node in nodes_aligned:

                    counterpart = possible_node

                    counterpart_dict[node_index] = counterpart

                    nodes_aligned.append(counterpart)

                    break        



    # matches = np.argmax(mat, axis= 1)

    n = emb1.shape[0]

    matching = np.c_[ list(counterpart_dict.values()), np.linspace(0, n-1, n).astype(int)]

    matching = matching[np.argsort(matching[:,0])]

    matching = matching[matching[:, 0].argsort()]



    return matching



def calc_personalized_pagerank_corr_func(dim, A):

    pagerank_G = np.zeros((len(A), dim+1))

    

    pr_orig = fast_pagerank.pagerank(A,p= 0.80)

    

    signature_vector = pr_orig



    nodes_sorted = signature_vector.argsort()[::-1]



    avg = len(nodes_sorted) / float(dim)

    out = []

    last = 0.0



    while last < len(nodes_sorted):

        out.append(nodes_sorted[int(last):int(last + avg)])

        last += avg

    

    pagerank_G[:,0] = pr_orig

    for i in range(len(out)):

        #dic = { j : 1 for j in out[i]}

        pers = np.zeros(len(A))

        pers[out[i]] = 1

        

        pagerankG_vector = fast_pagerank.pagerank(A,p=0.80,personalize= pers)

        

        pagerank_G[:, i] = pagerankG_vector

    

    return pagerank_G    



def calc_pagerank_corr_func(no_of_alphas, A):

    pagerank_G = np.zeros((len(A), no_of_alphas))

    #if lower_q==0: lower_q = 0.001

    #alphas = np.flip(1-np.logspace(np.log10(0), np.log(0.8), no_of_alphas))  # experiment with different distribution

    alphas = np.linspace(0.6,1, no_of_alphas)

    if no_of_alphas == 1:

        alphas[0] = 0.8

    print("alphas:", alphas)

    for i in range(len(alphas)):

        pagerankG_vector = fast_pagerank.pagerank(A,p=alphas[i])

        pagerank_G[:, i] = pagerankG_vector

    return pagerank_G



def iterative_closest_point(V1, V2, C, it,k,match,Cor1,Cor2,q):

    G1= V1[:, 0: k].T

    G2_emb = V2[:, 0: k].T

    n = np.shape(G2_emb)[1]



    for i in range(0,it):



        #print('icp iteration '+str(i))

        G1_emb=C@V1[:,0:k].T

       # print('calculating hungarian in icp')

        M=[]



        if (match == 1):

            M = nearest_neighbor_matching(G1_emb, G2_emb)

        if match == 2:

            M = sort_greedy(G1_emb, G2_emb)

        if match == 3:

            M = hungarian_matching(G1_emb, G2_emb)

        if match == 4:

            M = top_x_matching(G1_emb, G2_emb,10)

        if match == 5:

            M = greedyNN(G1_emb, G2_emb)

        if match == 6:

            M = kd_align(G1_emb.T, G2_emb.T)

        G2_cur=np.zeros([k,n])

       ## print('finding nearest neighbors in eigenvector matrix icp')

        for j in range(0,n):



            G2idx = M[j, 1]

            G2_cur[:, G2idx]=G2_emb[:, j]

       ## print('calculating correspondence matrix in icp')

        #C=calc_correspondence_matrix(G1,G2_cur,k)

        C = calc_correspondence_matrix_ortho(G1, G2_cur, k)

    G1_emb = C@V1[:,0:k].T



    if (match == 1):

        M = nearest_neighbor_matching(G1_emb, G2_emb)

    if match == 2:

        M = sort_greedy(G1_emb, G2_emb)

    if match == 3:

        M = hungarian_matching(G1_emb, G2_emb)

    if match == 4:

        M = top_x_matching(G1_emb, G2_emb,10)

    if match == 5:

        M = greedyNN(G1_emb, G2_emb)

    if match == 6:

        M = kd_align(G1_emb.T, G2_emb.T)



    return dict(M.astype(int))

def calc_correspondence_matrix_ortho_diag(A, B, k):

    C = np.zeros([k,k])

    At = A.T

    Bt = B.T



    for i in range(0,k):

        C[i, i] = np.sign(np.linalg.lstsq(Bt[:,i].reshape(-1,1), At[:,i].reshape(-1,1),rcond=None)[0])



    return C



def calc_correspondence_matrix_ortho(A, B, k):

    #C = np.zeros([k,k])

    At = A.T

    Bt = B.T

    C=sci.linalg.orthogonal_procrustes(Bt,At)[0]

    C_norms=np.linalg.norm(C)

    C_normalized=normalize(C,axis=1)

    return C_normalized





def nearest_neighbor_matching(G1_emb, G2_emb):

    n= np.shape(G1_emb)[1]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(G1_emb.T)

    distances, indices = nbrs.kneighbors(G2_emb.T)

    indices=np.c_[np.linspace(0, n-1, n).astype(int), indices.astype(int)]



    return indices



def hungarian_matching(G1_emb, G2_emb):

    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)

    n = np.shape(dist)[0]


    try:

        cols, rows, _ = lapjv.lapjv(dist)
        matching = np.c_[cols, np.linspace(0, n-1, n).astype(int)]

    except Exception:

        cols, rows = sci.optimize.linear_sum_assignment(dist) 
        print(cols)      
        print(rows)
        matching = np.c_[rows, cols]
        print(matching)

    matching=matching[matching[:,0].argsort()]
    print(matching.astype(int))
    return matching.astype(int)

def calc_rotation_matrices(Cor1, Cor2, V1,V2,k,q):

    rotV1,_,rotV2=np.linalg.svd(V1[:,0:k].T@Cor1@Cor2.T@V2[:,0:k])

    return rotV1, rotV2.T



def calc_C_as_in_quasiharmonicpaper(Cor1,Cor2,V1,V2,k,q):

    leftside=Cor1.T@V1[:,0:k]

    rightside=V2[:,0:k].T@Cor2

    left=np.diag(leftside[0,:])

    right=rightside[:,0]

    for i in range(1,q):

        left=np.concatenate((left,np.diag(leftside[i,:])))

        right=np.concatenate((right,rightside[:,i]))

    C_diag=np.linalg.lstsq(left, right,rcond=None)[0]

    return np.diag(C_diag)

def decompose_laplacian(A):

    Deg = np.diag((np.sum(A, axis=1)))

    n = np.shape(Deg)[0]

    Deg=sci.linalg.fractional_matrix_power(Deg, -0.5) #D^-1/2

    L  = np.identity(n) - Deg @ A @ Deg

    D, V = np.linalg.eigh(L) # return eigenvalue vector, eigenvector matrix of L



    return [D, V]



def decompose_unnormalized_laplacian(A):



    #  adjacency matrix



    Deg = np.diag((np.sum(A, axis=1)))



    n = np.shape(Deg)[0]



    L = Deg- A


    D, V = np.linalg.eig(L)



    return [D, V]



def decompose_rw_normalized_laplacian(A):



    #  adjacency matrix



    Deg = np.diag((np.sum(A, axis=1)))



    n = np.shape(Deg)[0]



    L = np.identity(n) - np.linalg.inv(Deg) @ A

    D, V = np.linalg.eig(L)



    return [D, V]



def decompose_rw_laplacian(A):



    #  adjacency matrix



    Deg = np.diag((np.sum(A, axis=1)))



    n = np.shape(Deg)[0]



    L = np.linalg.inv(Deg) @ A



    D, V = np.linalg.eig(L)



    return [D, V]


def calc_corresponding_functions(n, q, t, d, V):



    # corresponding functions are the heat kernel diagonals in each time step

    # t= time steps, d= eigenvalues, V= eigenvectors, n= number of nodes, q= number of corresponding functions

    t = t[:, np.newaxis] #newxis increas dimension of array by 1

    d = d[:, np.newaxis]



    V_square = np.square(V)



    time_and_eigv = np.dot((d), np.transpose(t))



    time_and_eigv = np.exp(-1*time_and_eigv)



    Cores=np.dot(V_square, time_and_eigv)



    return Cores





def calc_coefficient_matrix(Corr, V, k, q):

    coefficient_matrix = np.linalg.lstsq(V[:,0:k],Corr,rcond=None)

    #print(type(coefficient_matrix))

    return coefficient_matrix[0]



def calc_correspondence_matrix(A, B, k):

    C = np.zeros([k,k])

    At = A.T

    Bt = B.T



    for i in range(0,k):

        C[i, i] = np.linalg.lstsq(Bt[:,i].reshape(-1,1), At[:,i].reshape(-1,1),rcond=None)[0]



    return C

def main(data, **args):

    # Src = data['Src'].A

    # Tar = data['Tar'].A

    Src = data['Src']

    Tar = data['Tar']

    B=align_voting_heuristic(Src,Tar,**args)

    #print(B)

    return B





if __name__ == '__main__':

    args = parse_args()

    main(args)    