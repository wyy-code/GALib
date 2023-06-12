"""
This script contains the data I/O operations for Gromov-Wasserstein Learning

For the graphs without mutual connections, we sample their subsets independently
For the graphs with some mutual connections, we sample their subsets jointly
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# for synthetic data
def build_dict_syn(num_node: int, noise: float):
    """
    Create synthetic graph pairs with various noise level
    Args:
        num_node: the number of nodes in each graph
        noise: the level of noise, e.g., 25% of edges are random edges

    Returns:

    """
    src_index = {}
    tar_index = {}
    for i in range(num_node):
        src_index[float(i)] = i

    num_node_noise = int((1+noise) * num_node)
    for i in range(num_node_noise):
        tar_index[float(i)] = i

    real_interactions = []
    noisy_interactions = []
    num_edge = 0
    for i in range(num_node):
        # real edges
        idx = np.random.permutation(num_node)
        idx.tolist()
        # how many edges
        degree = np.random.permutation(int(num_node/10))
        degree.tolist()
        degree = degree[0]+1
        # count of the edges
        weight = np.random.permutation(10)
        weight.tolist()
        for d in range(degree):
            if idx[d] != i:
                j = idx[d]
                count = weight[d]+1
                num_edge += 1
                for c in range(count):
                    real_interactions.append([i, j])
    if noise > 0:
        for i in range(num_node_noise):
            # noisy edges
            idx = np.random.permutation(num_node_noise)
            idx.tolist()
            # how many edges
            degree = np.random.permutation(int(noise * num_node / 10)+1)
            degree.tolist()
            degree = degree[0] + 1
            # count of the edges
            weight = np.random.permutation(10)
            weight.tolist()
            for d in range(degree):
                if idx[d] != i:
                    j = idx[d]
                    count = weight[d] + 1
                    num_edge += 1
                    for c in range(count):
                        noisy_interactions.append([i, j])

    database = {'src_index': src_index,
                'tar_index': tar_index,
                'src_interactions': real_interactions,
                'tar_interactions': real_interactions + noisy_interactions,
                'mutual_interactions': None}
    return database


# for protein-protein interaction dataset
def read_tab_ppi(data_path: str):
    """
    Load node pairs from .tab file
    Args:
        data_path: the path of tab file

    Returns:
        node2idx: a dictionary, whose key is node name and value is index
        interactions: a list containing the pair of protein's interactions
    """
    node2idx = {}
    interactions = []
    idx = 0
    with open(data_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            tab_pos = line.find('\t')
            end_pos = line.find('\n')
            node_src = line[:tab_pos]
            node_dst = line[tab_pos + 1:end_pos]
            # print("Line {}: {}".format(cnt, line.strip()))
            # print('src: {}, dst: {}'.format(node_src, node_dst))
            #
            # generate node index
            import pdb 
            pdb.set_trace()
            if node_src not in node2idx.keys():
                node2idx[node_src] = idx
                idx += 1
            if node_dst not in node2idx.keys():
                node2idx[node_dst] = idx
                idx += 1

            # generate interaction list
            idx_src = node2idx[node_src]
            idx_dst = node2idx[node_dst]
            interactions.append([idx_src, idx_dst])

            if cnt % 10000 == 0:
                print('{}: {} rows are processed.'.format(data_path, cnt))
            line = fp.readline()
            cnt += 1
    return node2idx, interactions


def build_dict_ppi(src_net_path: str, tar_net_path: str) -> Dict:
    """
    This function builds the protein-protein interaction (PPI) network database

    Args:
        src_net_path: the path of source network's tab file
        tar_net_path: the path of target network's tab file

    Returns:
        database = {src_index: the dictionary mapping protein to index
                    tar_index: the dictionary mapping protein to index
                    src_interactions: the list containing the node pairs in the source protein network
                    tar_interactions: the list containing the node pairs in the target protein network
                    }
    """
    src_index, src_interactions = read_tab_ppi(src_net_path)
    add_interactions = []
    for n in range(8):
        for i in range(2**(n+1)):
            add_interactions = add_interactions + src_interactions[n::(i+1)]
    for n in range(4):
        for i in range(3**(n+1)):
            add_interactions = add_interactions + src_interactions[n::(i+1)]
    for n in range(4):
        for i in range(5**(n+1)):
            add_interactions = add_interactions + src_interactions[n::(i+1)]
    for n in range(3):
        for i in range(6**(n+1)):
            add_interactions = add_interactions + src_interactions[n::(i+1)]

    tar_index, tar_interactions = read_tab_ppi(tar_net_path)
    database = {'src_index': src_index,
                'tar_index': tar_index,
                'src_interactions': src_interactions + add_interactions,
                'tar_interactions': tar_interactions + add_interactions,
                'mutual_interactions': None}
    return database


# mc3 (email and call) dataset
def read_csv_mc3(data_path: str, node2idx: Dict, pair2idx: List=None):
    """
    This function load interactions within a graph
    Args:
        data_path: the path of network
        node2idx: a dictionary whose key is worker ID and value is worker index.
                  It indicates the workers we care about.
        pair2idx: a list of valid node pairs

    Returns:
        interactions: the list containing the node pairs in the network
    """
    df = pd.read_csv(data_path)  # , encoding="ISO-8859-1")#"utf8")
    interactions = []
    for i, row in df.iterrows():
        src = str(row[0])
        dst = str(row[2])
        # time = str(row[3])
        if src in node2idx.keys() and dst in node2idx.keys():
            idx1 = node2idx[src]
            idx2 = node2idx[dst]
            if [idx1, idx2] not in interactions:
                if pair2idx is not None:
                    if [idx1, idx2] in pair2idx:
                        interactions.append([idx1, idx2])
                else:
                    interactions.append([idx1, idx2])
        if i % 10000 == 0:
            print('{}: row {}/{}'.format(data_path, i, len(df)))
    return interactions


def build_dict_mc3(src_net_path: str, tar_net_path: str, node2idx: Dict, pair2idx: List=None):
    """
    Build database for the worker interaction data in MC3

    Args:
        src_net_path: the path of source network's csv file
        tar_net_path: the path of target network's csv file
        node2idx: a dictionary whose key is worker ID and value is worker index.
                  It indicates the workers we care about.
        pair2idx: a list of valid node pairs

    Returns:
        database = {src_index: the dictionary mapping workerID to index
                    tar_index: the dictionary mapping workerID to index
                    src_interactions: the list containing the node pairs in the source network
                    tar_interactions: the list containing the node pairs in the target network
                    }
    """
    src_interactions = read_csv_mc3(src_net_path, node2idx, pair2idx)
    tar_interactions = read_csv_mc3(tar_net_path, node2idx, pair2idx)
    database = {'src_index': node2idx,
                'tar_index': node2idx,
                'src_interactions': src_interactions,
                'tar_interactions': tar_interactions,
                'mutual_interactions': None}
    return database


# mimic-III dataset
def build_dict_mimic3(diagnose_dict_path: str,
                      diagnose_adm_path: str,
                      procedure_dict_path: str,
                      procedure_adm_path: str,
                      min_count: int):
    """
    This function builds the icd code database

    Args:
        diagnose_dict_path: the path of diagnose icd code list (csv)
        diagnose_adm_path: diagnose_adm_path: the full path of admission diagnose csv file
        procedure_dict_path: procedure_dict_path: the path of procedure icd code list (csv)
        procedure_adm_path: procedure_adm_path: the full path of admission procedure csv file
        min_count: the minimum counts of ICD code

    Returns:
        database = {src_index: the dictionary mapping diagnose ICD code to index
                    src_title: the dictionary mapping diagnose ICD code to its description
                    tar_index: the dictionary mapping procedure ICD code to index
                    tar_title: the dictionary mapping procedure ICD code to its description
                    src_interactions: the diagnose pairs
                    tar_interactions: the procedure pairs
                    mutual_interactions: the list containing the admission with diseases and procedures
                    }

    """
    df_diagnose = pd.read_csv(diagnose_adm_path)  # , encoding="ISO-8859-1")#"utf8")
    diag_counts = df_diagnose['ICD9_CODE'].value_counts()
    diag2idx = {}
    idx = 0
    for icd in diag_counts.keys():
        if diag_counts[icd] > min_count:
            diag2idx[str(icd)] = idx
            idx += 1

    df_procedure = pd.read_csv(procedure_adm_path)  # , encoding="ISO-8859-1")#"utf8")
    proc_counts = df_procedure['ICD9_CODE'].value_counts()
    proc2idx = {}
    idx = 0
    for icd in proc_counts.keys():
        if proc_counts[icd] > min_count:
            proc2idx[str(icd)] = idx
            idx += 1

    diag2title = {}
    df_diagnose = pd.read_csv(diagnose_dict_path)  # , encoding="ISO-8859-1")#"utf8")
    idx = 0
    for i, row in df_diagnose.iterrows():
        icd = str(row['ICD9_CODE'])
        des = str(row['LONG_TITLE'])
        if icd in diag2idx.keys():
            diag2title[icd] = des
            idx += 1
    logger.info('{} kinds of diagnoses are found.'.format(len(diag2idx)))

    proc2title = {}
    df_procedure = pd.read_csv(procedure_dict_path)  # , encoding="ISO-8859-1")#"utf8")
    idx = 0
    for i, row in df_procedure.iterrows():
        icd = str(row['ICD9_CODE'])
        des = str(row['LONG_TITLE'])
        if icd in proc2idx.keys():
            proc2title[icd] = des
            idx += 1
    logger.info('{} kinds of procedures are found.'.format(len(proc2idx)))

    diag_adm = {}
    df_diagnose = pd.read_csv(diagnose_adm_path)  # , encoding="ISO-8859-1")#"utf8")
    for i, row in df_diagnose.iterrows():
        adm = str(row['HADM_ID'])
        icd = str(row['ICD9_CODE'])
        if icd in diag2idx.keys():
            if adm not in diag_adm.keys():
                diag_adm[adm] = [diag2idx[icd]]
            else:
                diag_adm[adm].append(diag2idx[icd])
        if i % 10000 == 0:
            logger.info('{}/{} rows are processed.'.format(i, len(df_diagnose)))
    logger.info('{} diagnose admissions are found.'.format(len(diag_adm)))

    proc_adm = {}
    df_procedure = pd.read_csv(procedure_adm_path)  # , encoding="ISO-8859-1")#"utf8")
    for i, row in df_procedure.iterrows():
        adm = str(row['HADM_ID'])
        icd = str(row['ICD9_CODE'])
        if icd in proc2idx.keys():
            if adm not in proc_adm.keys():
                proc_adm[adm] = [proc2idx[icd]]
            else:
                proc_adm[adm].append(proc2idx[icd])
        if i % 10000 == 0:
            logger.info('{}/{} rows are processed.'.format(i, len(df_procedure)))
    logger.info('{} procedure admissions are found.'.format(len(proc_adm)))

    diag_w_proc = []
    for adm in diag_adm.keys():
        if adm in proc_adm.keys():
            diag_w_proc.append([diag_adm[adm], proc_adm[adm]])

    database = {'src_index': diag2idx,
                'src_title': diag2title,
                'tar_index': proc2idx,
                'tar_title': proc2title,
                'src_interactions': diag_adm,
                'tar_interactions': proc_adm,
                'mutual_interactions': diag_w_proc}
    return database


class IndexSampler(Dataset):
    """Sampling indices via minbatch"""
    def __init__(self, num: int):
        """
        :param num: the number of indices
        """
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return idx  # torch.LongTensor([idx])


def cost_sampler1(database, indices, device):
    """
    Sample a set of source entities and target entities from a database
    Args:
        database: a dictionary containing observed data
            database = {src_index: the dictionary mapping source entity to index
                        src_title: the dictionary mapping source entity to its description
                        tar_index: the dictionary mapping target entity to index
                        tar_title: the dictionary mapping target entity to its description
                        interactions: the list containing the interactions/coherent behaviors between source and target
                        }
        indices: the index of the element in database['mutual_interactions']
        device: the device storing the variables

    Returns:
        cost_s: (Ns, Ns) matrix representing the relationships among source entities
        cost_t: (Nt, Nt) matrix representing the relationships among target entities
        mu_s: (Ns, 1) vector representing marginal probability of source entities
        mu_t: (Nt, 1) vector representing marginal probability of target entities
        prior: (Ns, Nt) matrix representing the prior of proposed optimal transport
        index_s: (Ns,) a LongTensor indicating the indices of selected source entities
        index_t: (Nt,) a LongTensor indicating the indices of selected target entities
    """
    src_count = {}
    tar_count = {}
    src_index = {}
    tar_index = {}
    idx_src = 0
    idx_tar = 0
    for n in range(indices.size(0)):
        interactions = database['mutual_interactions'][indices[n]]
        source = interactions[0]
        target = interactions[1]
        for s in source:
            if s not in src_index.keys():
                src_index[s] = idx_src
                src_count[s] = 1
                idx_src += 1
            else:
                src_count[s] += 1

        for t in target:
            if t not in tar_index.keys():
                tar_index[t] = idx_tar
                tar_count[t] = 1
                idx_tar += 1
            else:
                tar_count[t] += 1

    ns = len(src_index)
    nt = len(tar_index)
    cost_s = torch.zeros(ns, ns)
    cost_t = torch.zeros(nt, nt)
    mu_s = torch.zeros(ns, 1)
    mu_t = torch.zeros(nt, 1)
    prior = torch.zeros(ns, nt)
    index_s = torch.zeros(ns)
    index_t = torch.zeros(nt)

    for s in src_index.keys():
        idx = src_index[s]
        mu_s[idx, 0] = src_count[s]
        index_s[idx] = s
    mu_s /= mu_s.sum()
    index_s = index_s.type(torch.LongTensor)

    for t in tar_index.keys():
        idx = tar_index[t]
        mu_t[idx, 0] = tar_count[t]
        index_t[idx] = t
    mu_t /= mu_t.sum()
    index_t = index_t.type(torch.LongTensor)

    for n in range(indices.size(0)):
        interactions = database['mutual_interactions'][indices[n]]
        source = interactions[0]
        target = interactions[1]
        for s in source:
            idx_s = src_index[s]
            for t in target:
                idx_t = tar_index[t]
                prior[idx_s, idx_t] += 1

        if len(source) > 1:
            for i in range(len(source)-1):
                s1 = src_index[source[i]]
                for j in range(i+1, len(source)):
                    s2 = src_index[source[j]]
                    cost_s[s1, s2] += 1

        if len(target) > 1:
            for i in range(len(target)-1):
                t1 = tar_index[target[i]]
                for j in range(i+1, len(target)):
                    t2 = tar_index[target[j]]
                    cost_t[t1, t2] += 1
    # # the cost mimics the correlation coefficient
    # cost_s /= (cost_s.max())
    # cost_s = 1 - cost_s
    # cost_t /= (cost_t.max())
    # cost_t = 1 - cost_t
    # prior /= (prior.max())
    # prior = 1 - prior
    mask_s = cost_s > 0
    mask_s = mask_s.type(torch.FloatTensor)
    mask_t = cost_t > 0
    mask_t = mask_t.type(torch.FloatTensor)
    mask_st = prior > 0
    mask_st = mask_st.type(torch.FloatTensor)

    # convert correlation to distance
    cost_s = 1 / (cost_s + 1)
    cost_s -= torch.diag(torch.diag(cost_s))
    cost_t = 1 / (cost_t + 1)
    cost_t -= torch.diag(torch.diag(cost_t))
    prior = 1 / (prior + 1)

    cost_s = cost_s.to(device)
    cost_t = cost_t.to(device)
    mu_s = mu_s.to(device)
    mu_t = mu_t.to(device)
    prior = prior.to(device)
    index_s = index_s.to(device)
    index_t = index_t.to(device)
    return cost_s, cost_t, mu_s, mu_t, index_s, index_t, prior, mask_s, mask_t, mask_st


def cost_sampler2(database, indices1, indices2, device):
    """
    Sample a set of source entities and target entities from a database
    Args:
        database: a dictionary containing observed data
            database = {src_index: the dictionary mapping source entity to index
                        src_title: the dictionary mapping source entity to its description
                        tar_index: the dictionary mapping target entity to index
                        tar_title: the dictionary mapping target entity to its description
                        interactions: the list containing the interactions/coherent behaviors between source and target
                        }
        indices1: the index of the element in database['src_interactions']
        indices2: the index of the element in database['tar_interactions']
        device: the device storing the variables

    Returns:
        cost_s: (Ns, Ns) matrix representing the relationships among source entities
        cost_t: (Nt, Nt) matrix representing the relationships among target entities
        mu_s: (Ns, 1) vector representing marginal probability of source entities
        mu_t: (Nt, 1) vector representing marginal probability of target entities
        index_s: (Ns,) a LongTensor indicating the indices of selected source entities
        index_t: (Nt,) a LongTensor indicating the indices of selected target entities
        prior: (Ns, Nt) matrix representing the prior of proposed optimal transport
    """
    src_count = {}
    tar_count = {}
    src_index = {}
    tar_index = {}
    idx_src = 0
    idx_tar = 0

    for n in range(indices1.size(0)):
        idx1, idx2 = database['src_interactions'][indices1[n]]
        if idx1 not in src_index.keys():
            src_index[idx1] = idx_src
            src_count[idx1] = 1
            idx_src += 1
        else:
            src_count[idx1] += 1

        if idx2 not in src_index.keys():
            src_index[idx2] = idx_src
            src_count[idx2] = 1
            idx_src += 1
        else:
            src_count[idx2] += 1

    for n in range(indices2.size(0)):
        idx1, idx2 = database['tar_interactions'][indices2[n]]
        if idx1 not in tar_index.keys():
            tar_index[idx1] = idx_tar
            tar_count[idx1] = 1
            idx_tar += 1
        else:
            tar_count[idx1] += 1

        if idx2 not in tar_index.keys():
            tar_index[idx2] = idx_tar
            tar_count[idx2] = 1
            idx_tar += 1
        else:
            tar_count[idx2] += 1

    ns = len(src_index)
    nt = len(tar_index)
    cost_s = torch.zeros(ns, ns)
    cost_t = torch.zeros(nt, nt)
    mu_s = torch.zeros(ns, 1)
    mu_t = torch.zeros(nt, 1)
    index_s = torch.zeros(ns)
    index_t = torch.zeros(nt)

    for s in src_index.keys():
        idx = src_index[s]
        mu_s[idx, 0] = src_count[s]
        # s
        # idx
        index_s[idx] = float(s)
    mu_s /= mu_s.sum()
    index_s = index_s.type(torch.LongTensor)

    for t in tar_index.keys():
        idx = tar_index[t]
        mu_t[idx, 0] = tar_count[t]
        index_t[idx] = float(t)
    mu_t /= mu_t.sum()
    index_t = index_t.type(torch.LongTensor)

    for n in range(indices1.size(0)):
        idx1, idx2 = database['src_interactions'][indices1[n]]
        s1, s2 = src_index[idx1], src_index[idx2]
        cost_s[s1, s2] += 1

    for n in range(indices2.size(0)):
        idx1, idx2 = database['tar_interactions'][indices2[n]]
        t1, t2 = tar_index[idx1], tar_index[idx2]
        cost_t[t1, t2] += 1

    # # the cost mimics the correlation coefficient
    # cost_s /= (cost_s.max())
    # cost_s = 1 - cost_s
    # cost_t /= (cost_t.max())
    # cost_t = 1 - cost_t
    mask_s = cost_s > 0
    mask_s = mask_s.type(torch.FloatTensor)
    mask_t = cost_t > 0
    mask_t = mask_t.type(torch.FloatTensor)

    # convert correlation to distance
    cost_s = 1/(cost_s + 1)
    cost_s -= torch.diag(torch.diag(cost_s))
    cost_t = 1/(cost_t + 1)
    cost_t -= torch.diag(torch.diag(cost_t))

    cost_s = cost_s.to(device)
    cost_t = cost_t.to(device)
    mask_s = mask_s.to(device)
    mask_t = mask_t.to(device)
    mu_s = mu_s.to(device)
    mu_t = mu_t.to(device)
    index_s = index_s.to(device)
    index_t = index_t.to(device)
    return cost_s, cost_t, mu_s, mu_t, index_s, index_t, mask_s, mask_t
