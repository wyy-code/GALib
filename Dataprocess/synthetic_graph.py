import networkx as nx
import json
from networkx.readwrite import json_graph
import random
import os
import numpy as np
from Dataprocess.dataset import Dataset

class SyntheticGraph():
    """
    this class provided these functions:
    - generate_REGAL_synthetic: generate a graph with the algorithms mentioned in REGAL paper.
    - generate_random_clone_synthetic: generate a graph with probability of adding connection and probability of removing connection
    """
    def __init__(self, networkx_dir, output_dir1, output_dir2=None, groundtruth_dir=None, seed=12306):
        """

        :param networkx_dir: directory contains graph data in networkx format
        :param output_dir1: output directory for subgraph1
        :param output_dir2: output directory for subgraph2
        """
        self.networkx_dir = networkx_dir
        self.output_dir1 = output_dir1
        self.output_dir2 = output_dir2
        self.groundtruth_dir = groundtruth_dir
        self.seed = seed
    
    def set_seed(self):
        random.seed = self.seed
        np.random.seed = self.seed


    def generate_random_clone_synthetic(self, p_new_connection, p_remove_connection, p_change_feats=None):
        print("===============")
        dataset = Dataset(self.networkx_dir)
        G = self.random_clone_synthetic(dataset, p_new_connection, p_remove_connection, self.seed)
        self._save_graph(G, self.output_dir1, p_change_feats)

    def generate_REGAL_synthetic(self):
        self.generate_random_clone_synthetic(0, 0.05)

    def _save_graph(self, G, output_dir, p_change_feats=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(output_dir+"/graphsage")
            os.makedirs(output_dir+"/edgelist")
        with open(os.path.join(output_dir,"graphsage/G.json"), "w+") as file:
            res = json_graph.node_link_data(G)
            file.write(json.dumps(res))
        with open(os.path.join(output_dir,"graphsage/id2idx.json"), "w+") as file:
            file.write(json.dumps(self._create_id2idx(G)))
        features = self._build_features(p_change_feats)
        if features is not None:
            np.save(os.path.join(output_dir,"graphsage/feats.npy"), features)
        nx.write_edgelist(G, os.path.join(output_dir,"edgelist/edgelist"), delimiter=' ', data=False)
        print("Graph has been saved to ", self.output_dir1)

    def _create_id2idx(self, G):
        id2idx = {}
        for idx, node in enumerate(G.nodes()):
            id2idx[node] = idx
        return id2idx

    def _build_features(self, p_change_feats=None):
        features = None
        if os.path.isfile(self.networkx_dir+"/feats.npy"):
            features_ori = np.load(self.networkx_dir+"/feats.npy")
            if p_change_feats is not None:
                classes = np.unique(features_ori, axis=0)
                mask = np.random.uniform(size=(features_ori.shape[0]))
                mask = mask <= p_change_feats
                indexes_choice = np.random.choice(np.arange(classes.shape[0]),
                                                      size=(mask.sum()), replace=True)
                features_ori[mask] = classes[indexes_choice]
            features = features_ori
        return features
    
    def random_clone_synthetic(self, dataset, p_new_connection, p_remove_connection, seed):
        np.random.seed = seed
        H = dataset.G.copy()
        adj = dataset.get_adjacency_matrix()
        adj *= np.tri(*adj.shape)

        idx2id = {v: k for k,v in dataset.id2idx.items()}
        connected = np.argwhere(adj==1)

        mask_remove = np.random.uniform(0,1, size=(len(connected))) < p_remove_connection
        edges_remove = [(idx2id[x[0]], idx2id[x[1]]) for idx, x in enumerate(connected)
                        if mask_remove[idx] == True]
        count_rm = mask_remove.sum()
        H.remove_edges_from(edges_remove)

        print("New graph:")
        print("- Number of nodes:", len(H.nodes()))
        print("- Number of edges:", len(H.edges()))
        return H