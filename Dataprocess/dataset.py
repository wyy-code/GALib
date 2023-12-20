import json
import os
import argparse
from scipy.io import loadmat
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from Dataprocess.data_preprocess import DataPreprocess

import utils.graph_utils as graph_utils

class Dataset:
    """
    this class receives input from graphsage format with predefined folder structure, the data folder must contains these files:
    G.json, id2idx.json, features.npy (optional)

    Arguments:
    - data_dir: Data directory which contains files mentioned above.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._load_G()
        self._load_id2idx()
        #if self.check_id2idx() is False:
        #   raise Exception("Error in id2idx") 
        self._load_features()
        self.load_edge_features()
        print("Dataset info:")
        print("- Nodes: ", len(self.G.nodes()))
        print("- Edges: ", len(self.G.edges()))

    def _load_G(self):
        G_data = json.load(open(os.path.join(self.data_dir, "G.json")))
        self.G = json_graph.node_link_graph(G_data)
        if type(self.G.nodes()[0]) is int:
            mapping = {k: str(k) for k in self.G.nodes()}
            self.G = nx.relabel_nodes(self.G, mapping)

    def _load_id2idx(self):
        id2idx_file = os.path.join(self.data_dir, 'id2idx.json')
        conversion = type(self.G.nodes()[0])
        self.id2idx = {}
        id2idx = json.load(open(id2idx_file))
        # print(conversion)
        # print("Test Finished")
        # exit()
        for k, v in id2idx.items():
            # self.id2idx[conversion(k)] = v
            self.id2idx[k] = v
        # for k, v in id2idx.items():
        #     try:
        #         self.id2idx[conversion(k)] = v
        #     except ValueError:
        #         print(f"{k} and {v}")

    def _load_features(self):
        self.features = None
        feats_path = os.path.join(self.data_dir, 'feats.npy')
        if os.path.isfile(feats_path):
            self.features = np.load(feats_path)
        else:
            self.features = None
        return self.features

    def load_edge_features(self):
        self.edge_features= None
        feats_path = os.path.join(self.data_dir, 'edge_feats.mat')
        if os.path.isfile(feats_path):
            edge_feats = loadmat(feats_path)['edge_feats']
            self.edge_features = np.zeros((len(edge_feats[0]),
                                           len(self.G.nodes()),
                                           len(self.G.nodes())))
                                        #    int(len(self.G.nodes())/2),
                                        #    int(len(self.G.nodes())/2)))
            print(self.edge_features.shape)
            for idx, matrix in enumerate(edge_feats[0]):
                print(matrix.shape)
                self.edge_features[idx] = matrix.toarray()
                # print("--------")
        else:
            self.edge_features = None
        return self.edge_features

    def get_adjacency_matrix(self):
        return graph_utils.construct_adjacency(self.G, self.id2idx)

    def get_nodes_degrees(self):
        return graph_utils.build_degrees(self.G, self.id2idx)

    def get_nodes_clustering(self):
        return graph_utils.build_clustering(self.G, self.id2idx)

    def get_edges(self):
        return graph_utils.get_edges(self.G, self.id2idx)

    def check_id2idx(self):
        # print("Checking format of dataset")
        for i, node in enumerate(self.G.nodes()):
            if (self.id2idx[node] != i):
                print("Failed at node %s" % str(node))
                return False
        # print("Pass")
        return True
