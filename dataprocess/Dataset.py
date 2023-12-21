import networkx as nx
import numpy as np
import pickle


class Dataset():
    """
    This class provided these methods:
    - edgelist2networkx: convert an edgelist file to graph in networkx format.
        If features file is present, it must be in dict format, with keys are nodes' id and values are features in list format.
    - networkx2edgelist: convert an graph of networkx format to edgelist file,
        save features if present in networkx.
    """

    def __init__(self, graph_file, groundtruth_file):
        self.graph = nx.read_edgelist(graph_file, nodetype=int, comments="%")
        with open(groundtruth_file, "rb") as true_alignments_file:
        # for python3, you need to use latin1 as the encoding method
            self.groundtruth = pickle.load(true_alignments_file, encoding = "latin1")


    def graph2adj(self):

        ##################### Load data ######################################
        # running normal graph alignment methods
        adj = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()) ).todense().astype(float)
        node_num = int(adj.shape[0] / 2)
        adjA = np.array(adj[:node_num, :node_num])
        adjB = np.array(adj[node_num:, node_num:])

        # print statistics data
        print("---------------")
        print(f"The number of nodes in a single graph is {node_num}")
        print(f"The number of edges in a the graph A is {nx.from_numpy_matrix(adjA).number_of_edges()}")
        print(f"The number of edges in a the graph B is {nx.from_numpy_matrix(adjB).number_of_edges()}")
        print("---------------")

        return adjA, adjB