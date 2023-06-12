import numpy as np
import random
import torch.nn.functional as F
import torch
import argparse
import json

def parse_args():    
    parser = argparse.ArgumentParser(description="Evaluation embedding based on link prediction")
    parser.add_argument('--embedding_path', default=None)
    parser.add_argument('--edgelist_path', default=None)    
    parser.add_argument('--idmap_path', default=None)    
    parser.add_argument('--file_format', default="word2vec", help="File format, choose word2vec or numpy")    
    return parser.parse_args()

def square_distance(matrix1, matrix2):
    if len(matrix1.shape) == 1:
        return np.sum((matrix1 - matrix2)**2)
    return np.sum((matrix1 - matrix2)**2, dim=1)

def cosine_distance(matrix1, matrix2):
    matrix1 = torch.FloatTensor(matrix1)
    matrix2 = torch.FloatTensor(matrix2)
    distance = (1.0 - F.cosine_similarity(matrix1, matrix2)) / 2
    return distance

def statistic_on_distance(embedding, edgelist):
    source_node = edgelist[:,0]
    target_node = edgelist[:,1]

    source_embedding = embedding[source_node]
    target_embedding = embedding[target_node]
    distance = torch.mean(cosine_distance(source_embedding, target_embedding))
    return distance.item()

def link_prediction(embedding, threshold_distance, edgelist):
    new_edge_list = []
    new_edge_list = [set(ele) for ele in edgelist if set(ele) not in new_edge_list]

    embedding = torch.FloatTensor(embedding)
    edges = []

    for node in range(min(len(embedding), 50)):
        print("Start finding neighbors for node: ", node)
        embedding_of_node = embedding[node]
        cos_distances = cosine_distance(torch.stack([embedding_of_node]*len(embedding)), embedding)
        for i, ele in enumerate(cos_distances):
            if ele < threshold_distance:
                if set([node, i]) not in edges:
                    edges.append(set([node, i]))
        
    print("start evaluate")
    count_true = 0
    for edge in edges:
        if edge in new_edge_list:
            count_true += 1
    precision = count_true/len(edges)
    recall = count_true/len(new_edge_list)

    return precision, recall

def load_embedding(embed_file, id_map, file_format):
    if file_format == "word2vec":
        try:
            with open(embed_file) as fp:
                descriptions = fp.readline().split()
                if len(descriptions) != 2:
                    raise Exception("Wrong format")
                num_nodes = int(descriptions[0])
                dim = int(descriptions[1])
                embeddings = np.zeros((num_nodes, dim))
                for line in fp:
                    tokens = line.split()
                    if len(tokens) != dim + 1:
                        raise Exception("Wrong format")
                    feature = np.zeros(dim)
                    for i in range(dim):
                        feature[i] = float(tokens[i+1])
                    embeddings[id_map[tokens[0]]] = feature
                fp.close()
        except Exception as e:
            print(e)
            print("The format might be wrong, consider trying --file_format flag with 'numpy' value")
            embeddings = None

    else:
        embeddings = np.load(embed_file)
        
    return embeddings

def load_edgelist(edgelist_file, id_map, file_format):
    if file_format == "word2vec":
        all_instances = []    
        with open(edgelist_file) as fp:
            for line in fp:
                ins = line.split()
                all_instances.append([id_map[ins[0]], id_map[ins[1]]])
        
        return np.array(all_instances)
    return np.load(edgelist_file)

if __name__ == '__main__':
    args = parse_args()
    print(args)

    id_map = json.loads(open(args.idmap_path, "r").read())
    embedding = load_embedding(args.embedding_path, id_map, args.file_format)
    edgelist = load_edgelist(args.edgelist_path, id_map, args.file_format)       
    if embedding is not None:
        cosine_similarity = statistic_on_distance(embedding, edgelist)
        precision, recall = link_prediction(embedding, cosine_similarity, edgelist)
        print("Mean distance between node: ", cosine_similarity)
        print("precision: ", precision)
        print('recall: ', recall)
