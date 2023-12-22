import numpy as np
import argparse
import time
from encoder.REGAL.xnetmf_config import *
import scipy.sparse as sps
from decoder.RefiNA.RefiNA import RefiNA
import decoder.refina_utils as refina_utils

from encoder.REGAL.REGAL import REGAL
from encoder.FINAL.FINAL import FINAL
from encoder.CONE.CONE import CONE
from encoder.Grampa.Grampa import Grampa
from encoder.IsoRank.IsoRank import IsoRank
from encoder.BigAlign.BigAlign import BigAlign
from encoder.NSD.NSD import NSD
from encoder.LREA.LREA import LREA
from encoder.Grasp.Grasp import Grasp
import math
from encoder.gwl import gwl_model
import torch.optim as optim
from torch.optim import lr_scheduler
from dataprocess.Dataset import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run CONE Align.")

    parser.add_argument('--true_align', nargs='?', default='data/synthetic-combined/arenas/arenas950-1/arenas_edges-mapping-permutation.txt',
                        help='True alignment file.')
    parser.add_argument('--combined_graph', nargs='?', default='data/synthetic-combined/arenas/arenas950-1/arenas_combined_edges.txt', help='Edgelist of combined input graph.')
    parser.add_argument("--level", default=3, type=int, help='Number of levels for coarseing')
    parser.add_argument('--output_alignment', nargs='?', default='output/alignment_matrix/arenas/arenas950-1', help='Output path for alignment matrix.')
    # Embedding Method
    parser.add_argument('--embmethod', nargs='?', default='netMF', help='Node embedding method.')
    # xnetmf parameters
    parser.add_argument('--attrvals', type=int, default=2,help='Number of attribute values. Only used if synthetic attributes are generated')

    # REFINA parameters
    parser.add_argument('--n-iter', type=int, default=100, help='Maximum #iter for RefiNA. Default is 20.') 
    parser.add_argument('--token-match', type=float, default = -1, help = "Token match score for each node.  Default of -1 sets it to reciprocal of largest graph #nodes rounded up to smallest power of 10")
    parser.add_argument('--n-update', type=int, default=-1, help='How many possible updates per node. Default is -1, or dense refinement.  Positive value uses sparse refinement')   
    # Alignment methods
    parser.add_argument('--alignmethod', nargs='?', default='REGAL', help='Network alignment method.')
    # Refinement methods
    parser.add_argument('--refinemethod', nargs='?', default=None, help='Network refinement method, to overcome the shortcoming of MILE')

    return parser.parse_args()


def main(args):

    dataset = Dataset(args.combined_graph, args.true_align)
    adjA, adjB = dataset.graph2adj()


    if (args.embmethod == "gwl"):
        # parse the data to be gwl readable format
        print("Parse the data to be gwl readable format")
        data_gwl = {}
        data_gwl['src_index'] = {}
        data_gwl['tar_index'] = {}
        data_gwl['src_interactions'] = []
        data_gwl['tar_interactions'] = []
        data_gwl['mutual_interactions'] = []
        for i in range(adjA.shape[0]):
            data_gwl['src_index'][float(i)] = i
        for i in range(adjB.shape[0]):
            data_gwl['tar_index'][float(i)] = i
        ma,mb = adjA.nonzero()
        for i in range(ma.shape[0]):
            data_gwl['src_interactions'].append([ma[i], mb[i]])
        ma,mb = adjB.nonzero()
        for i in range(ma.shape[0]):
            data_gwl['tar_interactions'].append([ma[i], mb[i]])
        after_emb = time.time()
    else:
        print("No preprocessing needed for FINAL")
        after_emb = time.time()

    ##################### Alignment ######################################
    before_align = time.time()
    # step2 and 3: align embedding spaces and match nodes with similar embeddings
    if args.alignmethod == 'REGAL':
        encoder = REGAL(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'FINAL':
        encoder = FINAL(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'IsoRank':
        encoder = IsoRank(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'BigAlign':
        encoder = BigAlign(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'CONE':
        encoder = CONE(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'Grampa':
        encoder = Grampa(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'NSD':
        encoder = NSD(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'LREA':
        encoder = LREA(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'Grasp':
        encoder = Grasp(adjA, adjB)
        alignment_matrix = encoder.align()
        
    elif args.alignmethod == "gwl":
        result_folder = 'gwl_test'
        cost_type = ['cosine']
        method = ['proximal']
        opt_dict = {'epochs': 30,
                    'batch_size': 57000,
                    'use_cuda': False,
                    'strategy': 'soft',
                    'beta': 1e-2,
                    'outer_iteration': 200,
                    'inner_iteration': 1,
                    'sgd_iteration': 500,
                    'prior': False,
                    'prefix': result_folder,
                    'display': False}
        for m in method:
            for c in cost_type:
                hyperpara_dict = {'src_number': len(data_gwl['src_index']),
                                'tar_number': len(data_gwl['tar_index']),
                                'dimension': 256,
                                'loss_type': 'L2',
                                'cost_type': c,
                                'ot_method': m}
                gwd_model = gwl_model.GromovWassersteinLearning(hyperpara_dict)

                # initialize optimizer
                optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=1e-2)
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

                # Gromov-Wasserstein learning
                gwd_model.train_without_prior(data_gwl, optimizer, opt_dict, scheduler=None)
                # save model
                gwd_model.save_model('{}/model_{}_{}.pt'.format(result_folder, m, c))
                gwd_model.save_recommend('{}/result_{}_{}.pkl'.format(result_folder, m, c))
                alignment_matrix = gwd_model.trans

    ##################### Refine Alignment embeddings ######################################
    if args.refinemethod is not None:
        if args.refinemethod == "RefiNA":
            if sps.issparse(alignment_matrix):
                alignment_matrix = np.array(alignment_matrix.todense())
            if args.n_update > 0:
                alignment_matrix = sps.csr_matrix(alignment_matrix)
                adjA = sps.csr_matrix(adjA)
                adjB = sps.csr_matrix(adjB)
                # alignment_matrix = refina.refina(alignment_matrix, adjA, adjB, true_alignments = true_align) 
                decoder = RefiNA(alignment_matrix, adjA, adjB, n_update=args.n_update,true_alignments = dataset.groundtruth)
                alignment_matrix = decoder.refine_align()  
                print(f"args.refinemethod is {args.refinemethod}")    
    node_num = alignment_matrix.shape[0]
    after_align = time.time()


    if dataset.groundtruth is not None:
        score, _ = refina_utils.score_alignment_matrix(alignment_matrix, topk = 1, true_alignments = dataset.groundtruth)
        mnc = refina_utils.score_MNC(alignment_matrix, adjA, adjB)
        print("Top 1 accuracy: %.5f" % score)
        print("MNC: %.5f" % mnc)

        # pred = matcher.greedy_match(alignment_matrix)
        # print(pred.shape)
        # print(len(true_align))

        # groundtruth_matrix = matcher.load_gt(true_align)
        # metrics.get_statistics(pred, groundtruth_matrix)

        # matcher.sinkhorn_match(alignment_matrix)
        # greedy_match_acc = metrics.get_statistics(pred, groundtruth_matrix)
        # print("Accuracy: %.4f" % greedy_match_acc)


    # evaluation
    # total_time = (after_align - before_align) + (after_emb - before_emb)
    # print(("score for NA: %f" % score))
    # print(("time (in seconds): %f" % total_time))


    # with open(args.output_stats, "w") as log:
    #     log.write("score: %f\n" % score)
    #     log.writelines("time(in seconds): %f\n"% total_time)

if __name__ == "__main__":
    args = parse_args()
    main(args)
