from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score, auc, roc_curve
import numpy as np
from matcher.matcher import top_k, greedy_match, sinkhorn_match

#Metrics using for benchmark
# greedy match
# - accuracy
# top-k
# - f1_score
# - map (average_precision_score)
# - auc
# - roc

import pdb
def get_statistics(alignment_matrix, groundtruth_matrix):
    pred = greedy_match(alignment_matrix)
    greedy_match_acc = compute_accuracy(pred, groundtruth_matrix)
    # print("Accuracy: %.4f" % greedy_match_acc)
    print("Precision: %.4f" % greedy_match_acc)

    # F1 = f1_score(pred, groundtruth_matrix, labels=[0, 1], pos_label=1, average='micro')
    # MAP, AUC, Hit = compute_MAP_AUC_Hit(alignment_matrix, groundtruth_matrix)

    # print("MAP: %.4f" % MAP)
    # print("AUC: %.4f" % AUC)
    # print("Hit-precision: %.4f" % Hit)

    # pred_top_1 = top_k(alignment_matrix, 1)
    # precision_1 = compute_precision_k(pred_top_1, groundtruth_matrix)
    # print("Precision_1: %.4f" % precision_1)
    # pred_top_5 = top_k(alignment_matrix, 5)
    # precision_5 = compute_precision_k(pred_top_5, groundtruth_matrix)
    # print("Precision_5: %.4f" % precision_5)
    # pred_top_10 = top_k(alignment_matrix, 10)
    # precision_10 = compute_precision_k(pred_top_10, groundtruth_matrix)
    # print("Precision_10: %.4f" % precision_10)

    # sin_pred = sinkhorn_match(alignment_matrix)
    # sin_match_acc = compute_accuracy(sin_pred, groundtruth_matrix)
    # print("Sinkhorn Precision: %.4f" % sin_match_acc)

    # return greedy_match_acc, MAP, AUC, Hit, precision_5, precision_10

# def compute_accuracy(matched, groundtruth):
#     n_matched = 0
#     total_nodes = len(groundtruth.items())
#     not_matched = 0
#     for src, trg in groundtruth.items():
#         try:
#             if trg in matched[src]:
#                 n_matched+=1
#         except:
#             not_matched += 1
#     return n_matched*100/total_nodes

def compute_precision_k(top_k_matrix, gt):
    n_matched = 0
    gt_candidates = np.argmax(gt, axis = 1)
    for i in range(gt.shape[0]):
        if gt[i][gt_candidates[i]] == 1 and top_k_matrix[i][gt_candidates[i]] == 1:
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes

def compute_accuracy(greedy_matched, gt):
    # print(gt)
    n_matched = 0
    for i in range(greedy_matched.shape[0]):
        if greedy_matched[i].sum() > 0 and np.array_equal(greedy_matched[i], gt[i]):
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes

def compute_MAP_AUC_Hit(alignment_matrix, gt):
    S_argsort = alignment_matrix.argsort(axis=1)[:, ::-1]
    m = gt.shape[1] - 1
    MAP = 0
    AUC = 0
    Hit = 0
    for i in range(len(S_argsort)):
        predicted_source_to_target = S_argsort[i]
        # true_source_to_target = gt[i]
        for j in range(gt.shape[1]):
            if gt[i, j] == 1:
                for k in range(len(predicted_source_to_target)):
                    if predicted_source_to_target[k] == j:
                        ra = k + 1
                        MAP += 1/ra
                        AUC += (m+1-ra)/m
                        Hit += (m+2-ra)/(m+1)
                        break
                break
    n_nodes = (gt==1).sum()
    MAP /= n_nodes
    AUC /= n_nodes
    Hit /= n_nodes
    return MAP, AUC, Hit

# def compute_AUC(alignment_matrix, gt):
#     S_argsort = alignment_matrix.argsort(axis=1)[:, ::-1]
#     m = len(gt) - 1
#     AUC = 0
#     for i in range(len(S_argsort)):
#         predicted_source_to_target = S_argsort[i]
#         true_source_to_target = gt[i]
#         for j in range(gt.shape[1]):
#             if gt[i, j] == 1:
#                 AUC += (m + 1 - predicted_source_to_target[j])/(m)
#                 break
#     AUC /= min(gt.shape)
#     return AUC

# def compute_Hit_Precision(alignment_matrix, gt):
#     S_argsort = alignment_matrix.argsort(axis=1)[:, ::-1]
#     S_argsort += 1
#     m = len(gt) - 1
#     Hit = 0
#     for i in range(len(S_argsort)):
#         predicted_source_to_target = S_argsort[i]
#         true_source_to_target = gt[i]
#         for j in range(gt.shape[1]):
#             if gt[i, j] == 1:
#                 Hit += (m + 2 - predicted_source_to_target[j])/(m+1)
#                 break
#     Hit /= min(gt.shape)
#     return Hit

if __name__ == '__main__':
    y_true = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
    y_pred = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
    # print(precision_score(y_true, y_pred, average='micro'))
    # print(recall_score(y_true, y_pred, average='micro'))
    print(compute_accuracy(y_true, y_pred))
    # print(f1_score(y_true, y_pred, average='micro'))










