#!/data/lijianchen/miniconda3/envs/pytorch/bin/python
# -*- coding: utf-8 -*-
"""
    Define metric function (minDCF, EER)
"""
import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
import argparse


def compute_eer(scores, labels, eps=1e-6, showfig=False):
    """
    If score > threshold, prediction is positive, else negative

    :param scores: similarity for target and non-target trials
    :param labels: true labels for target and non-target trials
    :param showfig: if true, the DET curve is displayed
    :return:
            eer: percent equal eeror rate (EER)
            dcf: minimum detection cost function (DCF) with voxceleb parameters

    :Reference: Microsoft MSRTookit: compute_eer.m
            数据挖掘导论 p183
    :Author: li-jianchen(matln)
    """

    # Get the index list after sorting the scores list
    sorted_index = [index for index, value in sorted(
        enumerate(scores), key=lambda x: x[1])]
    # Sort the labels list
    sorted_labels = [labels[i] for i in sorted_index]
    sorted_labels = np.array(sorted_labels)

    FN = np.cumsum(sorted_labels == 1) / (sum(sorted_labels == 1) + eps)
    TN = np.cumsum(sorted_labels == 0) / (sum(sorted_labels == 0) + eps)
    FP = 1 - TN
    TP = 1 - FN

    FNR = FN / (TP + FN + eps)
    FPR = FP / (TN + FP + eps)
    difs = FNR - FPR
    idx1 = np.where(difs < 0, difs, float('-inf')).argmax(axis=0)
    idx2 = np.where(difs >= 0, difs, float('inf')).argmin(axis=0)
    # the x-axis of two points
    x = [FPR[idx1], FPR[idx2]]
    # the y-axis of two points
    y = [FNR[idx1], FNR[idx2]]
    # compute the intersection of the straight line connecting (x1, y1), (x2, y2)
    # and y = x.
    # Derivation: (x-x1) / (x2-x1) = (x-y1) / (y2-y1)                 ->
    #             (x-x1)(y2-y1) = (x-y1)(x2-x1)                       ->
    #             x(y2-y1-x2-x1) = x1(y2-y1) - y1(x2-x1)              ->
    #                            = x1(x2-x1) - y1(x2-x1)
    #                              + x1(y2-y1) - x1(x2-x1)            ->
    #                            = (x1-x2)(x2-x1) + x1(y2-y1-x2+x1)   ->
    #             x = x1 + (x1-x2)(x2-x1) / (y2-y1-x2-x1)
    a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])
    eer = 100 * (x[0] + a * (y[0] - x[0]))

    # Compute dcf
    # VoxCeleb performance parameter
    Cmiss = 1
    Cfa = 1
    avg = 0

    # for Ptarget in [0.01, 0.001]:
    for Ptarget in [0.01]:
        Cdet = Cmiss * FNR * Ptarget + Cfa * FPR * (1 - Ptarget)
        Cdef = min(Cmiss * Ptarget, Cfa * (1 - Ptarget))
        minDCF = min(Cdet) / Cdef
        avg += minDCF

    # avg = avg / 2

    # figure
    if showfig:
        plot_det(FPR, FNR)

    return eer, avg


def plot_det(FPR, FNR):
    # Plots the detection error tradeoff (DET) curve
    # Reference: compute_eer.m
    pass


def compute_eer_sklearn(y_score, y, pos=1):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    # pos: 1 if higher is positive; 0 is lower is positive

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100, thresh

# --------------------------------------------------------------------------------------
# ## 给定一些类数据，计算等错误率，
# def compute_eer(fnr, fpr):
#     """ computes the equal error rate (EER) given FNR and FPR values calculated
#         for a range of operating points on the DET curve
#     """
# 
#     diff_pm_fa = fnr - fpr
#     x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
#     x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
#     a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
#     return fnr[x1] + a * (fnr[x2] - fnr[x1])
# 
# 
# def compute_pmiss_pfa(scores, labels):
#     """ computes false positive rate (FPR) and false negative rate (FNR)
#     given trial scores and their labels. A weights option is also provided
#     to equalize the counts over score partitions (if there is such
#     partitioning).
#     """
# 
#     sorted_ndx = np.argsort(scores)
#     labels = np.array(labels)[sorted_ndx]
# 
#     tgt = (labels == 1).astype('f8')
#     imp = (labels == 0).astype('f8')
# 
#     fnr = np.cumsum(tgt) / np.sum(tgt)
#     fpr = 1 - np.cumsum(imp) / np.sum(imp)
#     return fnr, fpr
# 
# 
# def compute_min_cost(scores, labels, p_target=0.01):
#     fnr, fpr = compute_pmiss_pfa(scores, labels)
#     eer = compute_eer(fnr, fpr)
#     min_c = compute_c_norm(fnr, fpr, p_target)
#     return eer, min_c
# 
# 
# def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
#     """ computes normalized minimum detection cost function (DCF) given
#         the costs for false accepts and false rejects as well as a priori
#         probability for target speakers
#     """
#     dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
#     c_det = np.min(dcf)
#     c_def = min(c_miss * p_target, c_fa * (1 - p_target))
#     return c_det/c_def
# --------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="",
        formatter_class=argparse.RawTextHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument("--scores", type=str, default="", help="")
    parser.add_argument("--trials", type=str, default="", help="")
    args = parser.parse_args()

    scores = []
    with open(args.scores, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(" ")
            scores.append(float(line[2]))

    labels = []
    with open(args.trials, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(" ")
            if line[2] == 'target':
                labels.append(1)
            elif line[2] == 'nontarget':
                labels.append(0)


    eer, minDCF = compute_eer(scores, labels)
    # eer, minDCF = compute_min_cost(scores, labels)
    # eer, _ = compute_eer_sklearn(scores, labels)
    print("{:.3f} {:.3f}".format(eer, minDCF))
    # print("{:.3f}".format(minDCF))
    # print("{:.2f}".format(eer))






