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

    for Ptarget in [0.01, 0.001]:
        Cdet = Cmiss * FNR * Ptarget + Cfa * FPR * (1 - Ptarget)
        Cdef = min(Cmiss * Ptarget, Cfa * (1 - Ptarget))
        minDCF = min(Cdet) / Cdef
        avg += minDCF

    avg = avg / 2

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
    # print("{:.3f} {:.3f}".format(eer, minDCF))
    print("{:.3f}".format(minDCF))






