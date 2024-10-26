# metrics/metrics.py

import torch


def dice_coeff(pred, target, smooth=1e-7):
    """Dice係数を計算する"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (
        pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    )
    return dice.mean()


def precision_and_recall(pred, target, smooth=1e-7):
    """PrecisionとRecallを計算する"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    true_positive = (pred * target).sum(dim=(2, 3))
    false_positive = (pred * (1 - target)).sum(dim=(2, 3))
    false_negative = ((1 - pred) * target).sum(dim=(2, 3))

    precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
    recall = (true_positive + smooth) / (true_positive + false_negative + smooth)

    return precision.mean(), recall.mean()


def accuracy(pred, target):
    """Accuracy（精度）を計算する"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    correct = (pred == target).float().sum(dim=(2, 3))
    total = target.size(2) * target.size(3)

    accuracy = correct / total
    return accuracy.mean()
