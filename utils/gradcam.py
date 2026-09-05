import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def calculate_top1(preds, labels):
    return accuracy_score(labels, preds)


def calculate_top5(outputs, labels):

    top5 = outputs.topk(5, dim=1).indices

    correct = top5.eq(labels.view(-1, 1))

    return correct.any(dim=1).float().mean().item()


def calculate_macro_f1(preds, labels):

    return f1_score(
        labels,
        preds,
        average="macro"
    )


def get_classification_report(
    labels,
    preds,
    class_names
):

    return classification_report(
        labels,
        preds,
        target_names=class_names,
        digits=4
    )


def get_confusion_matrix(labels, preds):

    return confusion_matrix(
        labels,
        preds
    )
