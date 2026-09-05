import torch
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from data.dataset import (
    test_loader,
    data
)

from models.model import (
    model,
    device
)

from utils.metrics import (
    print_classification_report,
    plot_confusion_matrix
)


# =========================
# 加载最佳模型
# =========================

model.load_state_dict(
    torch.load(
        "baseline_best.pth",
        map_location=device
    )
)

model.eval()


# =========================
# 测试集预测
# =========================

all_preds = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        preds = outputs.argmax(
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# =========================
# Top-1 Accuracy
# =========================

top1_acc = accuracy_score(
    all_labels,
    all_preds
)


# =========================
# Macro-F1
# =========================

macro_f1 = f1_score(
    all_labels,
    all_preds,
    average="macro"
)


print("Baseline测试集结果")

print(
    "Top-1 Accuracy：",
    top1_acc
)

print(
    "Macro-F1：",
    macro_f1
)


# =========================
# Classification Report
# =========================

print("Baseline分类报告")

print_classification_report(
    all_labels,
    all_preds,
    data.classes
)


# =========================
# Confusion Matrix
# =========================

plot_confusion_matrix(
    all_labels,
    all_preds,
    data.classes,
    title="Confusion Matrix - Baseline"
)
