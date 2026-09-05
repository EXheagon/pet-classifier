import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import (
    train_loader,
    val_loader
)

from models.model import (
    model,
    device,
    model_improved
)


# =========================
# Baseline 损失函数和优化器
# =========================

criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# =========================
# 训练函数
# =========================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() *
            images.size(0)
        )

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += (
            predicted.eq(labels)
        ).sum().item()

    avg_loss = total_loss / total

    accuracy = correct / total

    return avg_loss, accuracy


# =========================
# 验证函数
# =========================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += (
                predicted.eq(labels)
            ).sum().item()

    avg_loss = total_loss / total

    accuracy = correct / total

    return avg_loss, accuracy


# =========================
# Baseline训练
# =========================

num_epochs = 10

train_losses = []
train_accs = []

val_losses = []
val_accs = []

best_val_acc = 0.0


for epoch in range(num_epochs):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "baseline_best.pth"
        )


print("Baseline训练完成！")
print(
    "最佳验证集准确率：",
    best_val_acc
)
