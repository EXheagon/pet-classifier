# 基于深度学习的牛津宠物细粒度分类

## 项目简介
本项目基于深度学习方法，实现牛津宠物数据集的细粒度图像分类任务。

## 实验环境
- 模型：ResNet-18
- 优化器：AdamW
- 学习率：1e-4
- Batch Size：32
- Epoch：15
- 训练平台：T4 GPU

## 数据集
Oxford-IIIT Pet Dataset

- 37个类别
- 约7300张图片

## 实验方法
- Baseline：ResNet-18
- 数据增强：MixUp
- 正则化方法：Label Smoothing

## 文件说明

pet_classifier_training.ipynb

包含完整训练代码和实验过程。
