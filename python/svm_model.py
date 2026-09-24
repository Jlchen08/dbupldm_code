"""
SVM 基线模型训练与评估
功能：使用 scikit-learn 的 SVM 在与 MLP 相同的数据预处理基础上训练并评估
"""

import time
from typing import Dict

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    与 MLP 训练器保持一致的指标计算逻辑。
    """
    metrics: Dict[str, float] = {}

    # 基本分类指标
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # G-Mean
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:  # 二分类
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["g_mean"] = np.sqrt(sensitivity * specificity)
    else:  # 多分类
        recalls = []
        for i in range(cm.shape[0]):
            denom = cm[i, :].sum()
            recall_i = cm[i, i] / denom if denom > 0 else 0
            recalls.append(recall_i)
        # 防止全零导致 nan
        if len(recalls) > 0 and all(r > 0 for r in recalls):
            metrics["g_mean"] = float(np.power(np.prod(recalls), 1 / len(recalls)))
        else:
            metrics["g_mean"] = 0.0

    # AUC（多分类用 OvR）
    try:
        if len(np.unique(y_true)) == 2:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        metrics["auc_roc"] = 0.0

    return metrics


def train_and_evaluate_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    class_weight: str | dict | None = "balanced",
    probability: bool = True,
    max_iter: int = -1,
) -> Dict[str, float]:
    """
    训练并评估 SVM 模型。

    参数：
        X_train, y_train, X_test, y_test: 标准化后的特征与标签
        kernel, C, gamma, class_weight, probability, max_iter: SVC 参数

    返回：
        Dict[str, float]: 指标与时间信息
    """
    start = time.time()

    clf = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=probability,
        max_iter=max_iter,
        decision_function_shape="ovr",
        random_state=None,
    )
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)
    if probability:
        y_prob = clf.predict_proba(X_test)
    else:
        # 若未开启概率，退化为使用决策函数经归一化（避免 AUC 报错）
        dec = clf.decision_function(X_test)
        if dec.ndim == 1:
            # 二分类：转为两列概率形状
            dec = np.vstack([-dec, dec]).T
        # 归一化到 [0,1]
        dec = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)
        y_prob = dec

    training_time = time.time() - start

    metrics = _calculate_metrics(y_test, y_pred, y_prob)
    metrics["training_time"] = training_time
    # SVM 无 epoch 概念，保持字段一致
    metrics["epochs_trained"] = 1

    return metrics

