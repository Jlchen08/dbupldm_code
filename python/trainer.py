"""
训练器模块
功能：训练MLP模型并评估性能
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from typing import Dict, Tuple, List
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

class MLPTrainer:
    """
    MLP训练器类
    负责模型训练、验证和评估
    """
    
    def __init__(self, model, criterion, device='cpu'):
        """
        初始化训练器
        
        参数:
            model: MLP模型实例
            criterion: 损失函数
            device (str): 计算设备 ('cpu' 或 'cuda')
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.training_history = {'train_loss': [], 'train_acc': []}
        
    def train_epoch(self, train_loader: DataLoader, 
                   optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        训练一个epoch
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            optimizer (optim.Optimizer): 优化器
            
        返回:
            Tuple[float, float]: (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型性能
        
        参数:
            test_loader (DataLoader): 测试数据加载器
            
        返回:
            Dict[str, float]: 评估指标字典
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算评估指标
        metrics = self._calculate_metrics(
            np.array(all_targets), 
            np.array(all_predictions), 
            np.array(all_probabilities)
        )
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict[str, float]:
        """
        计算各种评估指标
        
        参数:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测标签
            y_prob (np.ndarray): 预测概率
            
        返回:
            Dict[str, float]: 评估指标字典
        """
        metrics = {}
        
        # 基本分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 计算G-mean (几何平均数)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape[0] == 2:  # 二分类
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['g_mean'] = np.sqrt(sensitivity * specificity)
        else:  # 多分类
            recalls = []
            for i in range(cm.shape[0]):
                recall_i = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
                recalls.append(recall_i)
            metrics['g_mean'] = np.power(np.prod(recalls), 1/len(recalls))
        
        # AUC (仅适用于二分类或使用OvR策略)
        try:
            if len(np.unique(y_true)) == 2:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"无法计算AUC: {str(e)}")
            metrics['auc_roc'] = 0.0
        
        return metrics
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader,
              epochs: int = 100, learning_rate: float = 0.001,
              patience: int = 10, min_delta: float = 0.001) -> Dict[str, List]:
        """
        完整训练流程
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            test_loader (DataLoader): 测试数据加载器
            epochs (int): 训练轮数
            learning_rate (float): 学习率
            patience (int): 早停耐心值
            min_delta (float): 早停最小改善阈值
            
        返回:
            Dict[str, List]: 训练历史记录
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_f1 = 0.0
        patience_counter = 0
        start_time = time.time()
        
        logger.info(f"开始训练，总轮数: {epochs}")
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # 评估模型
            test_metrics = self.evaluate(test_loader)
            
            # 更新学习率
            scheduler.step(train_loss)
            
            # 记录训练历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            
            # 早停检查
            current_f1 = test_metrics['f1_score']
            if current_f1 > best_f1 + min_delta:
                best_f1 = current_f1
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"Train Loss: {train_loss:.4f}, "
                           f"Train Acc: {train_acc:.4f}, "
                           f"Test F1: {current_f1:.4f}")
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 恢复最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        training_time = time.time() - start_time
        logger.info(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 最终评估
        final_metrics = self.evaluate(test_loader)
        final_metrics['training_time'] = training_time
        final_metrics['epochs_trained'] = epoch + 1
        
        return final_metrics


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    创建数据加载器
    
    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        batch_size (int): 批次大小
        
    返回:
        Tuple[DataLoader, DataLoader]: 训练和测试数据加载器
    """
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader