"""
MLP模型与Focal Loss实现模块
功能：实现多层感知机模型和Focal Loss损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss实现
    专门用于处理类别不平衡问题的损失函数
    
    论文: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        参数:
            alpha (torch.Tensor, optional): 类别权重，用于处理类别不平衡
            gamma (float): 聚焦参数，控制难易样本的权重
            reduction (str): 损失聚合方式 ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算Focal Loss
        
        参数:
            inputs (torch.Tensor): 模型预测logits，形状为 (N, C)
            targets (torch.Tensor): 真实标签，形状为 (N,)
            
        返回:
            torch.Tensor: Focal Loss值
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 应用alpha权重
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            ce_loss = ce_loss * at
        
        # 计算Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MLPClassifier(nn.Module):
    """
    多层感知机分类器
    支持可配置的隐藏层结构和dropout
    """
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dims: list = [128, 64], 
                 dropout_rate: float = 0.3,
                 activation: str = 'relu'):
        """
        初始化MLP分类器
        
        参数:
            input_dim (int): 输入特征维度
            num_classes (int): 类别数量
            hidden_dims (list): 隐藏层维度列表
            dropout_rate (float): Dropout比例
            activation (str): 激活函数类型
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
        
        logger.info(f"创建MLP模型: 输入维度={input_dim}, "
                   f"隐藏层={hidden_dims}, 输出类别={num_classes}")
    
    def _initialize_weights(self):
        """
        权重初始化
        使用Xavier初始化方法
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征，形状为 (batch_size, input_dim)
            
        返回:
            torch.Tensor: 输出logits，形状为 (batch_size, num_classes)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            torch.Tensor: 预测概率
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            torch.Tensor: 预测类别
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


def calculate_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """
    计算类别权重用于Focal Loss
    
    参数:
        y_train (np.ndarray): 训练集标签
        
    返回:
        torch.Tensor: 类别权重张量
    """
    unique_classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    
    # 计算逆频率权重
    weights = total_samples / (len(unique_classes) * counts)
    
    # 归一化权重
    weights = weights / weights.sum() * len(unique_classes)
    
    logger.info(f"计算类别权重: {dict(zip(unique_classes, weights))}")
    
    return torch.FloatTensor(weights)