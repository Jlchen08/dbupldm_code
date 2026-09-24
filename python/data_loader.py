"""
数据集加载器模块
功能：加载和预处理MATLAB格式的数据集
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    数据集加载器类
    负责加载MATLAB格式数据集并进行预处理
    """
    
    def __init__(self, dataset_path: str):
        """
        初始化数据集加载器
        
        参数:
            dataset_path (str): 数据集文件夹路径
        """
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_mat_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载单个MATLAB文件
        
        参数:
            file_path (str): .mat文件路径
            
        返回:
            Tuple[np.ndarray, np.ndarray]: 特征矩阵和标签向量
            
        异常:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
        """
        try:
            # 加载.mat文件
            mat_data = loadmat(file_path)
            
            # 获取数据变量（排除MATLAB元数据）
            data_keys = [key for key in mat_data.keys() 
                        if not key.startswith('__')]
            
            if len(data_keys) != 1:
                raise ValueError(f"期望找到1个数据变量，实际找到{len(data_keys)}个")
                
            data = mat_data[data_keys[0]]
            
            # 分离特征和标签（假设最后一列是标签）
            X = data[:, :-1].astype(np.float32)
            y = data[:, -1].astype(np.int32)
            
            logger.info(f"成功加载 {os.path.basename(file_path)}: "
                       f"样本数={X.shape[0]}, 特征数={X.shape[1]}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            raise
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray, 
                       test_size: float = 0.3, 
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray]:
        """
        数据预处理：标准化和划分训练测试集
        
        参数:
            X (np.ndarray): 特征矩阵
            y (np.ndarray): 标签向量
            test_size (float): 测试集比例
            random_state (int): 随机种子
            
        返回:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            # 标签编码（确保标签从0开始）
            y_encoded = self.label_encoder.fit_transform(y)
            
            # 划分训练测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, 
                random_state=random_state, stratify=y_encoded
            )
            
            # 特征标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 计算类别分布
            unique, counts = np.unique(y_train, return_counts=True)
            class_distribution = dict(zip(unique, counts))
            
            logger.info(f"训练集类别分布: {class_distribution}")
            logger.info(f"不平衡比例: {max(counts)/min(counts):.2f}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise
    
    def get_all_datasets(self) -> List[str]:
        """
        获取所有数据集文件名
        
        返回:
            List[str]: 数据集文件名列表
        """
        try:
            mat_files = [f for f in os.listdir(self.dataset_path) 
                        if f.endswith('.mat')]
            mat_files.sort()
            logger.info(f"找到 {len(mat_files)} 个数据集文件")
            return mat_files
            
        except Exception as e:
            logger.error(f"获取数据集列表失败: {str(e)}")
            raise
    
    def load_dataset_by_name(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray]:
        """
        根据数据集名称加载并预处理数据
        
        参数:
            dataset_name (str): 数据集文件名
            
        返回:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        file_path = os.path.join(self.dataset_path, dataset_name)
        X, y = self.load_mat_file(file_path)
        return self.preprocess_data(X, y)