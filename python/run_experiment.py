"""
主实验脚本
功能：运行完整的MLP+Focal Loss实验并生成结果报告
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_loader import DatasetLoader
from mlp_focal_model import MLPClassifier, FocalLoss, calculate_class_weights
from svm_model import train_and_evaluate_svm
from trainer import MLPTrainer, create_data_loaders

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    实验运行器类
    负责协调整个实验流程
    """
    
    def __init__(self, dataset_path: str, results_dir: str = 'results'):
        """
        初始化实验运行器
        
        参数:
            dataset_path (str): 数据集路径
            results_dir (str): 结果保存目录
        """
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 初始化数据加载器
        self.data_loader = DatasetLoader(dataset_path)
        
        logger.info(f"实验运行器初始化完成，使用设备: {self.device}")
    
    def run_single_experiment(self, dataset_name: str, 
                            config: Dict = None) -> Dict[str, float]:
        """
        运行单个数据集的实验
        
        参数:
            dataset_name (str): 数据集名称
            config (Dict): 实验配置参数
            
        返回:
            Dict[str, float]: 实验结果
        """
        logger.info(f"开始实验: {dataset_name}")
        
        # 默认配置
        default_config = {
            'hidden_dims': [128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'gamma': 2.0,  # Focal Loss参数
            'patience': 15,
            'random_state': 42
        }
        
        if config:
            default_config.update(config)
        config = default_config
        
        try:
            # 加载和预处理数据
            X_train, X_test, y_train, y_test = self.data_loader.load_dataset_by_name(dataset_name)
            
            # 获取数据集信息
            n_features = X_train.shape[1]
            n_classes = len(np.unique(y_train))
            n_samples = len(X_train) + len(X_test)
            
            # 计算类别不平衡比例
            unique, counts = np.unique(y_train, return_counts=True)
            imbalance_ratio = max(counts) / min(counts)
            
            logger.info(f"{dataset_name}: 样本数={n_samples}, 特征数={n_features}, "
                       f"类别数={n_classes}, 不平衡比例={imbalance_ratio:.2f}")
            
            # 创建数据加载器
            train_loader, test_loader = create_data_loaders(
                X_train, y_train, X_test, y_test, 
                batch_size=config['batch_size']
            )
            
            # 计算类别权重
            class_weights = calculate_class_weights(y_train)
            
            # 创建模型
            model = MLPClassifier(
                input_dim=n_features,
                num_classes=n_classes,
                hidden_dims=config['hidden_dims'],
                dropout_rate=config['dropout_rate']
            )
            
            # 创建Focal Loss
            criterion = FocalLoss(
                alpha=class_weights,
                gamma=config['gamma']
            )
            
            # 创建训练器
            trainer = MLPTrainer(model, criterion, device=self.device)
            
            # 训练模型
            results = trainer.train(
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=config['epochs'],
                learning_rate=config['learning_rate'],
                patience=config['patience']
            )
            
            # 添加数据集信息到结果中
            results.update({
                'dataset_name': dataset_name,
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': n_classes,
                'imbalance_ratio': imbalance_ratio,
                'config': config
            })
            
            logger.info(f"{dataset_name} 实验完成: F1={results['f1_score']:.4f}, "
                       f"AUC={results['auc_roc']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"{dataset_name} 实验失败: {str(e)}")
            return {
                'dataset_name': dataset_name,
                'error': str(e),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'g_mean': 0.0,
                'auc_roc': 0.0
            }

    def run_single_experiment_svm(self, dataset_name: str, config: Dict | None = None) -> Dict[str, float]:
        """
        使用 SVM 运行单个数据集实验。
        """
        logger.info(f"开始 SVM 实验: {dataset_name}")

        default_config = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True,
            'max_iter': -1,
            'random_state': 42,
        }
        if config:
            default_config.update(config)
        config = default_config

        try:
            X_train, X_test, y_train, y_test = self.data_loader.load_dataset_by_name(dataset_name)

            n_features = X_train.shape[1]
            n_classes = len(np.unique(y_train))
            n_samples = len(X_train) + len(X_test)

            unique, counts = np.unique(y_train, return_counts=True)
            imbalance_ratio = max(counts) / min(counts)

            logger.info(f"{dataset_name} [SVM]: 样本数={n_samples}, 特征数={n_features}, "
                        f"类别数={n_classes}, 不平衡比={imbalance_ratio:.2f}")

            metrics = train_and_evaluate_svm(
                X_train, y_train, X_test, y_test,
                kernel=config['kernel'],
                C=config['C'],
                gamma=config['gamma'],
                class_weight=config['class_weight'],
                probability=config['probability'],
                max_iter=config['max_iter'],
            )

            metrics.update({
                'dataset_name': dataset_name,
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': n_classes,
                'imbalance_ratio': imbalance_ratio,
                'config': config
            })

            logger.info(f"{dataset_name} SVM 实验完成: F1={metrics['f1_score']:.4f}, "
                        f"AUC={metrics['auc_roc']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"{dataset_name} SVM 实验失败: {str(e)}")
            return {
                'dataset_name': dataset_name,
                'error': str(e),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'g_mean': 0.0,
                'auc_roc': 0.0
            }
    
    def run_all_experiments(self, config: Dict = None) -> List[Dict]:
        """
        运行所有数据集的实验
        
        参数:
            config (Dict): 实验配置参数
            
        返回:
            List[Dict]: 所有实验结果
        """
        # 获取所有数据集
        datasets = self.data_loader.get_all_datasets()
        
        logger.info(f"开始运行 {len(datasets)} 个数据集的实验")
        
        all_results = []
        
        for i, dataset_name in enumerate(datasets, 1):
            logger.info(f"进度: {i}/{len(datasets)} - {dataset_name}")
            
            result = self.run_single_experiment(dataset_name, config)
            all_results.append(result)
            
            # 保存中间结果
            self.save_results(all_results, f'intermediate_results_{i}.json')
        
        logger.info("所有实验完成")
        return all_results

    def run_all_experiments_svm(self, config: Dict | None = None) -> List[Dict]:
        """
        运行所有数据集的 SVM 实验。
        """
        datasets = self.data_loader.get_all_datasets()

        logger.info(f"开始运行 {len(datasets)} 个数据集的 SVM 实验")

        all_results: List[Dict] = []
        for i, dataset_name in enumerate(datasets, 1):
            logger.info(f"[SVM] 进度: {i}/{len(datasets)} - {dataset_name}")

            result = self.run_single_experiment_svm(dataset_name, config)
            all_results.append(result)

            self.save_results(all_results, f'svm_intermediate_results_{i}.json')

        logger.info("所有 SVM 实验完成")
        return all_results
    
    def save_results(self, results: List[Dict], filename: str = None):
        """
        保存实验结果
        
        参数:
            results (List[Dict]): 实验结果列表
            filename (str): 保存文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'mlp_focal_results_{timestamp}.json'
        
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"结果已保存到: {filepath}")
    
    def generate_summary_table(self, results: List[Dict]) -> pd.DataFrame:
        """
        生成结果汇总表
        
        参数:
            results (List[Dict]): 实验结果列表
            
        返回:
            pd.DataFrame: 汇总表
        """
        # 提取关键指标
        summary_data = []
        
        for result in results:
            if 'error' not in result:
                summary_data.append({
                    'Dataset': result['dataset_name'].replace('.mat', ''),
                    'Samples': result['n_samples'],
                    'Features': result['n_features'],
                    'Classes': result['n_classes'],
                    'IR': f"{result['imbalance_ratio']:.2f}",
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'Precision': f"{result['precision']:.4f}",
                    'Recall': f"{result['recall']:.4f}",
                    'F1-Score': f"{result['f1_score']:.4f}",
                    'G-Mean': f"{result['g_mean']:.4f}",
                    'AUC': f"{result['auc_roc']:.4f}",
                    'Time(s)': f"{result['training_time']:.1f}"
                })
        
        df = pd.DataFrame(summary_data)
        
        # 计算平均值
        numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'G-Mean', 'AUC']
        avg_row = {'Dataset': 'Average'}
        
        for col in numeric_cols:
            values = [float(df[col].iloc[i]) for i in range(len(df))]
            avg_row[col] = f"{np.mean(values):.4f}"
        
        # 添加其他列的平均值或总计
        avg_row['Samples'] = int(df['Samples'].sum())
        avg_row['Features'] = f"{df['Features'].mean():.1f}"
        avg_row['Classes'] = f"{df['Classes'].mean():.1f}"
        avg_row['IR'] = f"{np.mean([float(df['IR'].iloc[i]) for i in range(len(df))]):.2f}"
        avg_row['Time(s)'] = f"{np.sum([float(df['Time(s)'].iloc[i]) for i in range(len(df))]):.1f}"
        
        # 添加平均行
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        return df


def main():
    """
    主函数：运行完整实验流程
    """
    # 实验配置
    dataset_path = r"e:\OneDrive\文档\大学\团队资料\论文修改\DBUPLDM\workspace\datasets"
    
    # 检查数据集路径
    if not os.path.exists(dataset_path):
        logger.error(f"数据集路径不存在: {dataset_path}")
        return
    
    # 创建实验运行器
    runner = ExperimentRunner(dataset_path)
    
    # 实验配置
    experiment_config = {
        # 'hidden_dims': [128, 64],
        'hidden_dims': [8, 4],
        # 'dropout_rate': 0.3,
        'dropout_rate': 0.3,
        # 'learning_rate': 0.001,
        'learning_rate': 0.001,
        # 'batch_size': 32,
        'batch_size': 32,
        'epochs': 100,
        # 'gamma': 2.0,
        'gamma': 2.0,
        # 'patience': 15,
        'patience': 15,
        'random_state': 42
    }
    
    logger.info("开始 MLP + Focal Loss 实验")
    logger.info(f"MLP 实验配置: {experiment_config}")

    # 运行 MLP 实验
    mlp_results = runner.run_all_experiments(experiment_config)
    runner.save_results(mlp_results, 'mlp_focal_complete_results.json')
    mlp_summary_df = runner.generate_summary_table(mlp_results)
    mlp_summary_path = os.path.join(runner.results_dir, 'mlp_focal_summary.csv')
    mlp_summary_df.to_csv(mlp_summary_path, index=False)

    print("\n" + "="*80)
    print("MLP + Focal Loss 实验结果汇总")
    print("="*80)
    print(mlp_summary_df.to_string(index=False))
    print("="*80)
    logger.info(f"MLP 实验完成！汇总表已保存到: {mlp_summary_path}")

    # 运行 SVM 实验（作为对比基线）
    svm_config = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True,
        'max_iter': -1,
        'random_state': 42,
    }
    logger.info("开始 SVM 基线实验")
    logger.info(f"SVM 实验配置: {svm_config}")

    svm_results = runner.run_all_experiments_svm(svm_config)
    runner.save_results(svm_results, 'svm_complete_results.json')
    svm_summary_df = runner.generate_summary_table(svm_results)
    svm_summary_path = os.path.join(runner.results_dir, 'svm_summary.csv')
    svm_summary_df.to_csv(svm_summary_path, index=False)

    print("\n" + "="*80)
    print("SVM 基线 实验结果汇总")
    print("="*80)
    print(svm_summary_df.to_string(index=False))
    print("="*80)
    logger.info(f"SVM 实验完成！汇总表已保存到: {svm_summary_path}")

    # 生成简单对比（按 F1-Score）
    try:
        mlp_simple = mlp_summary_df[mlp_summary_df['Dataset'] != 'Average'][['Dataset', 'F1-Score']].copy()
        svm_simple = svm_summary_df[svm_summary_df['Dataset'] != 'Average'][['Dataset', 'F1-Score']].copy()
        mlp_simple.rename(columns={'F1-Score': 'MLP_F1'}, inplace=True)
        svm_simple.rename(columns={'F1-Score': 'SVM_F1'}, inplace=True)
        comp = pd.merge(mlp_simple, svm_simple, on='Dataset', how='inner')
        comp['MLP_F1'] = comp['MLP_F1'].astype(float)
        comp['SVM_F1'] = comp['SVM_F1'].astype(float)
        comp['Delta(MLP-SVM)'] = (comp['MLP_F1'] - comp['SVM_F1']).round(4)
        comp_path = os.path.join(runner.results_dir, 'mlp_vs_svm_comparison.csv')
        comp.to_csv(comp_path, index=False)
        logger.info(f"MLP vs SVM 对比表已保存到: {comp_path}")
    except Exception as e:
        logger.warning(f"生成对比表失败: {e}")


if __name__ == "__main__":
    main()
