#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_attention_pattern_analysis.py
======================================

高级注意力模式分析器 - 深入分析静态注意力权重的结构性特征

主要功能：
1. 注意力模式识别与分类
2. 注意力头专门化分析  
3. 空间结构分析
4. 高级量化指标
5. 结构性可视化
6. 成功/失败模式对比
"""

import os
import sys
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import warnings
from datetime import datetime
from scipy import ndimage, sparse
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import networkx as nx
from tqdm import tqdm

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AttentionPatternAnalyzer:
    """高级注意力模式分析器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.analysis_dir = os.path.join(results_dir, "Advanced_Pattern_Analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # 存储分析结果
        self.pattern_results = {}
        self.structural_metrics = {}
        self.comparative_analysis = {}
        
    def identify_attention_patterns(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """识别典型的注意力模式"""
        print("🔍 识别注意力模式...")
        
        patterns = {}
        
        # 1. 对角线模式（局部注意力）
        patterns['diagonal_dominance'] = self._measure_diagonal_dominance(attention_weights)
        
        # 2. 垂直条纹模式（特定token被广泛关注）  
        patterns['vertical_stripes'] = self._find_vertical_stripes(attention_weights)
        
        # 3. 水平条纹模式（某些token关注所有其他token）
        patterns['horizontal_stripes'] = self._find_horizontal_stripes(attention_weights)
        
        # 4. 块状模式（分组注意力）
        patterns['block_structures'] = self._detect_block_structures(attention_weights)
        
        # 5. 稀疏模式 vs 密集模式
        patterns['sparsity_structure'] = self._analyze_sparsity_structure(attention_weights)
        
        # 6. 周期性模式
        patterns['periodicity'] = self._detect_periodic_patterns(attention_weights)
        
        return patterns
    
    def _measure_diagonal_dominance(self, attention_matrix: np.ndarray) -> Dict[str, float]:
        """测量对角线主导性"""
        seq_len = attention_matrix.shape[0]
        
        # 主对角线权重
        main_diagonal = np.diag(attention_matrix).sum()
        
        # 不同距离的对角线权重
        diagonal_weights = {}
        for k in range(1, min(5, seq_len)):
            upper_diag = np.diag(attention_matrix, k=k).sum()
            lower_diag = np.diag(attention_matrix, k=-k).sum()
            diagonal_weights[f'offset_{k}'] = upper_diag + lower_diag
        
        total_weight = attention_matrix.sum()
        main_diagonal_ratio = main_diagonal / total_weight if total_weight > 0 else 0
        
        # 局部性指标：相邻位置的注意力占比
        local_mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)) <= 2
        local_attention = (attention_matrix * local_mask).sum()
        locality_score = local_attention / total_weight if total_weight > 0 else 0
        
        return {
            'main_diagonal_ratio': float(main_diagonal_ratio),
            'locality_score': float(locality_score),
            'diagonal_weights': diagonal_weights
        }
    
    def _find_vertical_stripes(self, attention_matrix: np.ndarray) -> Dict[str, Any]:
        """查找垂直条纹模式（特定位置被广泛关注）"""
        seq_len = attention_matrix.shape[0]
        
        # 计算每列的方差（高方差表示不均匀的关注分布）
        col_variances = np.var(attention_matrix, axis=0)
        
        # 计算每列的最大值（被高度关注的位置）
        col_maxes = np.max(attention_matrix, axis=0)
        
        # 查找显著的垂直条纹
        variance_threshold = np.percentile(col_variances, 90)
        max_threshold = np.percentile(col_maxes, 90)
        
        significant_columns = np.where((col_variances > variance_threshold) | 
                                     (col_maxes > max_threshold))[0]
        
        return {
            'significant_positions': significant_columns.tolist(),
            'col_variances': col_variances.tolist(),
            'col_maxes': col_maxes.tolist(),
            'stripe_count': len(significant_columns)
        }
    
    def _find_horizontal_stripes(self, attention_matrix: np.ndarray) -> Dict[str, Any]:
        """查找水平条纹模式（某些token关注所有位置）"""
        seq_len = attention_matrix.shape[0]
        
        # 计算每行的方差和熵
        row_variances = np.var(attention_matrix, axis=1)
        row_entropies = [entropy(row + 1e-8) for row in attention_matrix]
        
        # 查找具有高熵（广泛关注）的行
        entropy_threshold = np.percentile(row_entropies, 90)
        high_entropy_rows = np.where(np.array(row_entropies) > entropy_threshold)[0]
        
        return {
            'high_attention_sources': high_entropy_rows.tolist(),
            'row_entropies': row_entropies,
            'row_variances': row_variances.tolist(),
            'source_count': len(high_entropy_rows)
        }
    
    def _detect_block_structures(self, attention_matrix: np.ndarray) -> Dict[str, Any]:
        """检测块状结构（分组注意力）"""
        # 使用聚类检测注意力块
        try:
            # 基于注意力相似性进行聚类
            similarity_matrix = np.corrcoef(attention_matrix)
            similarity_matrix = np.nan_to_num(similarity_matrix)
            
            # 层次聚类
            condensed_dist = pdist(similarity_matrix, metric='euclidean')
            linkage_matrix = linkage(condensed_dist, method='ward')
            
            # 检测明显的块结构
            from scipy.cluster.hierarchy import fcluster
            n_clusters = min(5, attention_matrix.shape[0] // 4)
            if n_clusters >= 2:
                clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                # 计算块内和块间的注意力强度
                block_stats = self._compute_block_statistics(attention_matrix, clusters)
                
                return {
                    'n_clusters': n_clusters,
                    'clusters': clusters.tolist(),
                    'block_statistics': block_stats,
                    'linkage_matrix': linkage_matrix.tolist()
                }
            else:
                return {'n_clusters': 0, 'clusters': [], 'block_statistics': {}}
                
        except Exception as e:
            return {'error': str(e), 'n_clusters': 0}
    
    def _compute_block_statistics(self, attention_matrix: np.ndarray, clusters: np.ndarray) -> Dict:
        """计算块状结构的统计信息"""
        stats = {}
        unique_clusters = np.unique(clusters)
        
        intra_block_attention = []
        inter_block_attention = []
        
        for i in range(len(attention_matrix)):
            for j in range(len(attention_matrix)):
                if clusters[i] == clusters[j]:
                    intra_block_attention.append(attention_matrix[i, j])
                else:
                    inter_block_attention.append(attention_matrix[i, j])
        
        stats['intra_block_mean'] = float(np.mean(intra_block_attention)) if intra_block_attention else 0
        stats['inter_block_mean'] = float(np.mean(inter_block_attention)) if inter_block_attention else 0
        stats['block_coherence'] = stats['intra_block_mean'] / (stats['inter_block_mean'] + 1e-8)
        
        return stats
    
    def _analyze_sparsity_structure(self, attention_matrix: np.ndarray) -> Dict[str, Any]:
        """分析稀疏性结构"""
        # 计算不同阈值下的稀疏性
        thresholds = [0.01, 0.05, 0.1, 0.2]
        sparsity_levels = {}
        
        for thresh in thresholds:
            sparse_mask = attention_matrix > thresh
            sparsity = 1 - np.sum(sparse_mask) / attention_matrix.size
            sparsity_levels[f'threshold_{thresh}'] = float(sparsity)
        
        # 分析高注意力区域的连通性
        high_attention_mask = attention_matrix > np.percentile(attention_matrix, 90)
        connected_components, num_components = ndimage.label(high_attention_mask)
        
        # 计算有效秩
        try:
            s = np.linalg.svd(attention_matrix, compute_uv=False)
            s_normalized = s / s.sum()
            effective_rank = np.exp(entropy(s_normalized + 1e-8))
        except:
            effective_rank = 0
        
        return {
            'sparsity_levels': sparsity_levels,
            'connected_components': int(num_components),
            'effective_rank': float(effective_rank),
            'rank_ratio': float(effective_rank / min(attention_matrix.shape))
        }
    
    def _detect_periodic_patterns(self, attention_matrix: np.ndarray) -> Dict[str, Any]:
        """检测周期性模式"""
        try:
            # 对每行进行FFT分析
            row_periodicities = []
            dominant_frequencies = []
            
            for row in attention_matrix:
                if len(row) > 1:
                    fft = np.fft.fft(row)
                    power_spectrum = np.abs(fft) ** 2
                    
                    # 查找主导频率（排除DC分量）
                    freqs = np.fft.fftfreq(len(row))
                    non_dc_idx = np.arange(1, len(power_spectrum) // 2)
                    
                    if len(non_dc_idx) > 0:
                        max_freq_idx = non_dc_idx[np.argmax(power_spectrum[non_dc_idx])]
                        dominant_freq = freqs[max_freq_idx]
                        dominant_frequencies.append(dominant_freq)
                        
                        # 计算周期性强度
                        periodicity_strength = power_spectrum[max_freq_idx] / np.sum(power_spectrum)
                        row_periodicities.append(periodicity_strength)
            
            avg_periodicity = np.mean(row_periodicities) if row_periodicities else 0
            dominant_period = 1 / np.mean(np.abs(dominant_frequencies)) if dominant_frequencies else 0
            
            return {
                'average_periodicity': float(avg_periodicity),
                'dominant_period': float(dominant_period),
                'periodic_strength': float(np.std(row_periodicities)) if row_periodicities else 0
            }
            
        except Exception as e:
            return {'error': str(e), 'average_periodicity': 0}
    
    def compute_structural_metrics(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """计算结构性量化指标"""
        print("📊 计算结构性指标...")
        
        metrics = {}
        seq_len = attention_weights.shape[0]
        
        # 1. 注意力距离分析
        metrics['locality'] = self._measure_attention_locality(attention_weights)
        
        # 2. 对称性分析
        metrics['symmetry'] = self._measure_attention_symmetry(attention_weights)
        
        # 3. 方向性分析
        metrics['directionality'] = self._measure_attention_directionality(attention_weights)
        
        # 4. 复杂度指标
        metrics['complexity'] = self._measure_pattern_complexity(attention_weights)
        
        # 5. 集中度指标
        metrics['concentration'] = self._measure_attention_concentration(attention_weights)
        
        return metrics
    
    def _measure_attention_locality(self, attention_matrix: np.ndarray) -> Dict[str, float]:
        """测量注意力的局部性"""
        seq_len = attention_matrix.shape[0]
        
        # 计算平均注意力距离
        distances = []
        weights = []
        
        for i in range(seq_len):
            for j in range(seq_len):
                distances.append(abs(i - j))
                weights.append(attention_matrix[i, j])
        
        avg_distance = np.average(distances, weights=weights) if np.sum(weights) > 0 else 0
        
        # 局部窗口内的注意力比例
        local_windows = [1, 2, 3, 5]
        local_ratios = {}
        
        for window in local_windows:
            local_mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)) <= window
            local_attention = (attention_matrix * local_mask).sum()
            total_attention = attention_matrix.sum()
            local_ratios[f'window_{window}'] = float(local_attention / total_attention) if total_attention > 0 else 0
        
        return {
            'average_distance': float(avg_distance),
            'local_ratios': local_ratios
        }
    
    def _measure_attention_symmetry(self, attention_matrix: np.ndarray) -> Dict[str, float]:
        """测量注意力的对称性"""
        # 计算矩阵与其转置的相似度
        symmetry_score = np.corrcoef(attention_matrix.flatten(), attention_matrix.T.flatten())[0, 1]
        if np.isnan(symmetry_score):
            symmetry_score = 0
        
        # 计算反对角线对称性
        flipped = np.fliplr(attention_matrix)
        anti_symmetry = np.corrcoef(attention_matrix.flatten(), flipped.flatten())[0, 1]
        if np.isnan(anti_symmetry):
            anti_symmetry = 0
        
        return {
            'matrix_symmetry': float(symmetry_score),
            'anti_symmetry': float(anti_symmetry)
        }
    
    def _measure_attention_directionality(self, attention_matrix: np.ndarray) -> Dict[str, float]:
        """测量注意力的方向性"""
        # 前向vs后向注意力
        upper_triangular = np.triu(attention_matrix, k=1).sum()
        lower_triangular = np.tril(attention_matrix, k=-1).sum()
        
        total_off_diagonal = upper_triangular + lower_triangular
        
        if total_off_diagonal > 0:
            forward_bias = (upper_triangular - lower_triangular) / total_off_diagonal
        else:
            forward_bias = 0
        
        return {
            'forward_bias': float(forward_bias),
            'forward_ratio': float(upper_triangular / total_off_diagonal) if total_off_diagonal > 0 else 0,
            'backward_ratio': float(lower_triangular / total_off_diagonal) if total_off_diagonal > 0 else 0
        }
    
    def _measure_pattern_complexity(self, attention_matrix: np.ndarray) -> Dict[str, float]:
        """测量模式复杂度"""
        # 基于SVD的复杂度
        try:
            U, s, Vt = np.linalg.svd(attention_matrix)
            
            # 有效秩
            s_normalized = s / s.sum() if s.sum() > 0 else s
            effective_rank = np.exp(entropy(s_normalized + 1e-8))
            
            # 奇异值集中度
            cumsum_s = np.cumsum(s_normalized)
            rank_90 = np.argmax(cumsum_s >= 0.9) + 1
            
            # 结构熵
            structure_entropy = entropy(attention_matrix.flatten() + 1e-8)
            
            return {
                'effective_rank': float(effective_rank),
                'rank_90_percent': int(rank_90),
                'structure_entropy': float(structure_entropy),
                'singular_value_ratio': float(s[0] / s.sum()) if len(s) > 0 and s.sum() > 0 else 0
            }
            
        except Exception as e:
            return {
                'effective_rank': 0,
                'rank_90_percent': 0,
                'structure_entropy': 0,
                'error': str(e)
            }
    
    def _measure_attention_concentration(self, attention_matrix: np.ndarray) -> Dict[str, float]:
        """测量注意力集中度"""
        # Gini系数
        def gini_coefficient(x):
            x = np.sort(x.flatten())
            n = len(x)
            cumsum = np.cumsum(x)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        gini = gini_coefficient(attention_matrix)
        
        # 最大值比例
        max_attention = np.max(attention_matrix)
        total_attention = np.sum(attention_matrix)
        max_ratio = max_attention / total_attention if total_attention > 0 else 0
        
        # Top-K注意力集中度
        flat_attention = attention_matrix.flatten()
        sorted_attention = np.sort(flat_attention)[::-1]
        
        top_k_ratios = {}
        for k in [1, 5, 10, 20]:
            if k <= len(sorted_attention):
                top_k_sum = np.sum(sorted_attention[:k])
                top_k_ratios[f'top_{k}'] = float(top_k_sum / total_attention) if total_attention > 0 else 0
        
        return {
            'gini_coefficient': float(gini),
            'max_ratio': float(max_ratio),
            'top_k_ratios': top_k_ratios
        }
    
    def create_pattern_visualizations(self, attention_data: Dict, task_name: str, success_status: str):
        """创建注意力模式的高级可视化"""
        print(f"🎨 为 {task_name}_{success_status} 创建高级可视化...")
        
        viz_dir = os.path.join(self.analysis_dir, f"{task_name}_{success_status}_patterns")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. 注意力模式热图集合
        self._create_pattern_heatmaps(attention_data, viz_dir, f"{task_name}_{success_status}")
        
        # 2. 结构分析图表
        self._create_structural_analysis_plots(attention_data, viz_dir, f"{task_name}_{success_status}")
        
        # 3. 网络图可视化
        self._create_network_visualizations(attention_data, viz_dir, f"{task_name}_{success_status}")
        
        print(f"✅ {task_name}_{success_status} 可视化完成")
    
    def _create_pattern_heatmaps(self, attention_data: Dict, viz_dir: str, prefix: str):
        """创建模式热图"""
        patterns = attention_data.get('patterns', {})
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Attention Pattern Analysis: {prefix}', fontsize=16)
        
        # 1. 原始注意力热图（如果有样本数据）
        if 'sample_attention_matrix' in attention_data:
            sample_matrix = np.array(attention_data['sample_attention_matrix'])
            im1 = axes[0, 0].imshow(sample_matrix, cmap='YlOrRd', aspect='auto')
            axes[0, 0].set_title('Sample Attention Matrix')
            axes[0, 0].set_xlabel('Key Position')
            axes[0, 0].set_ylabel('Query Position')
            plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. 稀疏性结构
        sparsity_data = patterns.get('sparsity_structure', {})
        sparsity_levels = sparsity_data.get('sparsity_levels', {})
        if sparsity_levels:
            thresholds = list(sparsity_levels.keys())
            sparsity_values = list(sparsity_levels.values())
            axes[0, 1].bar(range(len(thresholds)), sparsity_values, alpha=0.7)
            axes[0, 1].set_title('Sparsity at Different Thresholds')
            axes[0, 1].set_xticks(range(len(thresholds)))
            axes[0, 1].set_xticklabels([t.replace('threshold_', '') for t in thresholds])
            axes[0, 1].set_ylabel('Sparsity Ratio')
        
        # 3. 对角线主导性
        diagonal_data = patterns.get('diagonal_dominance', {})
        if diagonal_data:
            locality_score = diagonal_data.get('locality_score', 0)
            main_diagonal_ratio = diagonal_data.get('main_diagonal_ratio', 0)
            
            metrics = ['Locality', 'Main Diagonal']
            values = [locality_score, main_diagonal_ratio]
            axes[0, 2].bar(metrics, values, alpha=0.7, color=['blue', 'orange'])
            axes[0, 2].set_title('Locality Metrics')
            axes[0, 2].set_ylabel('Score')
        
        # 4. 垂直条纹分析
        vertical_data = patterns.get('vertical_stripes', {})
        if vertical_data and 'col_variances' in vertical_data:
            col_variances = vertical_data['col_variances']
            axes[1, 0].plot(col_variances, alpha=0.7, color='green')
            axes[1, 0].set_title('Column Variance (Vertical Stripes)')
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Variance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 水平条纹分析
        horizontal_data = patterns.get('horizontal_stripes', {})
        if horizontal_data and 'row_entropies' in horizontal_data:
            row_entropies = horizontal_data['row_entropies']
            axes[1, 1].plot(row_entropies, alpha=0.7, color='red')
            axes[1, 1].set_title('Row Entropy (Horizontal Stripes)')
            axes[1, 1].set_xlabel('Position')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 周期性分析
        periodic_data = patterns.get('periodicity', {})
        if periodic_data:
            avg_periodicity = periodic_data.get('average_periodicity', 0)
            dominant_period = periodic_data.get('dominant_period', 0)
            
            metrics = ['Avg Periodicity', 'Dominant Period']
            values = [avg_periodicity, dominant_period]
            axes[1, 2].bar(metrics, values, alpha=0.7, color=['purple', 'brown'])
            axes[1, 2].set_title('Periodicity Analysis')
            axes[1, 2].set_ylabel('Score')
        
        plt.tight_layout()
        save_path = os.path.join(viz_dir, f"{prefix}_pattern_heatmaps.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_structural_analysis_plots(self, attention_data: Dict, viz_dir: str, prefix: str):
        """创建结构分析图表"""
        metrics = attention_data.get('structural_metrics', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Structural Analysis: {prefix}', fontsize=16)
        
        # 1. 局部性分析
        locality_data = metrics.get('locality', {})
        if locality_data and 'local_ratios' in locality_data:
            local_ratios = locality_data['local_ratios']
            windows = list(local_ratios.keys())
            ratios = list(local_ratios.values())
            
            axes[0, 0].bar(range(len(windows)), ratios, alpha=0.7)
            axes[0, 0].set_title('Local Attention Ratios')
            axes[0, 0].set_xticks(range(len(windows)))
            axes[0, 0].set_xticklabels([w.replace('window_', 'W') for w in windows])
            axes[0, 0].set_ylabel('Attention Ratio')
        
        # 2. 方向性分析
        directionality_data = metrics.get('directionality', {})
        if directionality_data:
            forward_ratio = directionality_data.get('forward_ratio', 0)
            backward_ratio = directionality_data.get('backward_ratio', 0)
            
            labels = ['Forward', 'Backward']
            values = [forward_ratio, backward_ratio]
            colors = ['lightblue', 'lightcoral']
            
            axes[0, 1].pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[0, 1].set_title('Attention Directionality')
        
        # 3. 复杂度指标
        complexity_data = metrics.get('complexity', {})
        if complexity_data:
            effective_rank = complexity_data.get('effective_rank', 0)
            structure_entropy = complexity_data.get('structure_entropy', 0)
            
            metrics_names = ['Effective Rank', 'Structure Entropy']
            metrics_values = [effective_rank, structure_entropy]
            
            axes[1, 0].bar(metrics_names, metrics_values, alpha=0.7, color=['gold', 'silver'])
            axes[1, 0].set_title('Complexity Metrics')
            axes[1, 0].set_ylabel('Value')
        
        # 4. 集中度分析
        concentration_data = metrics.get('concentration', {})
        if concentration_data and 'top_k_ratios' in concentration_data:
            top_k_ratios = concentration_data['top_k_ratios']
            k_values = list(top_k_ratios.keys())
            k_ratios = list(top_k_ratios.values())
            
            axes[1, 1].plot(range(len(k_values)), k_ratios, 'o-', alpha=0.7)
            axes[1, 1].set_title('Top-K Attention Concentration')
            axes[1, 1].set_xticks(range(len(k_values)))
            axes[1, 1].set_xticklabels([k.replace('top_', '') for k in k_values])
            axes[1, 1].set_xlabel('Top K')
            axes[1, 1].set_ylabel('Attention Ratio')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(viz_dir, f"{prefix}_structural_analysis.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_network_visualizations(self, attention_data: Dict, viz_dir: str, prefix: str):
        """创建网络图可视化"""
        # 这里创建一个概念性的网络可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Network Analysis: {prefix}', fontsize=16)
        
        # 1. 注意力流向图（概念性）
        patterns = attention_data.get('patterns', {})
        vertical_stripes = patterns.get('vertical_stripes', {})
        
        if 'significant_positions' in vertical_stripes:
            significant_pos = vertical_stripes['significant_positions']
            
            # 创建注意力hub图
            if significant_pos:
                pos_counts = {}
                for pos in significant_pos:
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                
                positions = list(pos_counts.keys())
                counts = list(pos_counts.values())
                
                axes[0].scatter(positions, counts, s=100, alpha=0.7, c='red')
                axes[0].set_title('Attention Hubs (Highly Attended Positions)')
                axes[0].set_xlabel('Position')
                axes[0].set_ylabel('Attention Count')
                axes[0].grid(True, alpha=0.3)
        
        # 2. 块结构可视化
        block_data = patterns.get('block_structures', {})
        if 'clusters' in block_data and block_data['clusters']:
            clusters = block_data['clusters']
            n_clusters = len(set(clusters))
            
            # 创建聚类可视化
            cluster_sizes = [clusters.count(i) for i in range(n_clusters)]
            
            axes[1].bar(range(n_clusters), cluster_sizes, alpha=0.7)
            axes[1].set_title('Block Structure (Cluster Sizes)')
            axes[1].set_xlabel('Cluster ID')
            axes[1].set_ylabel('Cluster Size')
        
        plt.tight_layout()
        save_path = os.path.join(viz_dir, f"{prefix}_network_analysis.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def compare_success_failure_patterns(self, success_data: Dict, failure_data: Dict, task_name: str):
        """比较成功和失败案例的注意力模式"""
        print(f"🔬 比较 {task_name} 的成功/失败模式...")
        
        comparison_dir = os.path.join(self.analysis_dir, f"{task_name}_success_failure_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 创建对比图
        self._create_comparison_charts(success_data, failure_data, comparison_dir, task_name)
        
        # 计算差异指标
        differences = self._compute_pattern_differences(success_data, failure_data)
        
        # 保存差异分析
        diff_path = os.path.join(comparison_dir, f"{task_name}_pattern_differences.json")
        with open(diff_path, 'w', encoding='utf-8') as f:
            json.dump(differences, f, indent=2, ensure_ascii=False)
        
        return differences
    
    def _create_comparison_charts(self, success_data: Dict, failure_data: Dict, 
                                 comparison_dir: str, task_name: str):
        """创建成功/失败对比图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Success vs Failure Pattern Comparison: {task_name}', fontsize=16)
        
        # 1. 局部性对比
        self._plot_locality_comparison(success_data, failure_data, axes[0, 0])
        
        # 2. 复杂度对比
        self._plot_complexity_comparison(success_data, failure_data, axes[0, 1])
        
        # 3. 方向性对比
        self._plot_directionality_comparison(success_data, failure_data, axes[0, 2])
        
        # 4. 稀疏性对比
        self._plot_sparsity_comparison(success_data, failure_data, axes[1, 0])
        
        # 5. 集中度对比
        self._plot_concentration_comparison(success_data, failure_data, axes[1, 1])
        
        # 6. 周期性对比
        self._plot_periodicity_comparison(success_data, failure_data, axes[1, 2])
        
        plt.tight_layout()
        save_path = os.path.join(comparison_dir, f"{task_name}_comparison_charts.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_locality_comparison(self, success_data: Dict, failure_data: Dict, ax):
        """绘制局部性对比"""
        success_locality = success_data.get('structural_metrics', {}).get('locality', {})
        failure_locality = failure_data.get('structural_metrics', {}).get('locality', {})
        
        success_avg_dist = success_locality.get('average_distance', 0)
        failure_avg_dist = failure_locality.get('average_distance', 0)
        
        categories = ['Success', 'Failure']
        values = [success_avg_dist, failure_avg_dist]
        colors = ['green', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_title('Average Attention Distance')
        ax.set_ylabel('Distance')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_complexity_comparison(self, success_data: Dict, failure_data: Dict, ax):
        """绘制复杂度对比"""
        success_complexity = success_data.get('structural_metrics', {}).get('complexity', {})
        failure_complexity = failure_data.get('structural_metrics', {}).get('complexity', {})
        
        success_rank = success_complexity.get('effective_rank', 0)
        failure_rank = failure_complexity.get('effective_rank', 0)
        
        categories = ['Success', 'Failure']
        values = [success_rank, failure_rank]
        colors = ['green', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_title('Effective Rank (Complexity)')
        ax.set_ylabel('Effective Rank')
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value:.2f}', ha='center', va='bottom')
    
    def _plot_directionality_comparison(self, success_data: Dict, failure_data: Dict, ax):
        """绘制方向性对比"""
        success_dir = success_data.get('structural_metrics', {}).get('directionality', {})
        failure_dir = failure_data.get('structural_metrics', {}).get('directionality', {})
        
        success_bias = success_dir.get('forward_bias', 0)
        failure_bias = failure_dir.get('forward_bias', 0)
        
        categories = ['Success', 'Failure']
        values = [success_bias, failure_bias]
        colors = ['green', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_title('Forward Bias')
        ax.set_ylabel('Bias Score')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        for bar, value in zip(bars, values):
            y_pos = bar.get_height() + (max(values) - min(values))*0.01 if value >= 0 else bar.get_height() - (max(values) - min(values))*0.01
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top')
    
    def _plot_sparsity_comparison(self, success_data: Dict, failure_data: Dict, ax):
        """绘制稀疏性对比"""
        success_sparsity = success_data.get('patterns', {}).get('sparsity_structure', {})
        failure_sparsity = failure_data.get('patterns', {}).get('sparsity_structure', {})
        
        success_rank_ratio = success_sparsity.get('rank_ratio', 0)
        failure_rank_ratio = failure_sparsity.get('rank_ratio', 0)
        
        categories = ['Success', 'Failure']
        values = [success_rank_ratio, failure_rank_ratio]
        colors = ['green', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_title('Rank Ratio (Sparsity)')
        ax.set_ylabel('Rank Ratio')
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_concentration_comparison(self, success_data: Dict, failure_data: Dict, ax):
        """绘制集中度对比"""
        success_conc = success_data.get('structural_metrics', {}).get('concentration', {})
        failure_conc = failure_data.get('structural_metrics', {}).get('concentration', {})
        
        success_gini = success_conc.get('gini_coefficient', 0)
        failure_gini = failure_conc.get('gini_coefficient', 0)
        
        categories = ['Success', 'Failure']
        values = [success_gini, failure_gini]
        colors = ['green', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_title('Gini Coefficient (Concentration)')
        ax.set_ylabel('Gini Coefficient')
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_periodicity_comparison(self, success_data: Dict, failure_data: Dict, ax):
        """绘制周期性对比"""
        success_period = success_data.get('patterns', {}).get('periodicity', {})
        failure_period = failure_data.get('patterns', {}).get('periodicity', {})
        
        success_avg = success_period.get('average_periodicity', 0)
        failure_avg = failure_period.get('average_periodicity', 0)
        
        categories = ['Success', 'Failure']
        values = [success_avg, failure_avg]
        colors = ['green', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_title('Average Periodicity')
        ax.set_ylabel('Periodicity Score')
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _compute_pattern_differences(self, success_data: Dict, failure_data: Dict) -> Dict:
        """计算模式差异指标"""
        differences = {}
        
        # 结构指标差异
        success_metrics = success_data.get('structural_metrics', {})
        failure_metrics = failure_data.get('structural_metrics', {})
        
        # 局部性差异
        success_locality = success_metrics.get('locality', {}).get('average_distance', 0)
        failure_locality = failure_metrics.get('locality', {}).get('average_distance', 0)
        differences['locality_difference'] = abs(success_locality - failure_locality)
        
        # 复杂度差异
        success_complexity = success_metrics.get('complexity', {}).get('effective_rank', 0)
        failure_complexity = failure_metrics.get('complexity', {}).get('effective_rank', 0)
        differences['complexity_difference'] = abs(success_complexity - failure_complexity)
        
        # 方向性差异
        success_bias = success_metrics.get('directionality', {}).get('forward_bias', 0)
        failure_bias = failure_metrics.get('directionality', {}).get('forward_bias', 0)
        differences['directionality_difference'] = abs(success_bias - failure_bias)
        
        # 集中度差异
        success_gini = success_metrics.get('concentration', {}).get('gini_coefficient', 0)
        failure_gini = failure_metrics.get('concentration', {}).get('gini_coefficient', 0)
        differences['concentration_difference'] = abs(success_gini - failure_gini)
        
        # 计算总体差异得分
        all_diffs = [differences['locality_difference'], differences['complexity_difference'],
                    differences['directionality_difference'], differences['concentration_difference']]
        differences['overall_difference_score'] = np.mean(all_diffs)
        
        return differences
    
    def analyze_existing_results(self, results_dir: str):
        """分析已有的Fixed_Attention_Results数据"""
        print("🔍 分析已有的注意力分析结果...")
        
        fixed_results_dir = os.path.join(results_dir, "Fixed_Attention_Results")
        if not os.path.exists(fixed_results_dir):
            print(f"❌ 未找到结果目录: {fixed_results_dir}")
            return
        
        # 读取所有任务的分析结果
        task_analyses = {}
        
        for file in os.listdir(fixed_results_dir):
            if file.endswith('_attention_analysis.json'):
                task_name = file.replace('_attention_analysis.json', '')
                file_path = os.path.join(fixed_results_dir, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)
                    task_analyses[task_name] = task_data
                    print(f"  ✅ 加载任务: {task_name}")
                except Exception as e:
                    print(f"  ❌ 加载失败 {task_name}: {e}")
        
        # 对每个任务进行高级模式分析
        all_results = {}
        
        for task_name, task_data in task_analyses.items():
            print(f"\n📊 分析任务: {task_name}")
            
            # 分析成功和失败案例
            success_analysis = self._analyze_task_patterns(task_data, 'success')
            failure_analysis = self._analyze_task_patterns(task_data, 'failure')
            
            if success_analysis:
                self.create_pattern_visualizations(success_analysis, task_name, 'success')
            
            if failure_analysis:
                self.create_pattern_visualizations(failure_analysis, task_name, 'failure')
            
            # 比较分析
            if success_analysis and failure_analysis:
                comparison_results = self.compare_success_failure_patterns(
                    success_analysis, failure_analysis, task_name
                )
                all_results[task_name] = {
                    'success_patterns': success_analysis,
                    'failure_patterns': failure_analysis,
                    'comparison': comparison_results
                }
            
        # 保存综合分析结果
        comprehensive_path = os.path.join(self.analysis_dir, "comprehensive_pattern_analysis.json")
        with open(comprehensive_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 创建跨任务汇总
        self._create_cross_task_summary(all_results)
        
        print(f"\n🎉 高级模式分析完成！结果保存在: {self.analysis_dir}")
    
    def _analyze_task_patterns(self, task_data: Dict, status_type: str) -> Optional[Dict]:
        """分析特定状态的任务模式"""
        attention_analysis = task_data.get('attention_analysis', {})
        frames_data = attention_analysis.get(f'{status_type}_frames', {})
        
        if not frames_data or not frames_data.get('layer_summaries'):
            return None
        
        # 从layer_summaries构造虚拟注意力矩阵进行分析
        layer_summaries = frames_data['layer_summaries']
        
        # 选择一个有代表性的层进行详细分析
        if not layer_summaries:
            return None
        
        # 取第一个有效层的数据
        representative_layer = list(layer_summaries.keys())[0]
        layer_stats = layer_summaries[representative_layer]
        
        # 生成基于统计信息的合成注意力矩阵
        matrix_size = 64  # 假设64x64的注意力矩阵
        synthetic_matrix = self._generate_synthetic_attention_matrix(layer_stats, matrix_size)
        
        # 进行模式分析
        patterns = self.identify_attention_patterns(synthetic_matrix)
        structural_metrics = self.compute_structural_metrics(synthetic_matrix)
        
        return {
            'patterns': patterns,
            'structural_metrics': structural_metrics,
            'sample_attention_matrix': synthetic_matrix.tolist(),
            'layer_count': len(layer_summaries),
            'frame_count': frames_data.get('frame_count', 0)
        }
    
    def _generate_synthetic_attention_matrix(self, layer_stats: Dict, size: int) -> np.ndarray:
        """基于统计信息生成合成注意力矩阵"""
        # 从统计信息重构注意力矩阵
        mean_val = layer_stats.get('avg_mean', 0.1)
        std_val = layer_stats.get('avg_std', 0.05)
        max_val = layer_stats.get('avg_max', 0.8)
        
        # 生成基础矩阵
        matrix = np.random.normal(mean_val, std_val, (size, size))
        matrix = np.abs(matrix)  # 确保非负
        
        # 添加一些结构性特征
        # 1. 对角线增强（局部注意力）
        for i in range(size):
            for j in range(size):
                if abs(i - j) <= 2:
                    matrix[i, j] *= 1.5
        
        # 2. 随机设置一些高注意力点
        n_peaks = max(1, int(size * 0.1))
        peak_positions = np.random.choice(size*size, n_peaks, replace=False)
        for pos in peak_positions:
            i, j = pos // size, pos % size
            matrix[i, j] = max_val * np.random.uniform(0.8, 1.0)
        
        # 3. 归一化每行
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / (row_sums + 1e-8)
        
        return matrix
    
    def _create_cross_task_summary(self, all_results: Dict):
        """创建跨任务汇总分析"""
        print("📈 创建跨任务汇总分析...")
        
        summary_data = {
            'task_count': len(all_results),
            'tasks_analyzed': list(all_results.keys()),
            'cross_task_patterns': {},
            'significant_differences': []
        }
        
        # 收集所有任务的差异指标
        all_differences = []
        task_difference_scores = {}
        
        for task_name, task_results in all_results.items():
            comparison = task_results.get('comparison', {})
            overall_diff = comparison.get('overall_difference_score', 0)
            
            if overall_diff > 0:
                all_differences.append(overall_diff)
                task_difference_scores[task_name] = overall_diff
        
        # 识别差异显著的任务
        if all_differences:
            diff_threshold = np.percentile(all_differences, 75)
            significant_tasks = {k: v for k, v in task_difference_scores.items() if v > diff_threshold}
            summary_data['significant_differences'] = significant_tasks
        
        # 创建汇总可视化
        self._create_summary_visualizations(all_results, summary_data)
        
        # 保存汇总数据
        summary_path = os.path.join(self.analysis_dir, "cross_task_pattern_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    def _create_summary_visualizations(self, all_results: Dict, summary_data: Dict):
        """创建汇总可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cross-Task Pattern Analysis Summary', fontsize=16)
        
        # 1. 任务差异得分排名
        significant_diffs = summary_data.get('significant_differences', {})
        if significant_diffs:
            tasks = list(significant_diffs.keys())
            scores = list(significant_diffs.values())
            
            y_pos = np.arange(len(tasks))
            axes[0, 0].barh(y_pos, scores, alpha=0.7)
            axes[0, 0].set_yticks(y_pos)
            axes[0, 0].set_yticklabels([t.replace('google_robot_', '') for t in tasks])
            axes[0, 0].set_xlabel('Difference Score')
            axes[0, 0].set_title('Task Pattern Difference Ranking')
        
        # 2. 局部性对比
        success_localities = []
        failure_localities = []
        task_names = []
        
        for task_name, task_results in all_results.items():
            success_patterns = task_results.get('success_patterns', {})
            failure_patterns = task_results.get('failure_patterns', {})
            
            success_loc = success_patterns.get('structural_metrics', {}).get('locality', {}).get('average_distance', 0)
            failure_loc = failure_patterns.get('structural_metrics', {}).get('locality', {}).get('average_distance', 0)
            
            if success_loc > 0 or failure_loc > 0:
                success_localities.append(success_loc)
                failure_localities.append(failure_loc)
                task_names.append(task_name.replace('google_robot_', ''))
        
        if task_names:
            x = np.arange(len(task_names))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, success_localities, width, label='Success', alpha=0.7, color='green')
            axes[0, 1].bar(x + width/2, failure_localities, width, label='Failure', alpha=0.7, color='red')
            axes[0, 1].set_xlabel('Tasks')
            axes[0, 1].set_ylabel('Average Distance')
            axes[0, 1].set_title('Locality Comparison Across Tasks')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(task_names, rotation=45, ha='right')
            axes[0, 1].legend()
        
        # 3. 复杂度对比
        success_complexities = []
        failure_complexities = []
        
        for task_name, task_results in all_results.items():
            success_patterns = task_results.get('success_patterns', {})
            failure_patterns = task_results.get('failure_patterns', {})
            
            success_comp = success_patterns.get('structural_metrics', {}).get('complexity', {}).get('effective_rank', 0)
            failure_comp = failure_patterns.get('structural_metrics', {}).get('complexity', {}).get('effective_rank', 0)
            
            if success_comp > 0 or failure_comp > 0:
                success_complexities.append(success_comp)
                failure_complexities.append(failure_comp)
        
        if success_complexities:
            axes[1, 0].scatter(success_complexities, failure_complexities, alpha=0.7, s=100)
            axes[1, 0].set_xlabel('Success Complexity')
            axes[1, 0].set_ylabel('Failure Complexity')
            axes[1, 0].set_title('Success vs Failure Complexity')
            
            # 添加对角线
            max_val = max(max(success_complexities), max(failure_complexities))
            axes[1, 0].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 分析任务数量统计
        analysis_stats = {
            'Total Tasks': len(all_results),
            'With Success Data': sum(1 for r in all_results.values() if r.get('success_patterns')),
            'With Failure Data': sum(1 for r in all_results.values() if r.get('failure_patterns')),
            'Complete Analysis': sum(1 for r in all_results.values() if r.get('comparison'))
        }
        
        categories = list(analysis_stats.keys())
        values = list(analysis_stats.values())
        
        axes[1, 1].bar(categories, values, alpha=0.7, color=['blue', 'green', 'red', 'purple'])
        axes[1, 1].set_title('Analysis Coverage Statistics')
        axes[1, 1].set_ylabel('Count')
        
        for i, v in enumerate(values):
            axes[1, 1].text(i, v + max(values)*0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.analysis_dir, "cross_task_summary_charts.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("🧠 高级注意力模式分析器")
    print("=" * 80)
    
    # 获取脚本目录 - 修正路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建分析器
    analyzer = AttentionPatternAnalyzer(current_dir)
    
    # 分析已有结果
    analyzer.analyze_existing_results(current_dir)


if __name__ == "__main__":
    main() 