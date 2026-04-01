#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M³-Net相关论文PDF批量下载脚本
Batch download papers for M³-Net architecture
"""

import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# 论文列表 (arXiv ID + 标题)
PAPERS = {
    # ===== 1. 多模态学习 =====
    "multimodal": [
        ("2306.06031", "FinGPT: Open-Source Financial Large Language Models"),
        ("2310.09605", "PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance"),
    ],
    
    # ===== 2. 混合专家模型MoE =====
    "moe": [
        ("2101.03961", "Switch Transformers: Scaling to Trillion Parameter Models"),
        ("2112.06905", "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"),
        ("2202.08906", "ST-MoE: Designing Stable and Transferable Sparse Expert Models"),
    ],
    
    # ===== 3. 世界模型 =====
    "world_model": [
        ("1803.10122", "World Models"),
        ("1912.01603", "Dreamer: Deep Reinforcement Learning for World Models"),
        ("2010.02193", "DreamerV2: Mastering Atari with Discrete World Models"),
        ("2301.04104", "DreamerV3: Mastering Diverse Domains"),
    ],
    
    # ===== 4. 时序Transformer =====
    "timeseries": [
        ("2310.06625", "iTransformer: Inverted Transformers for Time Series Forecasting"),
        ("2211.14730", "PatchTST: A Time Series is Worth 64 Words"),
        ("2106.13008", "Autoformer: Decomposition Transformers with Auto-Correlation"),
        ("2201.12740", "FEDformer: Frequency Enhanced Decomposed Transformer"),
        ("2012.07436", "Informer: Beyond Efficient Transformer for Long Sequence"),
        ("2210.02186", "TimesNet: Temporal 2D-Variation Modeling"),
        ("2403.10496", "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis"),
    ],
    
    # ===== 5. 图神经网络 =====
    "gnn": [
        ("1710.10903", "Graph Attention Networks"),
        ("1706.02216", "GraphSAGE: Inductive Representation Learning on Large Graphs"),
        ("2006.10637", "Temporal Graph Networks for Deep Learning on Dynamic Graphs"),
    ],
    
    # ===== 6. 强化学习 =====
    "rl": [
        ("1707.06347", "Proximal Policy Optimization Algorithms"),
        ("1801.01290", "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"),
    ],
    
    # ===== 7. 金融大语言模型 =====
    "finllm": [
        ("2303.17564", "BloombergGPT: A Large Language Model for Finance"),
    ],
}

# 需要搜索的非arXiv论文（只有标题/关键词）
NON_ARXIV_PAPERS = {
    "multimodal_search": [
        "Multi-Modal Machine Learning for Financial Forecasting 2024",
        "MM-Stock: Multi-Modal Stock Price Prediction 2024",
    ],
    "moe_finance": [
        "Mixture of Experts for Finance: Market Regime Detection 2024",
        "Dynamic Multi-Expert Networks for Stock Prediction 2024",
    ],
    "world_model_finance": [
        "World Models for Market Simulation and Trading 2024",
        "Model-Based Reinforcement Learning for Portfolio Optimization 2024",
    ],
    "gnn_finance": [
        "Stock Market Prediction Based on Graph Neural Networks 2023-2024",
        "Relational Graph Convolutional Networks for Stock Prediction 2023",
        "Dynamic Graph Learning for Financial Market Prediction 2024",
    ],
    "rl_trading": [
        "Deep Reinforcement Learning for Trading 2020-2024",
        "Deep Reinforcement Learning for Portfolio Management 2024",
        "Multi-Agent Reinforcement Learning for Market Making 2024",
    ],
    "causal": [
        "Causal Machine Learning for Trading Strategies 2024",
        "Neuro-Symbolic AI for Financial Reasoning 2024",
    ],
}

class ArXivDownloader:
    """arXiv论文下载器"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = Path.home() / ".openclaw" / "research-papers" / "m3-net-papers"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建分类子目录
        for category in PAPERS.keys():
            (self.output_dir / category).mkdir(exist_ok=True)
        
        self.downloaded = []
        self.failed = []
        self.skipped = []
    
    def download_pdf(self, arxiv_id: str, title: str, category: str) -> bool:
        """
        下载单篇论文PDF
        
        Args:
            arxiv_id: arXiv ID (如 2310.06625)
            title: 论文标题
            category: 分类目录
            
        Returns:
            是否成功
        """
        # 构建文件名
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title[:50]  # 限制长度
        filename = f"{arxiv_id}_{safe_title.replace(' ', '_')}.pdf"
        filepath = self.output_dir / category / filename
        
        # 检查是否已存在
        if filepath.exists():
            print(f"  ⏭️  已存在: {filename}")
            self.skipped.append((arxiv_id, title))
            return True
        
        # 构建下载URL
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        try:
            print(f"  📥 下载: {arxiv_id} - {title[:40]}...")
            
            # 设置请求头
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            # 下载文件
            with urllib.request.urlopen(req, timeout=60) as response:
                with open(filepath, 'wb') as f:
                    f.write(response.read())
            
            print(f"  ✅ 完成: {filepath.name}")
            self.downloaded.append((arxiv_id, title, filepath))
            return True
            
        except urllib.error.HTTPError as e:
            print(f"  ❌ HTTP错误 {e.code}: {arxiv_id}")
            self.failed.append((arxiv_id, title, f"HTTP {e.code}"))
            return False
        except urllib.error.URLError as e:
            print(f"  ❌ 网络错误: {arxiv_id} - {e.reason}")
            self.failed.append((arxiv_id, title, str(e.reason)))
            return False
        except Exception as e:
            print(f"  ❌ 错误: {arxiv_id} - {str(e)}")
            self.failed.append((arxiv_id, title, str(e)))
            return False
    
    def download_all(self):
        """批量下载所有论文"""
        print("=" * 70)
        print("M³-Net架构相关论文PDF批量下载")
        print(f"保存目录: {self.output_dir}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()
        
        total = sum(len(papers) for papers in PAPERS.values())
        current = 0
        
        for category, papers in PAPERS.items():
            print(f"\n📂 [{category.upper()}] - {len(papers)}篇论文")
            print("-" * 70)
            
            for arxiv_id, title in papers:
                current += 1
                print(f"\n[{current}/{total}]")
                self.download_pdf(arxiv_id, title, category)
                
                # 礼貌等待，避免请求过快
                time.sleep(2)
        
        # 打印汇总
        self._print_summary()
    
    def _print_summary(self):
        """打印下载汇总"""
        print("\n" + "=" * 70)
        print("下载完成汇总")
        print("=" * 70)
        print(f"\n✅ 成功下载: {len(self.downloaded)} 篇")
        print(f"⏭️  已存在跳过: {len(self.skipped)} 篇")
        print(f"❌ 下载失败: {len(self.failed)} 篇")
        
        if self.failed:
            print("\n❌ 失败的论文:")
            for arxiv_id, title, error in self.failed:
                print(f"  - {arxiv_id}: {title[:40]}... ({error})")
        
        if self.downloaded:
            print(f"\n📁 保存位置: {self.output_dir}")
            print("\n📚 已下载论文分类:")
            for category in PAPERS.keys():
                category_path = self.output_dir / category
                files = list(category_path.glob("*.pdf"))
                if files:
                    print(f"  📂 {category}: {len(files)} 篇")
        
        # 保存非arXiv论文列表（需要手动搜索）
        self._save_non_arxiv_list()
    
    def _save_non_arxiv_list(self):
        """保存需要手动搜索的论文列表"""
        list_file = self.output_dir / "non-arxiv-papers-to-search.txt"
        
        with open(list_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("以下论文不在arXiv，需要手动搜索下载\n")
            f.write("建议搜索: Google Scholar, ResearchGate, 期刊官网\n")
            f.write("=" * 70 + "\n\n")
            
            for category, papers in NON_ARXIV_PAPERS.items():
                f.write(f"\n【{category.upper()}】\n")
                f.write("-" * 70 + "\n")
                for paper in papers:
                    f.write(f"- {paper}\n")
                    # 添加搜索链接
                    query = paper.replace(' ', '+')
                    f.write(f"  搜索: https://scholar.google.com/scholar?q={query}\n")
                    f.write(f"  搜索: https://arxiv.org/search/?query={query}\n\n")
        
        print(f"\n📝 非arXiv论文列表已保存: {list_file}")


def main():
    """主函数"""
    downloader = ArXivDownloader()
    downloader.download_all()
    
    print("\n" + "=" * 70)
    print("提示:")
    print("- arXiv论文已全部尝试下载")
    print("- 非arXiv论文列表见 non-arxiv-papers-to-search.txt")
    print("- 部分论文可能需要通过学校图书馆或ResearchGate获取")
    print("=" * 70)


if __name__ == "__main__":
    main()
