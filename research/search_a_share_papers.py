#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股量化模型论文搜索脚本
Search for best A-share quantitative trading models since 2025
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# 添加 skill 路径
skill_path = Path.home() / ".openclaw" / "workspace" / "skills" / "paper-recommendation" / "research-paper-monitor-1.0.2" / "scripts"
sys.path.insert(0, str(skill_path))

# 搜索配置
SEARCH_CONFIG = {
    "keywords": [
        # A股/中国市场
        "A-share",
        "Chinese stock market",
        "China equity",
        # 量化交易
        "quantitative trading",
        "algorithmic trading",
        "statistical arbitrage",
        # 机器学习
        "machine learning trading",
        "deep learning finance",
        "reinforcement learning trading",
        # 时间序列
        "time series forecasting",
        "stock prediction",
        "price prediction",
        # 高级模型
        "transformer finance",
        "LSTM stock",
        "graph neural network GNN finance",
        "large language model LLM trading",
        # 多因子
        "multi-factor model",
        "factor investing",
        "cross-sectional returns",
        # 高频/日内
        "high frequency trading",
        "intraday trading",
        "market microstructure"
    ],
    "sources": ["arxiv", "google_scholar"],
    "date_range_days": 365,  # 2025年至今
    "min_score": 70
}

def search_arxiv_papers():
    """搜索 arXiv 论文"""
    try:
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET
        
        # 构建搜索查询
        query_parts = []
        for kw in SEARCH_CONFIG["keywords"][:10]:  # 前10个关键词
            query_parts.append(kw.replace(" ", "+"))
        
        # arXiv API 查询
        base_url = "http://export.arxiv.org/api/query"
        query = " OR ".join(query_parts)
        
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 50,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        print(f"正在搜索 arXiv: {query[:80]}...")
        
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
            
        # 解析 XML
        root = ET.fromstring(data)
        
        # 命名空间
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns)
            summary = entry.find('atom:summary', ns)
            published = entry.find('atom:published', ns)
            authors = entry.findall('atom:author', ns)
            link = entry.find('atom:link[@rel="alternate"]', ns)
            
            if title is not None and published is not None:
                # 检查日期（2025年以后）
                pub_date = datetime.fromisoformat(published.text.replace('Z', '+00:00'))
                if pub_date.year >= 2025:
                    paper = {
                        "title": title.text.strip(),
                        "abstract": summary.text.strip() if summary is not None else "",
                        "published": published.text[:10],
                        "authors": [a.find('atom:name', ns).text for a in authors if a.find('atom:name', ns) is not None],
                        "url": link.get('href') if link is not None else "",
                        "source": "arXiv"
                    }
                    papers.append(paper)
        
        return papers
        
    except Exception as e:
        print(f"arXiv 搜索出错: {e}")
        return []

def search_quant_papers():
    """搜索量化交易论文主函数"""
    print("=" * 60)
    print("A股量化模型论文搜索")
    print("搜索范围: 2025年至今")
    print("=" * 60)
    print()
    
    papers = search_arxiv_papers()
    
    if not papers:
        print("未找到论文，尝试直接访问 arXiv 网站...")
        print()
        print("推荐的 arXiv 搜索链接:")
        print("1. 量化交易 + 深度学习:")
        print("   https://arxiv.org/search/?query=quantitative+trading+deep+learning&searchtype=all")
        print()
        print("2. 时间序列预测 + 金融:")
        print("   https://arxiv.org/search/?query=time+series+forecasting+finance&searchtype=all")
        print()
        print("3. 强化学习 + 交易:")
        print("   https://arxiv.org/search/?query=reinforcement+learning+trading&searchtype=all")
        print()
        print("4. Transformer + 股票预测:")
        print("   https://arxiv.org/search/?query=transformer+stock+prediction&searchtype=all")
        return []
    
    print(f"找到 {len(papers)} 篇论文\n")
    
    # 分类展示
    categories = {
        "深度学习模型": ["deep learning", "neural network", "LSTM", "transformer"],
        "强化学习": ["reinforcement learning", "RL", "actor-critic", "PPO"],
        "图神经网络": ["graph neural", "GNN", "graph attention"],
        "多因子模型": ["factor", "multi-factor", "cross-sectional"],
        "高频交易": ["high frequency", "intraday", "microstructure"],
        "其他": []
    }
    
    categorized = {k: [] for k in categories}
    
    for paper in papers:
        title_lower = paper["title"].lower()
        abstract_lower = paper["abstract"].lower()
        text = title_lower + " " + abstract_lower
        
        assigned = False
        for cat, keywords in categories.items():
            if cat == "其他":
                continue
            for kw in keywords:
                if kw.lower() in text:
                    categorized[cat].append(paper)
                    assigned = True
                    break
            if assigned:
                break
        
        if not assigned:
            categorized["其他"].append(paper)
    
    # 输出结果
    for cat, cat_papers in categorized.items():
        if cat_papers:
            print(f"\n{'='*60}")
            print(f"【{cat}】- {len(cat_papers)} 篇")
            print('='*60)
            
            for i, paper in enumerate(cat_papers[:5], 1):  # 每类最多显示5篇
                print(f"\n{i}. {paper['title']}")
                print(f"   作者: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                print(f"   日期: {paper['published']}")
                print(f"   链接: {paper['url']}")
                
                # 简短摘要
                abstract = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
                print(f"   摘要: {abstract}")
    
    # 保存结果
    output_dir = Path.home() / ".openclaw" / "research-papers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"a-share-quant-papers-{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n结果已保存: {output_file}")
    
    return papers

if __name__ == "__main__":
    search_quant_papers()
    
    print("\n" + "=" * 60)
    print("推荐关注的高影响力论文（2024-2025）:")
    print("=" * 60)
    print()
    print("1. PatchTST: A Time Series is Worth 64 Words")
    print("   - Transformer架构，特别适合金融时间序列")
    print("   - arXiv: 2211.14730")
    print()
    print("2. iTransformer: Inverted Transformers for Time Series")
    print("   - 2024年最新，SOTA效果")
    print("   - arXiv: 2310.06625")
    print()
    print("3. TimesNet: Temporal 2D-Variation Modeling")
    print("   - 多周期建模，捕捉市场周期")
    print("   - arXiv: 2210.02186")
    print()
    print("4. ModernTCN: A Modern Pure Convolution Structure")
    print("   - 纯卷积架构，计算快效果好")
    print("   - arXiv: 2403.10496")
    print()
    print("5. Reinforcement Learning for Trading")
    print("   - PPO/DDPG 在量化交易中的应用")
    print("   - 搜索关键词: 'deep reinforcement learning trading'")
    print()
    print("6. Graph Neural Networks for Stock Prediction")
    print("   - 建模股票关系图")
    print("   - 搜索关键词: 'GNN stock prediction relation'")
    print()
    print("7. Large Language Models for Financial Forecasting")
    print("   - 2024-2025热点，LLM预测股价")
    print("   - 搜索关键词: 'LLM financial forecasting stock'")
    print()
    print("=" * 60)
    print("提示: 复制上面链接到浏览器查看论文详情")
    print("=" * 60)
