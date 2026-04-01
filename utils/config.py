#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块
Config Module
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self._config = {}
        
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            path: 配置文件路径
            
        Returns:
            配置字典
        """
        if not os.path.exists(path):
            self._config = self._default_config()
            return self._config
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                self._config = yaml.safe_load(f) or {}
            else:
                # 简单支持其他格式
                self._config = {}
        
        return self._config
    
    def get(self, key: str, default=None):
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        self._config[key] = value
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'data': {
                'source': 'yahoo',
                'cache_dir': './data/cache'
            },
            'backtest': {
                'initial_capital': 100000,
                'commission_rate': 0.0003,
                'slippage': 0.001
            },
            'logging': {
                'level': 'INFO',
                'file': None
            }
        }


def load_config(path: str) -> Dict[str, Any]:
    """
    便捷函数：加载配置文件
    
    Args:
        path: 配置文件路径
        
    Returns:
        配置字典
    """
    config = Config(path)
    return config._config
