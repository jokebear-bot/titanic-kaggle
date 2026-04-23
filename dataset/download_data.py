#!/usr/bin/env python3
"""
Download Titanic dataset from Kaggle
从Kaggle下载泰坦尼克数据集

Usage:
  python download_data.py

Requirements:
  - Kaggle API token configured in ~/.kaggle/kaggle.json
  - Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables
"""

import os
import sys

def download():
    """Download Titanic competition data from Kaggle"""
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system("pip install kagglehub -q")
        import kagglehub
    
    print("Downloading Titanic dataset from Kaggle...")
    print("正在从Kaggle下载泰坦尼克数据集...")
    
    try:
        path = kagglehub.competition_download('titanic')
        print(f"\n✓ Downloaded to: {path}")
        print(f"✓ 下载完成: {path}")
        
        # List files
        import subprocess
        result = subprocess.run(['ls', '-la', path], capture_output=True, text=True)
        print("\nFiles in dataset:")
        print(result.stdout)
        
        return path
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease make sure you have configured Kaggle API credentials:")
        print("请确保已配置Kaggle API凭证:")
        print("  1. Go to https://www.kaggle.com/settings/account")
        print("  2. Click 'Create New API Token'")
        print("  3. Place kaggle.json in ~/.kaggle/")
        sys.exit(1)

if __name__ == '__main__':
    download()
