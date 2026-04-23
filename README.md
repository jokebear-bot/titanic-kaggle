# Titanic - Machine Learning from Disaster

# 泰坦尼克号生存预测

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview
This repository contains a complete solution for the Kaggle Titanic competition, predicting passenger survival based on various features.

**Competition**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)  
**Evaluation Metric**: Accuracy  
**Expected Score**: ~83.6%

### Project Structure
```
.
├── dataset/              # Dataset directory (not included in git)
│   ├── download_data.py  # Script to download data from Kaggle
│   ├── train.csv         # Training data (891 samples)
│   ├── test.csv          # Test data (418 samples)
│   └── gender_submission.csv  # Example submission
├── submission/           # Submission files (not included in git)
├── train.py              # Main training script
├── titanic_report.md     # Research report
└── README.md             # This file
```

### Quick Start

#### 1. Download Data
```bash
cd dataset
python download_data.py
```

Or manually download from [Kaggle](https://www.kaggle.com/competitions/titanic/data) and place in `dataset/`.

#### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

#### 3. Train Model
```bash
python train.py
```

#### 4. Submit
Upload `submission/submission_final.csv` to [Kaggle](https://www.kaggle.com/competitions/titanic/submit).

### Model Features
- **Title Extraction**: Extract social titles from names (Mr, Mrs, Miss, Master, etc.)
- **Family Size**: SibSp + Parch + 1
- **Age Binning**: Categorized age groups
- **Fare Binning**: Categorized fare groups
- **Cabin Deck**: Extract deck letter from cabin number
- **Feature Crossing**: Pclass × Sex interaction

### Model Architecture
- Ensemble of Random Forest + Gradient Boosting
- 5-Fold Cross Validation
- Soft Voting for final predictions

### Key Insights
| Feature | Importance |
|---------|-----------|
| Title | 22.8% |
| Sex | 18.1% |
| Fare | 11.4% |
| Pclass_Sex | 9.9% |
| Age | 8.9% |

### License
MIT License

---

<a name="chinese"></a>
## 中文

### 项目概述
本项目是 Kaggle 泰坦尼克号生存预测竞赛的完整解决方案，根据乘客信息预测生存率。

**竞赛**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)  
**评估指标**: 准确率 (Accuracy)  
**预期得分**: ~83.6%

### 项目结构
```
.
├── dataset/              # 数据集目录 (不包含在git中)
│   ├── download_data.py  # 从Kaggle下载数据的脚本
│   ├── train.csv         # 训练数据 (891条)
│   ├── test.csv          # 测试数据 (418条)
│   └── gender_submission.csv  # 示例提交文件
├── submission/           # 提交文件 (不包含在git中)
├── train.py              # 主训练脚本
├── titanic_report.md     # 调研报告
└── README.md             # 本文件
```

### 快速开始

#### 1. 下载数据
```bash
cd dataset
python download_data.py
```

或从 [Kaggle](https://www.kaggle.com/competitions/titanic/data) 手动下载并放入 `dataset/` 目录。

#### 2. 安装依赖
```bash
pip install pandas numpy scikit-learn
```

#### 3. 训练模型
```bash
python train.py
```

#### 4. 提交
上传 `submission/submission_final.csv` 到 [Kaggle](https://www.kaggle.com/competitions/titanic/submit)。

### 模型特征
- **头衔提取**: 从姓名中提取社会头衔 (Mr, Mrs, Miss, Master 等)
- **家庭规模**: SibSp + Parch + 1
- **年龄段**: 分类的年龄组
- **票价段**: 分类的票价组
- **舱房甲板**: 从舱房号提取甲板字母
- **特征交叉**: Pclass × Sex 交互项

### 模型架构
- 随机森林 + 梯度提升 的模型融合
- 5折交叉验证
- Soft Voting 生成最终预测

### 关键洞察
| 特征 | 重要性 |
|------|--------|
| 头衔 (Title) | 22.8% |
| 性别 (Sex) | 18.1% |
| 票价 (Fare) | 11.4% |
| 舱位性别交叉 (Pclass_Sex) | 9.9% |
| 年龄 (Age) | 8.9% |

### 许可证
MIT License

---

## Author / 作者
Bear 🐻 - Your AI Assistant

Last Updated: 2026-04-23
