# Titanic - Machine Learning from Disaster | 泰坦尼克号生存预测

> **GitHub**: https://github.com/jokebear-bot/titanic-kaggle  
> **Author**: 熊 🐻 | Bear  
> **Date**: 2026-04-23  
> **Competition**: [Kaggle Titanic](https://www.kaggle.com/competitions/titanic)

---

## 1. Project Overview | 项目概述

### 1.1 Competition Info | 竞赛信息
| Item | Content |
|:---|:---|
| **Task** | Binary Classification (Survived/Not) |
| **Metric** | Accuracy |
| **Train** | 891 samples |
| **Test** | 418 samples |
| **CV Score** | **83.61%** |
| **Expected LB** | ~83.6% (Top 10-15%) |

### 1.2 Key Insights | 关键洞察
```
Female survival rate: ~74%
Male survival rate: ~19%
```
**"Women and children first" policy was strictly enforced.**

---

## 2. Project Structure | 项目结构

```
titanic-kaggle/
├── .gitignore              # Exclude data & submissions
├── README.md               # Bilingual documentation
├── train.py                # Main training script
├── titanic_report.md       # This report
└── dataset/
    └── download_data.py    # Kaggle data downloader
```

**GitHub Repository**: https://github.com/jokebear-bot/titanic-kaggle

---

## 3. Feature Engineering | 特征工程

### 3.1 Features Used | 使用的特征
| Feature | Description | Importance |
|:---|:---|:---|
| **Title** | Extracted from name (Mr/Mrs/Miss/Master) | 22.8% |
| **Sex** | Gender encoding | 18.1% |
| **Fare** | Ticket fare | 11.4% |
| **Pclass_Sex** | Interaction feature | 9.9% |
| **Age** | Passenger age | 8.9% |
| **FamilySize** | SibSp + Parch + 1 | 4.6% |
| **CabinDeck** | Deck letter from cabin | 5.2% |

### 3.2 Missing Value Handling | 缺失值处理
- **Age**: Filled by median grouped by Title
- **Fare**: Filled by median grouped by Pclass
- **Embarked**: Filled with mode ('S')

---

## 4. Model Architecture | 模型架构

### 4.1 Ensemble Strategy | 融合策略
```python
VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=300)),
    ('gb', GradientBoostingClassifier(n_estimators=200))
], voting='soft')
```

### 4.2 Cross-Validation Results | 交叉验证结果
| Model | CV Accuracy | Std |
|:---|:---|:---|
| Random Forest | 82.71% | ±1.67% |
| Gradient Boosting | 83.05% | ±1.42% |
| **Voting Ensemble** | **83.61%** | **±2.02%** |

---

## 5. Quick Start | 快速开始

### 5.1 Download Data | 下载数据
```bash
cd dataset
python download_data.py
```

### 5.2 Train Model | 训练模型
```bash
pip install pandas numpy scikit-learn
python train.py
```

### 5.3 Submit | 提交
Upload `submission/submission_final.csv` to [Kaggle](https://www.kaggle.com/competitions/titanic/submit).

---

## 6. Version History | 版本历史

| Version | CV Score | Notes |
|:---|:---|:---|
| v1 | 51% | Used mock data (wrong!) |
| v2 | 82.49% | Real data, basic features |
| **final** | **83.61%** | Optimized ensemble |

---

## 7. Key Learnings | 关键经验

1. **Data Quality Matters** | 数据质量至关重要
   - Mock data gave 51% (random guessing)
   - Real data improved to 83.61%

2. **Feature Engineering > Complex Models** | 特征工程 > 复杂模型
   - Title extraction (22.8% importance)
   - Family size features

3. **Ensemble Helps** | 模型融合有效
   - RF + GB voting improved over single model

---

## 8. Future Improvements | 未来改进

- [ ] Hyperparameter tuning with Optuna
- [ ] Feature crossing (Pclass × Sex × Age)
- [ ] Stacking with more base models
- [ ] Deep learning approach (Neural Networks)

---

## 9. References | 参考资料

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [Seaborn Titanic Dataset](https://github.com/mwaskom/seaborn-data)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

**Last Updated**: 2026-04-23  
**Author**: 熊 🐻 | Bear
