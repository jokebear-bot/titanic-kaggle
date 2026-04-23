# Titanic - Machine Learning from Disaster | 泰坦尼克号生存预测

> **GitHub**: https://github.com/jokebear-bot/titanic-kaggle  
> **Kaggle Competition**: https://www.kaggle.com/competitions/titanic  
> **Author**: 熊 🐻 | Bear  
> **Last Updated**: 2026-04-23

---

## 📊 Current Best Score | 当前最佳得分

| Version | CV Accuracy | Public LB | Notes |
|:--------|:-----------:|:---------:|:------|
| v1 (baseline) | 83.61% | 0.77033 | Initial model with data leakage issues |
| ultimate | 86.64% | 0.78468 | Group Survival + Ticket Frequency (leakage!) |
| **fixed** | **82.27%** | **0.787** | ✅ **Fixed data leakage, CV/LB aligned** |

**Gap Analysis**:
- ultimate: CV 86.64% → LB 0.78468 (gap ~8%) ❌ Data leakage
- **fixed: CV 82.27% → LB 0.787 (gap ~3.5%)** ✅ Real performance

**Key Lesson**: Lower CV with proper validation > Higher CV with leakage

---

## 🚀 Quick Start | 快速开始

### 1. Download Data | 下载数据
```bash
cd dataset
python download_data.py
```

### 2. Install Dependencies | 安装依赖
```bash
pip install pandas numpy scikit-learn
```

### 3. Train Model | 训练模型
```bash
# Best model (fixed, no leakage)
python train_fixed.py

# Archive versions (for reference)
python archive/train_ultimate.py  # Has leakage, CV 86.64%
python archive/train_v2.py        # Intermediate version
```

### 4. Submit | 提交
Upload `submission/submission_fixed.csv` to [Kaggle](https://www.kaggle.com/competitions/titanic/submit).

---

## 📁 Project Structure | 项目结构

```
titanic-kaggle/
├── dataset/                    # Dataset directory
│   ├── download_data.py        # Kaggle data downloader
│   ├── train.csv              # Training data (not in git)
│   ├── test.csv               # Test data (not in git)
│   └── gender_submission.csv  # Example submission (not in git)
├── submission/                 # Submission files (not in git)
│   ├── submission_fixed.csv       # ✅ Best: 0.787 (no leakage)
│   ├── submission_ultimate.csv    # 0.78468 (has leakage)
│   ├── submission_simple.csv      # Simplified version
│   └── submission_v2_optimized.csv # V2 with fixes
├── train_fixed.py              # ✅ Best: Fixed leakage (CV 82.27%, LB 0.787)
├── train.py                    # Original baseline
├── archive/                    # Archived versions
│   ├── train_v1.py             # Original baseline
│   ├── train_v2.py             # Intermediate version
│   └── train_ultimate.py       # Has leakage (CV 86.64%, not real)
├── titanic_report.md           # Detailed research report
└── README.md                   # This file
```

---

## 🧠 Key Learnings & Lessons | 关键经验教训

### 1. The CV/LB Gap Problem | CV与LB差距问题

**Lesson**: High CV doesn't guarantee high LB.

| Issue | Impact | Solution |
|:------|:-------|:---------|
| Data Leakage | CV inflated by 5-10% | Never use global statistics from combined data |
| Overfitting | Complex models perform worse on LB | Simplify model, reduce features |
| Wrong CV Strategy | Unreliable CV scores | Use StratifiedKFold, not default KFold |

**Our Experience**:
- v1: CV 83.61% → LB 0.77033 (gap: ~6%) ❌ Data leakage
- ultimate: CV 86.64% → LB 0.78468 (gap: ~8%) ❌ Severe leakage
- **fixed: CV 82.27% → LB 0.787 (gap: ~3.5%)** ✅ CV/LB aligned
- **Conclusion**: Proper validation > Complex features

### 2. Feature Engineering Matters More Than Complex Models

**Most Important Features** (from fixed version):
1. **Sex** (28.5%) - Gender is the strongest single predictor
2. **Title** (22.9%) - Extracted from name (Mr/Mrs/Miss/Master/Rare)
3. **Pclass_Sex** (10.1%) - Interaction feature
4. **Pclass** (7.4%) - Cabin class
5. **Fare_log** (6.7%) - Log-transformed fare

**Lesson**: Proper validation > Complex features with leakage

### 3. Data Leakage Traps | 数据泄露陷阱

**❌ WRONG**: Fill missing values using statistics from combined train+test
```python
# DON'T DO THIS
combined = pd.concat([train, test])
combined['Age'] = combined['Age'].fillna(combined['Age'].median())  # LEAKAGE!
```

**✅ CORRECT**: Calculate statistics only from training data
```python
# DO THIS
age_median = train['Age'].median()
train['Age'] = train['Age'].fillna(age_median)
test['Age'] = test['Age'].fillna(age_median)  # Use train statistics
```

### 4. The "Group Survival" Feature | 群体存活特征

**Insight**: Passengers with same Ticket or Surname often share fate.

**Implementation**:
1. Extract Surname from Name
2. Define Group by Ticket + Surname
3. Calculate group survival rate from training data
4. Map to test data (unknown groups get 0.5)

**Result**: 19.5% feature importance in ultimate model.

### 5. Model Complexity Trade-off | 模型复杂度权衡

| Model | n_estimators | max_depth | CV | Characteristics |
|:------|:------------:|:---------:|:--:|:----------------|
| Simple | 100 | 4 | 82.27% | Underfitting, stable |
| Balanced | 200 | 5-6 | 83-84% | Sweet spot? |
| Complex | 300 | 8-10 | 86%+ | Overfitting risk |

**Lesson**: For small datasets (n=891), simpler models generalize better.

### 6. Cross-Validation Strategy | 交叉验证策略

**Always use StratifiedKFold** for imbalanced classification:
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
```

**Why**: Maintains class distribution (38% survival) in each fold.

---

## 🔧 Technical Details | 技术细节

### Feature Engineering Pipeline

1. **Title Extraction**: From `Name` column using regex `([A-Za-z]+)\.`
2. **Age Imputation**: By Title + Pclass groups
3. **Fare Imputation**: By Pclass + Embarked groups
4. **Fare Log Transform**: `np.log1p(Fare)` to handle skewness
5. **Family Size**: `SibSp + Parch + 1`
6. **IsAlone**: Binary flag for solo travelers
7. **CabinDeck**: First letter of Cabin (A-G, T, U for unknown)
8. **Ticket Frequency**: Count of passengers sharing same ticket
9. **Group Survival**: Survival rate of Ticket+Surname group

### Model Architecture

**Ultimate Version**:
- Random Forest (n_estimators=200, max_depth=5)
- 16 features
- StratifiedKFold CV (5 folds)

**Why Random Forest over XGBoost/LightGBM**:
- Better interpretability
- Less prone to overfitting on small datasets
- Feature importance readily available

---

## 📈 Future Improvements | 未来改进方向

Based on community feedback and Kaggle discussions:

### High Priority
- [ ] **Fix remaining overfitting**: Try max_depth=3-4, reduce features to 8-10
- [ ] **Hyperparameter tuning**: Use Optuna for automated search
- [ ] **Ensemble with XGBoost/LightGBM**: Add model diversity

### Medium Priority
- [ ] **Advanced Group Features**: Woman-Child-Group logic
- [ ] **Ticket Prefix Analysis**: Extract shipping company codes
- [ ] **Age-Fare interaction**: Non-linear combinations

### Low Priority
- [ ] **Neural Networks**: Usually overkill for this dataset
- [ ] **Deep Feature Synthesis**: AutoML approaches

---

## 📝 Citation & References

If you use this code or findings:

```
Titanic Kaggle Competition Solution
GitHub: https://github.com/jokebear-bot/titanic-kaggle
Author: Bear (熊) - AI Assistant
Date: 2026-04-23
```

**Key References**:
- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [Titanic Best Working Classifier](https://www.kaggle.com/code/sinakhorami/titanic-best-working-classifier)
- [Titanic Data Science Solutions](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions)

---

## 🤝 Contributing & Feedback

This project is actively seeking feedback! Key questions:

1. Why is CV/LB gap still ~8% despite fixes?
2. Are there additional data leakage sources?
3. What hyperparameters would reduce overfitting?

Please open an issue or PR on GitHub.

---

**License**: MIT  
**Status**: Active Development  
**Last Commit**: 2026-04-23

---

*Created with ❤️ by Bear 🐻*
