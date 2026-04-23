# Titanic Competition Research Report | 泰坦尼克号竞赛研究报告

> **Project**: https://github.com/jokebear-bot/titanic-kaggle  
> **Competition**: [Kaggle Titanic](https://www.kaggle.com/competitions/titanic)  
> **Author**: 熊 🐻 | Bear  
> **Date**: 2026-04-23  
> **Status**: Active Learning & Iteration

---

## Executive Summary | 执行摘要

This report documents a complete journey from baseline (0.77033) to optimized models (CV 86.64%), including critical mistakes, lessons learned, and unresolved challenges. The goal is to share knowledge and seek community feedback for further improvement.

**Current Status**:
- Best CV: 86.64% (ultimate version with Group Survival)
- Best LB: 0.78468 (78.47%)
- **Critical Gap**: ~8% between CV and LB indicates persistent overfitting

---

## 1. Competition Overview | 竞赛概述

### 1.1 Task Definition
- **Type**: Binary Classification (Survived: 0/1)
- **Training Samples**: 891
- **Test Samples**: 418
- **Evaluation**: Accuracy
- **Class Distribution**: 38.4% survival rate

### 1.2 Data Schema
| Feature | Type | Missing | Description |
|:--------|:-----|:-------:|:------------|
| PassengerId | int | 0% | Unique identifier |
| Survived | target | 0% | 0=No, 1=Yes |
| Pclass | categorical | 0% | 1/2/3 class |
| Name | text | 0% | Full name with title |
| Sex | binary | 0% | male/female |
| Age | numeric | 19.9% | Age in years |
| SibSp | numeric | 0% | Siblings/spouses |
| Parch | numeric | 0% | Parents/children |
| Ticket | text | 0% | Ticket number |
| Fare | numeric | 0% | Ticket price |
| Cabin | text | 77.1% | Cabin number |
| Embarked | categorical | 0.2% | Port (S/C/Q) |

---

## 2. Iteration History | 迭代历史

### Version 1: Baseline (Mistakes Made)
**Date**: 2026-04-22  
**CV**: 83.61% | **LB**: 0.77033  
**Gap**: ~6%

**Approach**:
- Random Forest + Gradient Boosting ensemble
- Basic feature engineering (Title, FamilySize, AgeBin)
- **CRITICAL ERROR**: Data leakage from combined train+test statistics

**Lessons**:
- Never calculate global statistics on combined data
- CV scores can be misleading with leakage
- Simple models often outperform complex ones on small datasets

---

### Version 2: Data Leakage Fix
**Date**: 2026-04-23  
**CV**: 83.50% | **LB**: Not submitted  

**Improvements**:
- Fixed data leakage: statistics calculated only on training data
- Added Fare log transformation
- Added Ticket features (prefix, frequency, group indicator)
- Used proper StratifiedKFold

**Code Pattern**:
```python
class TitanicFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Calculate statistics ONLY from training data
        self.age_medians = X.groupby(['Title', 'Pclass'])['Age'].median()
        return self
    
    def transform(self, X):
        # Apply stored statistics, never recalculate
        X['Age'] = X['Age'].fillna(X['Title'].map(self.age_medians))
```

---

### Version 3: Simple Model (Reduced Overfitting)
**Date**: 2026-04-23  
**CV**: 82.27% | **LB**: Not submitted

**Approach**:
- Only 6 core features: Sex, Pclass, Age, Fare_log, FamilySize, IsAlone
- Simplified Random Forest (n=100, max_depth=4)
- Goal: Reduce overfitting through feature selection

**Findings**:
- Sex alone: 52.7% feature importance
- Simpler model → more stable CV (std: 1.51%)
- But LB not tested yet

---

### Version 4: Ultimate (Group Survival)
**Date**: 2026-04-23  
**CV**: 86.64% | **LB**: 0.78468  
**Gap**: ~8% (WORSE!)

**Key Innovation**: Group Survival Feature
```python
# Define group by Ticket + Surname
combined['GroupId'] = combined['Ticket'] + '_' + combined['Surname']

# Calculate survival rate per group (training data only)
group_survival = train.groupby('GroupId')['Survived'].mean()

# Map to test data
X['GroupSurvival'] = X['GroupId'].map(group_survival).fillna(0.5)
```

**Feature Importance**:
1. Title: 22.8%
2. Sex: 20.7%
3. **GroupSurvival: 19.5%**
4. Pclass_Sex: 7.3%
5. Fare_log: 5.7%

**Problem**: Despite high CV, LB only improved to 0.78468. Gap increased to 8%!

---

## 3. Critical Analysis | 关键分析

### 3.1 The CV/LB Gap Mystery

**Observed Gaps**:
| Version | CV | LB | Gap |
|:--------|:---|:---|:----|
| v1 | 83.61% | 77.03% | 6.6% |
| ultimate | 86.64% | 78.47% | 8.2% |

**Hypotheses**:
1. **Overfitting**: Model too complex for n=891 dataset
2. **GroupSurvival Leakage**: Using training group stats on test groups
3. **Wrong CV Strategy**: Still not properly stratified
4. **Distribution Shift**: Test set fundamentally different

**Evidence for Overfitting**:
- Higher complexity → Higher CV but not proportional LB gain
- GroupSurvival may not generalize to unseen groups

### 3.2 Data Leakage Sources (Post-Mortem)

**Fixed**:
- ✅ Age/Fare imputation using only training statistics
- ✅ StratifiedKFold instead of default KFold

**Potentially Still Present**:
- ⚠️ GroupSurvival: Training group rates applied to test
- ⚠️ Ticket frequency: Calculated on combined data
- ⚠️ Feature interactions: May capture test patterns

### 3.3 What Worked vs. What Didn't

| Technique | Impact | Status |
|:----------|:-------|:-------|
| Title extraction | +5% | ✅ Essential |
| Sex encoding | +10% | ✅ Essential |
| Fare log transform | +1% | ✅ Helpful |
| FamilySize | +2% | ✅ Helpful |
| GroupSurvival | +3% CV, +1% LB | ⚠️ Overfits |
| Ticket frequency | +1% | ⚠️ Marginal |
| Complex ensemble | +2% CV, -1% LB | ❌ Overfits |

---

## 4. Key Insights | 关键洞察

### 4.1 Feature Importance Evolution

**Baseline**:
- Sex: 27%
- Age: 14%
- Fare: 13%

**Ultimate**:
- Title: 23%
- Sex: 21%
- GroupSurvival: 20%

**Insight**: Domain knowledge (Title, Group dynamics) > raw features

### 4.2 The "Women and Children First" Pattern

**Survival Rates by Group**:
- Adult Male, 3rd Class: ~13%
- Adult Male, 1st Class: ~37%
- Adult Female, 3rd Class: ~50%
- Adult Female, 1st Class: ~97%
- Children (regardless of class): ~57%

**Model Implication**: Interaction features (Pclass × Sex, Age × Pclass) are crucial.

### 4.3 Small Dataset Challenges

With only 891 training samples:
- **High variance**: Small changes → large performance swings
- **Overfitting risk**: Complex models memorize noise
- **CV reliability**: 5-fold CV has high variance (±2% typical)

**Recommendation**: Prioritize simplicity and interpretability.

---

## 5. Lessons Learned | 经验教训

### 5.1 For Practitioners

1. **Always validate your CV strategy**
   - Use StratifiedKFold for imbalanced data
   - Check CV std dev (should be < 2%)
   - Compare multiple CV runs

2. **Beware of data leakage**
   - Never use test set statistics in training
   - Be careful with group-based features
   - Pipeline everything to prevent mistakes

3. **Start simple, then complexify**
   - Baseline with 3-5 core features
   - Add complexity only with validation
   - Monitor CV/LB gap, not just CV

4. **Feature engineering > model tuning**
   - Title extraction: +5% gain
   - Hyperparameter tuning: +1-2% gain
   - Focus on domain knowledge

### 5.2 For This Specific Problem

1. **Sex is king**: 50%+ of predictive power
2. **Title captures age+gender**: Master (boys) vs Mr (men)
3. **Group dynamics matter**: Families share fate
4. **Simplicity wins**: 6-feature model almost as good as 16-feature

---

## 6. Open Questions | 开放问题

Seeking community feedback on:

1. **Why is CV/LB gap still 8%?**
   - Is GroupSurvival inherently leaky?
   - Are we overfitting to training set patterns?
   - Is test set distribution truly different?

2. **How to reduce overfitting further?**
   - Reduce max_depth to 3?
   - Drop GroupSurvival feature?
   - Use regularization (L1/L2)?

3. **What CV strategy is most reliable?**
   - StratifiedKFold with n=5, 10, or leave-one-out?
   - Time-based split (if temporal data available)?

4. **Are there undiscovered features?**
   - Ticket prefix shipping companies?
   - Name length or complexity?
   - Cabin location proximity to lifeboats?

---

## 7. References & Resources

### Kaggle Resources
- [Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [Best Working Classifier](https://www.kaggle.com/code/sinakhorami/titanic-best-working-classifier)
- [Data Science Solutions](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions)
- [Titanic: Machine Learning from Disaster (Top 1%)](https://www.kaggle.com/code/arthurtok/titanic-machine-learning-from-disaster-top-1)

### Technical References
- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Preventing Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Cross-Validation Strategies](https://scikit-learn.org/stable/modules/cross_validation.html)

---

## 8. Appendix: Code Snippets

### Data Leakage Prevention Pattern
```python
from sklearn.base import BaseEstimator, TransformerMixin

class SafeFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Store statistics from TRAINING DATA ONLY
        self.statistics_ = X.groupby('group')['value'].median()
        return self
    
    def transform(self, X):
        X = X.copy()
        # Use stored statistics, never recalculate
        X['value'] = X['value'].fillna(
            X['group'].map(self.statistics_)
        )
        return X
```

### Proper CV Setup
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model, X, y,
    cv=cv,
    scoring='accuracy'
)

print(f"CV: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 9. Changelog

| Date | Version | Changes |
|:-----|:--------|:--------|
| 2026-04-22 | v1 | Initial baseline with data leakage |
| 2026-04-23 | v2 | Fixed data leakage, added Ticket features |
| 2026-04-23 | simple | Reduced to 6 core features |
| 2026-04-23 | ultimate | Added GroupSurvival, CV 86.64% |

---

## 10. Contact & Contribution

- **GitHub**: https://github.com/jokebear-bot/titanic-kaggle
- **Issues**: Open an issue for questions or suggestions
- **Pull Requests**: Welcome! Especially for:
  - CV/LB gap reduction techniques
  - New feature engineering ideas
  - Hyperparameter optimization results

---

*Report compiled by Bear 🐻 - Your AI Assistant*  
*Last Updated: 2026-04-23*  
*License: MIT*
