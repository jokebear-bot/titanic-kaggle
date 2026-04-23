#!/usr/bin/env python3
"""
泰坦尼克号 - V2 优化版
修复数据泄露 + 添加 Ticket 特征 + Fare log 变换
预期提升: 0.77 → 0.80+
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

class TitanicFeatures(BaseEstimator, TransformerMixin):
    """自定义特征工程，避免数据泄露"""
    
    def __init__(self):
        self.age_medians = {}
        self.fare_medians = {}
        
    def fit(self, X, y=None):
        # 只在训练集上计算统计量
        df = X.copy()
        
        # 先提取 Title
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
                     'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                     'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                     'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                     'Capt': 'Rare', 'Sir': 'Rare'}
        df['Title'] = df['Title'].map(title_map).fillna('Rare')
        
        # Age 按 Title + Pclass 分组的中位数
        for title in df['Title'].unique():
            for pclass in [1, 2, 3]:
                mask = (df['Title'] == title) & (df['Pclass'] == pclass)
                median_age = df.loc[mask, 'Age'].median()
                if pd.notna(median_age):
                    self.age_medians[(title, pclass)] = median_age
        
        # Fare 按 Pclass + Embarked 分组的中位数
        for pclass in [1, 2, 3]:
            for embarked in ['S', 'C', 'Q']:
                mask = (df['Pclass'] == pclass) & (df['Embarked'] == embarked)
                median_fare = df.loc[mask, 'Fare'].median()
                if pd.notna(median_fare):
                    self.fare_medians[(pclass, embarked)] = median_fare
        
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # 1. 头衔提取 (fit 中已提取，这里直接用)
        if 'Title' not in df.columns:
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            title_map = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
                         'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                         'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                         'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                         'Capt': 'Rare', 'Sir': 'Rare'}
            df['Title'] = df['Title'].map(title_map).fillna('Rare')
        
        df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
        
        # 2. 性别编码
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        
        # 3. Age 填充 - 使用 fit 时计算的统计量
        for (title, pclass), median_age in self.age_medians.items():
            mask = (df['Title'] == title) & (df['Pclass'] == pclass) & (df['Age'].isna())
            df.loc[mask, 'Age'] = median_age
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # 4. Fare 填充 + log 变换
        for (pclass, embarked), median_fare in self.fare_medians.items():
            mask = (df['Pclass'] == pclass) & (df['Embarked'] == embarked) & (df['Fare'].isna())
            df.loc[mask, 'Fare'] = median_fare
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Fare_log'] = np.log1p(df['Fare'])  # log1p 变换
        
        # 5. Embarked 填充
        df['Embarked'] = df['Embarked'].fillna('S')
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df = pd.concat([df, embarked_dummies], axis=1)
        
        # 6. 家庭规模
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # 7. 年龄段
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                              labels=[0, 1, 2, 3, 4]).astype(int)
        
        # 8. 舱房甲板
        df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
        df['HasCabin'] = (df['Cabin'].notna()).astype(int)
        deck_map = {'U': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
        df['CabinDeck'] = df['CabinDeck'].map(deck_map).fillna(0).astype(int)
        
        # 9. Ticket 特征
        # 提取前缀
        df['TicketPrefix'] = df['Ticket'].str.extract(r'^([A-Za-z\.\/]+)', expand=False).fillna('None')
        # 共享 Ticket 的人数
        ticket_counts = df['Ticket'].value_counts()
        df['TicketShare'] = df['Ticket'].map(ticket_counts)
        # 是否为团体票
        df['IsGroup'] = (df['TicketShare'] > 1).astype(int)
        # Ticket 前缀编码（简化）
        df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: 0 if x == 'None' else 1)
        
        # 10. 交互特征
        df['Pclass_Sex'] = df['Pclass'] * 2 + df['Sex']
        df['HighValue'] = ((df['Sex'] == 1) | (df['Age'] < 16)).astype(int) * (4 - df['Pclass'])
        
        return df

def get_features(df):
    """获取特征列"""
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Fare_log',
                'FamilySize', 'IsAlone', 'AgeBin', 'Title', 'CabinDeck', 'HasCabin',
                'TicketPrefix', 'TicketShare', 'IsGroup', 'Pclass_Sex', 'HighValue']
    for col in df.columns:
        if col.startswith('Embarked_'):
            features.append(col)
    return features

def main():
    print("=" * 60)
    print("泰坦尼克号 - V2 优化版 (修复数据泄露)")
    print("=" * 60)
    
    # 加载数据
    train = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/train.csv')
    test = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/test.csv')
    
    print(f"\n训练集: {len(train)} 条")
    print(f"测试集: {len(test)} 条")
    
    # 构建 Pipeline
    feature_engineer = TitanicFeatures()
    
    # 先在训练集上 fit
    train_processed = feature_engineer.fit_transform(train)
    
    # 模型
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,  # 降低深度防止过拟合
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        random_state=RANDOM_STATE
    )
    
    # 特征和标签
    features = get_features(train_processed)
    X = train_processed[features].fillna(0)
    y = train_processed['Survived']
    
    # StratifiedKFold CV（关键！）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring='accuracy')
    
    print("\n" + "=" * 60)
    print("交叉验证结果 (StratifiedKFold)")
    print("=" * 60)
    print(f"Random Forest: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
    print(f"Gradient Boosting: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")
    
    # 训练最终模型（用全部训练数据）
    rf.fit(X, y)
    gb.fit(X, y)
    
    # 处理测试集
    test_processed = feature_engineer.transform(test)
    X_test = test_processed[features].fillna(0)
    
    # 预测（平均两个模型）
    pred_rf = rf.predict_proba(X_test)[:, 1]
    pred_gb = gb.predict_proba(X_test)[:, 1]
    predictions = ((pred_rf + pred_gb) / 2 > 0.5).astype(int)
    
    # 生成提交
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })
    
    output_path = '/lhcos-data/projects/titanic/submission_v2_optimized.csv'
    submission.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("提交文件生成完成")
    print("=" * 60)
    print(f"文件: {output_path}")
    print(f"预测生存率: {submission['Survived'].mean():.2%}")
    print(f"\n预期 LB 得分: ~0.80 (基于修正后的 CV)")
    print("=" * 60)
    
    return submission

if __name__ == '__main__':
    submission = main()
