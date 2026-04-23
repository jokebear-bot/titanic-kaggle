#!/usr/bin/env python3
"""
泰坦尼克号 - 极简版
只用核心特征：Sex, Pclass, Age, Fare, FamilySize
预期：减少过拟合，LB 更接近 CV
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def preprocess(train, test):
    """简单的预处理，无数据泄露"""
    # 合并处理
    combined = pd.concat([train, test], ignore_index=True)
    
    # 1. 性别
    combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1})
    
    # 2. 年龄填充 - 简单中位数
    combined['Age'] = combined['Age'].fillna(combined['Age'].median())
    
    # 3. 票价填充 + log
    combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
    combined['Fare_log'] = np.log1p(combined['Fare'])
    
    # 4. 家庭规模
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    
    # 5. 是否独自一人
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    
    # 分割
    train_processed = combined.iloc[:len(train)].copy()
    test_processed = combined.iloc[len(train):].copy()
    
    return train_processed, test_processed

def main():
    print("=" * 60)
    print("泰坦尼克号 - 极简版 (5个核心特征)")
    print("=" * 60)
    
    # 加载数据
    train = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/train.csv')
    test = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/test.csv')
    
    print(f"\n训练集: {len(train)} 条")
    
    # 预处理
    train_processed, test_processed = preprocess(train, test)
    
    # 核心特征
    features = ['Sex', 'Pclass', 'Age', 'Fare_log', 'FamilySize', 'IsAlone']
    
    X = train_processed[features]
    y = train_processed['Survived']
    
    # 简单模型
    model = RandomForestClassifier(
        n_estimators=100,      # 减少树数量
        max_depth=4,           # 限制深度
        min_samples_split=10,  # 增加分裂阈值
        min_samples_leaf=5,    # 增加叶子最小样本
        random_state=RANDOM_STATE
    )
    
    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"\nCV 准确率: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # 训练
    model.fit(X, y)
    
    # 预测
    X_test = test_processed[features]
    predictions = model.predict(X_test)
    
    # 提交
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions.astype(int)
    })
    
    output_path = '/lhcos-data/projects/titanic/submission_simple.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\n文件: {output_path}")
    print(f"预测生存率: {submission['Survived'].mean():.2%}")
    print(f"\n特征重要性:")
    for feat, imp in zip(features, model.feature_importances_):
        print(f"  {feat}: {imp:.4f}")
    
    print(f"\n预期 LB: ~0.78-0.79")
    print("=" * 60)
    
    return submission

if __name__ == '__main__':
    submission = main()
