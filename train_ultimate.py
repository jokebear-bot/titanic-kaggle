#!/usr/bin/env python3
"""
泰坦尼克号 - 终极版 (目标: 0.82+)
包含: Group Survival, Ticket Frequency, CatBoost
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def extract_surname(name):
    """提取姓氏"""
    return name.split(',')[0].strip()

def preprocess(train, test):
    """完整的特征工程"""
    # 保存原始索引
    train_len = len(train)
    
    # 合并处理
    combined = pd.concat([train, test], ignore_index=True)
    
    # 1. 基础编码
    combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1})
    
    # 2. 头衔提取
    combined['Title'] = combined['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
                 'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                 'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                 'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                 'Capt': 'Rare', 'Sir': 'Rare'}
    combined['Title'] = combined['Title'].map(title_map).fillna('Rare')
    combined['Title'] = combined['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
    
    # 3. Age 按 Title + Pclass 填充
    for title in [0, 1, 2, 3, 4]:
        for pclass in [1, 2, 3]:
            mask = (combined['Title'] == title) & (combined['Pclass'] == pclass)
            median_age = combined.loc[mask, 'Age'].median()
            if pd.notna(median_age):
                combined.loc[mask & combined['Age'].isna(), 'Age'] = median_age
    combined['Age'] = combined['Age'].fillna(combined['Age'].median())
    
    # 4. Fare 填充 + log
    combined['Fare'] = combined['Fare'].fillna(combined.groupby('Pclass')['Fare'].transform('median'))
    combined['Fare_log'] = np.log1p(combined['Fare'])
    
    # 5. Embarked
    combined['Embarked'] = combined['Embarked'].fillna('S')
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    
    # 6. Cabin 甲板
    combined['CabinDeck'] = combined['Cabin'].str[0].fillna('U')
    combined['HasCabin'] = (combined['Cabin'].notna()).astype(int)
    deck_map = {'U': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    combined['CabinDeck'] = combined['CabinDeck'].map(deck_map).fillna(0).astype(int)
    
    # 7. 家庭规模
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    
    # 8. Ticket Frequency (关键特征)
    ticket_counts = combined['Ticket'].value_counts()
    combined['TicketFreq'] = combined['Ticket'].map(ticket_counts)
    
    # 9. 姓氏提取
    combined['Surname'] = combined['Name'].apply(extract_surname)
    
    # 10. Group Survival (高分关键)
    # 基于 Ticket 和 Surname 定义群体
    combined['GroupId'] = combined['Ticket'] + '_' + combined['Surname']
    
    # 分割回 train/test
    train_processed = combined.iloc[:train_len].copy()
    test_processed = combined.iloc[train_len:].copy()
    
    # 计算群体存活率 (只在训练集上计算)
    group_survival = {}
    for group_id in train_processed['GroupId'].unique():
        group_data = train_processed[train_processed['GroupId'] == group_id]
        if len(group_data) > 1:  # 只有群体才计算
            survival_rate = group_data['Survived'].mean()
            group_survival[group_id] = survival_rate
    
    # 映射到训练集和测试集
    train_processed['GroupSurvival'] = train_processed['GroupId'].map(group_survival)
    test_processed['GroupSurvival'] = test_processed['GroupId'].map(group_survival)
    
    # 填充缺失（单人没有群体存活率）
    train_processed['GroupSurvival'] = train_processed['GroupSurvival'].fillna(0.5)
    test_processed['GroupSurvival'] = test_processed['GroupSurvival'].fillna(0.5)
    
    # 11. 交互特征
    combined = pd.concat([train_processed, test_processed], ignore_index=True)
    combined['Pclass_Sex'] = combined['Pclass'] * 2 + combined['Sex']
    combined['Age_Fare'] = combined['Age'] * combined['Fare_log'] / 100
    
    # 分割
    train_processed = combined.iloc[:train_len].copy()
    test_processed = combined.iloc[train_len:].copy()
    
    return train_processed, test_processed

def main():
    print("=" * 60)
    print("泰坦尼克号 - 终极版 (Group Survival + Ticket Freq)")
    print("=" * 60)
    
    # 加载数据
    train = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/train.csv')
    test = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/test.csv')
    
    print(f"\n训练集: {len(train)} 条")
    
    # 预处理
    train_processed, test_processed = preprocess(train, test)
    
    # 特征
    features = ['Pclass', 'Sex', 'Age', 'Fare_log', 'Title', 'CabinDeck', 'HasCabin',
                'FamilySize', 'IsAlone', 'TicketFreq', 'GroupSurvival', 'Pclass_Sex', 'Age_Fare']
    for col in train_processed.columns:
        if col.startswith('Embarked_'):
            features.append(col)
    
    print(f"使用 {len(features)} 个特征")
    
    X = train_processed[features].fillna(0)
    y = train_processed['Survived']
    
    # 模型
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    
    # StratifiedKFold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"\nCV 准确率: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # 训练
    model.fit(X, y)
    
    # 预测
    X_test = test_processed[features].fillna(0)
    predictions = model.predict(X_test)
    
    # 提交
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions.astype(int)
    })
    
    output_path = '/lhcos-data/projects/titanic/submission_ultimate.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\n文件: {output_path}")
    print(f"预测生存率: {submission['Survived'].mean():.2%}")
    print(f"\n特征重要性 (Top 8):")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for feat, imp in importances.head(8).items():
        print(f"  {feat}: {imp:.4f}")
    
    print(f"\n预期 LB: ~0.80-0.82")
    print("=" * 60)
    
    return submission

if __name__ == '__main__':
    submission = main()
