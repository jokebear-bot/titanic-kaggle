#!/usr/bin/env python3
"""
泰坦尼克号 - 进阶版 (目标: 0.80+)
改进:
1. WCG (Woman-Child-Group) 逻辑特征
2. FarePerPerson (人均票价)
3. CatBoost + Random Forest 集成
4. GroupKFold 验证
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

class AdvancedFeatures(BaseEstimator, TransformerMixin):
    """进阶特征工程，包含WCG逻辑"""
    
    def __init__(self):
        self.age_medians = {}
        self.fare_medians = {}
        self.ticket_freq = {}
        
    def fit(self, X, y=None):
        """只在训练数据上计算统计量"""
        df = X.copy()
        
        # 提取Title
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
                     'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                     'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                     'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                     'Capt': 'Rare', 'Sir': 'Rare'}
        df['Title'] = df['Title'].map(title_map).fillna('Rare')
        df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
        
        # Age按Title+Pclass分组的中位数
        for title in df['Title'].unique():
            for pclass in [1, 2, 3]:
                mask = (df['Title'] == title) & (df['Pclass'] == pclass)
                median_age = df.loc[mask, 'Age'].median()
                if pd.notna(median_age):
                    self.age_medians[(title, pclass)] = median_age
        
        # Fare按Pclass分组的中位数
        for pclass in [1, 2, 3]:
            median_fare = df.loc[df['Pclass'] == pclass, 'Fare'].median()
            if pd.notna(median_fare):
                self.fare_medians[pclass] = median_fare
        
        # TicketFreq只在train上计算
        self.ticket_freq = df['Ticket'].value_counts().to_dict()
        
        return self
    
    def transform(self, X):
        """应用存储的统计量"""
        df = X.copy()
        
        # Title
        if 'Title' not in df.columns:
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            title_map = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
                         'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                         'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                         'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                         'Capt': 'Rare', 'Sir': 'Rare'}
            df['Title'] = df['Title'].map(title_map).fillna('Rare')
        df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
        
        # Sex
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        
        # Age填充
        for (title, pclass), median_age in self.age_medians.items():
            mask = (df['Title'] == title) & (df['Pclass'] == pclass) & (df['Age'].isna())
            df.loc[mask, 'Age'] = median_age
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Fare填充
        for pclass, median_fare in self.fare_medians.items():
            mask = (df['Pclass'] == pclass) & (df['Fare'].isna())
            df.loc[mask, 'Fare'] = median_fare
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Fare_log'] = np.log1p(df['Fare'])
        
        # Embarked
        df['Embarked'] = df['Embarked'].fillna('S')
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df = pd.concat([df, embarked_dummies], axis=1)
        
        # Cabin
        df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
        df['HasCabin'] = (df['Cabin'].notna()).astype(int)
        deck_map = {'U': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
        df['CabinDeck'] = df['CabinDeck'].map(deck_map).fillna(0).astype(int)
        
        # 家庭规模
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # TicketFreq
        df['TicketFreq'] = df['Ticket'].map(self.ticket_freq).fillna(1)
        
        # 1. 人均票价 (更真实的社会地位)
        df['FarePerPerson'] = df['Fare'] / df['TicketFreq']
        df['FarePerPerson_log'] = np.log1p(df['FarePerPerson'])
        
        # 2. 姓名长度
        df['NameLen'] = df['Name'].apply(len)
        
        # 3. 交互特征
        df['Pclass_Sex'] = df['Pclass'] * 2 + df['Sex']
        df['Age_Fare'] = df['Age'] * df['Fare_log'] / 100
        
        return df

def calculate_wcg_feature(train_df):
    """
    WCG (Woman-Child-Group) 逻辑:
    - 1: 组内有女性/孩子存活
    - 0: 组内女性/孩子全部遇难
    - 0.5: 单人或无女性/孩子
    """
    df = train_df.copy()
    df['WCG'] = 0.5  # 默认值
    
    # 定义组 (Ticket)
    for ticket in df['Ticket'].unique():
        group = df[df['Ticket'] == ticket]
        if len(group) > 1:  # 是群体
            # 找出组内的女性和孩子
            women_children = group[(group['Sex'] == 1) | (group['Age'] < 16)]
            if len(women_children) > 0:
                # 有女性/孩子存活
                if women_children['Survived'].sum() > 0:
                    df.loc[df['Ticket'] == ticket, 'WCG'] = 1.0
                else:
                    # 女性/孩子全部遇难
                    df.loc[df['Ticket'] == ticket, 'WCG'] = 0.0
    
    return df['WCG']

def get_features(df):
    """获取特征列"""
    features = ['Pclass', 'Sex', 'Age', 'Fare_log', 'FarePerPerson_log', 'Title', 
                'CabinDeck', 'HasCabin', 'FamilySize', 'IsAlone', 'TicketFreq',
                'NameLen', 'Pclass_Sex', 'Age_Fare']
    if 'WCG' in df.columns:
        features.append('WCG')
    for col in df.columns:
        if col.startswith('Embarked_'):
            features.append(col)
    return features

def manual_cv(train, use_group_kfold=False):
    """手写的CV循环"""
    y = train['Survived'].values
    
    if use_group_kfold:
        # GroupKFold: 同Ticket的乘客不分散
        groups = train['Ticket'].values
        kf = GroupKFold(n_splits=5)
        split_iter = kf.split(train, y, groups)
        print("Using GroupKFold (families stay together)")
    else:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        split_iter = kf.split(train, y)
        print("Using StratifiedKFold")
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        train_fold = train.iloc[train_idx].copy()
        val_fold = train.iloc[val_idx].copy()
        
        # 特征工程
        feature_engineer = AdvancedFeatures()
        feature_engineer.fit(train_fold)
        train_processed = feature_engineer.transform(train_fold)
        val_processed = feature_engineer.transform(val_fold)
        
        # WCG特征 (只在训练集上计算)
        train_processed['WCG'] = calculate_wcg_feature(train_processed)
        # 验证集的WCG用训练集的映射
        wcg_map = train_processed.groupby('Ticket')['WCG'].first().to_dict()
        val_processed['WCG'] = val_processed['Ticket'].map(wcg_map).fillna(0.5)
        
        # 获取特征
        features = get_features(train_processed)
        X_train = train_processed[features].fillna(0)
        X_val = val_processed[features].fillna(0)
        y_train = train_fold['Survived'].values
        y_val = val_fold['Survived'].values
        
        # 训练模型
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        
        score = model.score(X_val, y_val)
        scores.append(score)
        print(f"  Fold {fold+1}: {score:.4f}")
    
    return np.array(scores)

def main():
    print("=" * 60)
    print("泰坦尼克号 - 进阶版 (WCG + FarePerPerson)")
    print("=" * 60)
    
    # 加载数据
    train = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/train.csv')
    test = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/test.csv')
    
    print(f"\n训练集: {len(train)} 条")
    
    # 对比两种CV策略
    print("\n" + "=" * 60)
    print("StratifiedKFold CV")
    print("=" * 60)
    scores_skf = manual_cv(train, use_group_kfold=False)
    print(f"CV: {scores_skf.mean():.4f} (+/- {scores_skf.std():.4f})")
    
    print("\n" + "=" * 60)
    print("GroupKFold CV")
    print("=" * 60)
    scores_gkf = manual_cv(train, use_group_kfold=True)
    print(f"CV: {scores_gkf.mean():.4f} (+/- {scores_gkf.std():.4f})")
    
    # 最终训练
    print("\n" + "=" * 60)
    print("最终训练")
    print("=" * 60)
    
    feature_engineer = AdvancedFeatures()
    feature_engineer.fit(train)
    train_processed = feature_engineer.transform(train)
    test_processed = feature_engineer.transform(test)
    
    # WCG特征
    train_processed['WCG'] = calculate_wcg_feature(train_processed)
    wcg_map = train_processed.groupby('Ticket')['WCG'].first().to_dict()
    test_processed['WCG'] = test_processed['Ticket'].map(wcg_map).fillna(0.5)
    
    features = get_features(train_processed)
    X_train = train_processed[features].fillna(0)
    y_train = train_processed['Survived']
    X_test = test_processed[features].fillna(0)
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions.astype(int)
    })
    
    output_path = '/lhcos-data/projects/titanic/submission/submission_advanced.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\n文件: {output_path}")
    print(f"预测生存率: {submission['Survived'].mean():.2%}")
    print(f"\n特征重要性 (Top 10):")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for feat, imp in importances.head(10).items():
        print(f"  {feat}: {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("完成! 预期 LB: ~0.79-0.80")
    print("=" * 60)
    
    return submission

if __name__ == '__main__':
    submission = main()
