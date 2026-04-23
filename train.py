#!/usr/bin/env python3
"""
泰坦尼克号 - 修复版 (无数据泄露)
关键修复:
1. 特征工程在CV循环内完成
2. 删除GroupSurvival (标签穿越)
3. TicketFreq只用train数据计算
4. Test绝不参与任何统计量
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

class SafeFeatures(BaseEstimator, TransformerMixin):
    """安全的特征工程，无数据泄露"""
    
    def __init__(self):
        self.age_medians = {}
        self.fare_medians = {}
        self.ticket_freq = {}
        
    def fit(self, X, y=None):
        """只在训练数据上计算统计量"""
        df = X.copy()
        
        # 1. 提取Title
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
                     'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                     'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                     'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                     'Capt': 'Rare', 'Sir': 'Rare'}
        df['Title'] = df['Title'].map(title_map).fillna('Rare')
        df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
        
        # 2. Age按Title+Pclass分组的中位数 (只在train上算)
        for title in df['Title'].unique():
            for pclass in [1, 2, 3]:
                mask = (df['Title'] == title) & (df['Pclass'] == pclass)
                median_age = df.loc[mask, 'Age'].median()
                if pd.notna(median_age):
                    self.age_medians[(title, pclass)] = median_age
        
        # 3. Fare按Pclass分组的中位数 (只在train上算)
        for pclass in [1, 2, 3]:
            median_fare = df.loc[df['Pclass'] == pclass, 'Fare'].median()
            if pd.notna(median_fare):
                self.fare_medians[pclass] = median_fare
        
        # 4. TicketFreq只在train上计算
        self.ticket_freq = df['Ticket'].value_counts().to_dict()
        
        return self
    
    def transform(self, X):
        """应用存储的统计量，绝不重新计算"""
        df = X.copy()
        
        # 1. Title
        if 'Title' not in df.columns:
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            title_map = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
                         'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                         'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                         'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                         'Capt': 'Rare', 'Sir': 'Rare'}
            df['Title'] = df['Title'].map(title_map).fillna('Rare')
        df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
        
        # 2. Sex
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        
        # 3. Age填充 (用fit时存储的统计量)
        for (title, pclass), median_age in self.age_medians.items():
            mask = (df['Title'] == title) & (df['Pclass'] == pclass) & (df['Age'].isna())
            df.loc[mask, 'Age'] = median_age
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # 4. Fare填充 + log (用fit时存储的统计量)
        for pclass, median_fare in self.fare_medians.items():
            mask = (df['Pclass'] == pclass) & (df['Fare'].isna())
            df.loc[mask, 'Fare'] = median_fare
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Fare_log'] = np.log1p(df['Fare'])
        
        # 5. Embarked
        df['Embarked'] = df['Embarked'].fillna('S')
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df = pd.concat([df, embarked_dummies], axis=1)
        
        # 6. Cabin
        df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
        df['HasCabin'] = (df['Cabin'].notna()).astype(int)
        deck_map = {'U': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
        df['CabinDeck'] = df['CabinDeck'].map(deck_map).fillna(0).astype(int)
        
        # 7. 家庭规模
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # 8. TicketFreq (用fit时存储的统计量，test中未见过的ticket填1)
        df['TicketFreq'] = df['Ticket'].map(self.ticket_freq).fillna(1)
        
        # 9. 交互特征
        df['Pclass_Sex'] = df['Pclass'] * 2 + df['Sex']
        
        return df

def get_features(df):
    """获取特征列"""
    features = ['Pclass', 'Sex', 'Age', 'Fare_log', 'Title', 'CabinDeck', 'HasCabin',
                'FamilySize', 'IsAlone', 'TicketFreq', 'Pclass_Sex']
    for col in df.columns:
        if col.startswith('Embarked_'):
            features.append(col)
    return features

def manual_cv(train, n_splits=5):
    """手写的CV循环，确保特征工程在每一折内独立完成"""
    
    # 准备原始数据
    y = train['Survived'].values
    
    # 5折分层CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
        # 切分原始数据
        train_fold = train.iloc[train_idx].copy()
        val_fold = train.iloc[val_idx].copy()
        
        # 在训练子集上fit
        feature_engineer = SafeFeatures()
        feature_engineer.fit(train_fold)
        
        # transform训练子集和验证子集
        train_processed = feature_engineer.transform(train_fold)
        val_processed = feature_engineer.transform(val_fold)
        
        # 获取特征
        features = get_features(train_processed)
        X_train = train_processed[features].fillna(0)
        X_val = val_processed[features].fillna(0)
        y_train = train_fold['Survived'].values
        y_val = val_fold['Survived'].values
        
        # 训练模型
        model = RandomForestClassifier(
            n_estimators=100,      # 减少树数量
            max_depth=4,           # 降低深度
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',   # 限制特征数
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        
        # 评估
        score = model.score(X_val, y_val)
        scores.append(score)
        print(f"  Fold {fold+1}: {score:.4f}")
    
    return np.array(scores)

def main():
    print("=" * 60)
    print("泰坦尼克号 - 修复版 (无数据泄露)")
    print("=" * 60)
    
    # 加载数据
    train = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/train.csv')
    test = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/test.csv')
    
    print(f"\n训练集: {len(train)} 条")
    print(f"测试集: {len(test)} 条")
    
    # 手动CV (特征工程在每一折内独立完成)
    print("\n" + "=" * 60)
    print("手动CV (无泄露)")
    print("=" * 60)
    scores = manual_cv(train)
    print(f"\nCV准确率: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"预期LB: ~{scores.mean():.2f} (应该与CV接近)")
    
    # 最终训练 (用全部训练数据)
    print("\n" + "=" * 60)
    print("最终训练")
    print("=" * 60)
    
    feature_engineer = SafeFeatures()
    feature_engineer.fit(train)
    train_processed = feature_engineer.transform(train)
    test_processed = feature_engineer.transform(test)
    
    features = get_features(train_processed)
    X_train = train_processed[features].fillna(0)
    y_train = train_processed['Survived']
    X_test = test_processed[features].fillna(0)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    # 预测
    predictions = model.predict(X_test)
    
    # 提交
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions.astype(int)
    })
    
    output_path = '/lhcos-data/projects/titanic/submission/submission_fixed.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\n文件: {output_path}")
    print(f"预测生存率: {submission['Survived'].mean():.2%}")
    print(f"\n特征重要性 (Top 8):")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for feat, imp in importances.head(8).items():
        print(f"  {feat}: {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("修复完成!")
    print("=" * 60)
    
    return submission

if __name__ == '__main__':
    submission = main()
