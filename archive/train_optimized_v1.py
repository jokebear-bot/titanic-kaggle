#!/usr/bin/env python3
"""
泰坦尼克号 - 优化版 (整合建议)
采纳: LightGBM, FarePerPerson, TicketPrefix
排除: WCG (泄露风险), GroupKFold (不适用), NameLen (噪声)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

class OptimizedFeatures(BaseEstimator, TransformerMixin):
    """优化特征工程，无泄露，精选高价值特征"""
    
    def __init__(self):
        self.age_medians = {}
        self.fare_medians = {}
        self.ticket_freq = {}
        
    def fit(self, X, y=None):
        df = X.copy()
        
        # Title
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
                     'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                     'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                     'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                     'Capt': 'Rare', 'Sir': 'Rare'}
        df['Title'] = df['Title'].map(title_map).fillna('Rare')
        df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
        
        # Age按Title+Pclass
        for title in df['Title'].unique():
            for pclass in [1, 2, 3]:
                mask = (df['Title'] == title) & (df['Pclass'] == pclass)
                median_age = df.loc[mask, 'Age'].median()
                if pd.notna(median_age):
                    self.age_medians[(title, pclass)] = median_age
        
        # Fare按Pclass
        for pclass in [1, 2, 3]:
            median_fare = df.loc[df['Pclass'] == pclass, 'Fare'].median()
            if pd.notna(median_fare):
                self.fare_medians[pclass] = median_fare
        
        # TicketFreq (train only)
        self.ticket_freq = df['Ticket'].value_counts().to_dict()
        
        return self
    
    def transform(self, X):
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
        
        # Age
        for (title, pclass), median_age in self.age_medians.items():
            mask = (df['Title'] == title) & (df['Pclass'] == pclass) & (df['Age'].isna())
            df.loc[mask, 'Age'] = median_age
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Fare
        for pclass, median_fare in self.fare_medians.items():
            mask = (df['Pclass'] == pclass) & (df['Fare'].isna())
            df.loc[mask, 'Fare'] = median_fare
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Fare_log'] = np.log1p(df['Fare'])
        
        # Embarked
        df['Embarked'] = df['Embarked'].fillna('S')
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df = pd.concat([df, embarked_dummies], axis=1)
        
        # Cabin (保留原始序数，不分箱)
        df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
        df['HasCabin'] = (df['Cabin'].notna()).astype(int)
        deck_map = {'U': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
        df['CabinDeck'] = df['CabinDeck'].map(deck_map).fillna(0).astype(int)
        
        # 家庭规模
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # TicketFreq
        df['TicketFreq'] = df['Ticket'].map(self.ticket_freq).fillna(1)
        
        # 1. FarePerPerson (高价值)
        df['FarePerPerson'] = df['Fare'] / df['TicketFreq']
        df['FarePerPerson_log'] = np.log1p(df['FarePerPerson'])
        
        # 2. TicketPrefix (高价值，Gemini没提)
        df['TicketPrefix'] = df['Ticket'].str.extract(r'^([A-Za-z\.\/]+)', expand=False).fillna('None')
        # 简化编码：常见前缀 vs 其他
        common_prefixes = ['PC', 'CA', 'A', 'STON', 'SOTON']
        df['TicketPrefix'] = df['TicketPrefix'].apply(lambda x: x if x in common_prefixes else 'Other')
        prefix_map = {'None': 0, 'PC': 1, 'CA': 2, 'A': 3, 'STON': 4, 'SOTON': 5, 'Other': 6}
        df['TicketPrefix'] = df['TicketPrefix'].map(prefix_map)
        
        # 3. 交互特征
        df['Pclass_Sex'] = df['Pclass'] * 2 + df['Sex']
        df['Age_Pclass'] = df['Age'] * df['Pclass'] / 10  # AgeClass
        
        return df

def get_features(df):
    """获取特征列"""
    features = ['Pclass', 'Sex', 'Age', 'Fare_log', 'FarePerPerson_log', 'Title',
                'CabinDeck', 'HasCabin', 'FamilySize', 'IsAlone', 'TicketFreq',
                'TicketPrefix', 'Pclass_Sex', 'Age_Pclass']
    for col in df.columns:
        if col.startswith('Embarked_'):
            features.append(col)
    return features

def try_lightgbm(X_train, y_train, X_val, y_val):
    """尝试LightGBM，如果安装了就使用"""
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        return model, score, 'LightGBM'
    except ImportError:
        return None, None, None

def manual_cv(train, use_lightgbm=False):
    """手写的CV循环"""
    y = train['Survived'].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    models_used = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train, y)):
        train_fold = train.iloc[train_idx].copy()
        val_fold = train.iloc[val_idx].copy()
        
        feature_engineer = OptimizedFeatures()
        feature_engineer.fit(train_fold)
        train_processed = feature_engineer.transform(train_fold)
        val_processed = feature_engineer.transform(val_fold)
        
        features = get_features(train_processed)
        X_train = train_processed[features].fillna(0)
        X_val = val_processed[features].fillna(0)
        y_train = train_fold['Survived'].values
        y_val = val_fold['Survived'].values
        
        # 优先尝试LightGBM
        if use_lightgbm:
            model, score, model_name = try_lightgbm(X_train, y_train, X_val, y_val)
            if model is None:
                # 回退到RandomForest
                model = RandomForestClassifier(
                    n_estimators=150, max_depth=4, min_samples_split=5,
                    min_samples_leaf=2, max_features='sqrt', random_state=RANDOM_STATE
                )
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                model_name = 'RandomForest'
        else:
            model = RandomForestClassifier(
                n_estimators=150, max_depth=4, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', random_state=RANDOM_STATE
            )
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            model_name = 'RandomForest'
        
        scores.append(score)
        models_used.append(model_name)
        print(f"  Fold {fold+1}: {score:.4f} ({model_name})")
    
    return np.array(scores), models_used

def main():
    print("=" * 60)
    print("泰坦尼克号 - 优化版 (整合建议)")
    print("=" * 60)
    
    # 检查LightGBM
    try:
        import lightgbm as lgb
        print("✓ LightGBM 已安装")
        has_lgb = True
    except ImportError:
        print("✗ LightGBM 未安装，使用 RandomForest")
        print("  安装: pip install lightgbm")
        has_lgb = False
    
    # 加载数据
    train = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/train.csv')
    test = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/test.csv')
    
    print(f"\n训练集: {len(train)} 条")
    
    # CV
    print("\n" + "=" * 60)
    print("交叉验证")
    print("=" * 60)
    scores, models_used = manual_cv(train, use_lightgbm=has_lgb)
    print(f"\nCV: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"模型: {set(models_used)}")
    
    # 最终训练
    print("\n" + "=" * 60)
    print("最终训练")
    print("=" * 60)
    
    feature_engineer = OptimizedFeatures()
    feature_engineer.fit(train)
    train_processed = feature_engineer.transform(train)
    test_processed = feature_engineer.transform(test)
    
    features = get_features(train_processed)
    X_train = train_processed[features].fillna(0)
    y_train = train_processed['Survived']
    X_test = test_processed[features].fillna(0)
    
    # 优先LightGBM
    if has_lgb:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1
        )
        model_name = 'LightGBM'
    else:
        model = RandomForestClassifier(
            n_estimators=150, max_depth=4, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=RANDOM_STATE
        )
        model_name = 'RandomForest'
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions.astype(int)
    })
    
    output_path = '/lhcos-data/projects/titanic/submission/submission_optimized.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\n文件: {output_path}")
    print(f"模型: {model_name}")
    print(f"预测生存率: {submission['Survived'].mean():.2%}")
    print(f"\n特征重要性 (Top 10):")
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        for feat, imp in importances.head(10).items():
            print(f"  {feat}: {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("完成! 预期 LB: ~0.79-0.80")
    print("=" * 60)
    
    return submission

if __name__ == '__main__':
    submission = main()
