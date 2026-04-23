#!/usr/bin/env python3
"""
泰坦尼克号 - 最终版（使用真实Kaggle数据）
预期得分: 0.78-0.82
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("泰坦尼克号 - 最终版（真实Kaggle数据）")
    print("=" * 60)
    
    # 加载真实数据
    train = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/train.csv')
    test = pd.read_csv('/root/.cache/kagglehub/competitions/titanic/test.csv')
    
    print(f"\n训练集: {len(train)} 条")
    print(f"测试集: {len(test)} 条")
    print(f"训练集生存率: {train['Survived'].mean():.2%}")
    
    # 特征工程
    def preprocess(df, is_train=True):
        df = df.copy()
        
        # 1. 性别编码（最关键特征）
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        
        # 2. 年龄填充 - 按性别和舱位分组
        for sex in [0, 1]:
            for pclass in [1, 2, 3]:
                mask = (df['Sex'] == sex) & (df['Pclass'] == pclass)
                median_age = df.loc[mask, 'Age'].median()
                if pd.notna(median_age):
                    df.loc[mask & df['Age'].isna(), 'Age'] = median_age
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # 3. 票价填充
        df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
        
        # 4. 登船港口填充
        df['Embarked'] = df['Embarked'].fillna('S')
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df = pd.concat([df, embarked_dummies], axis=1)
        
        # 5. 家庭规模
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # 6. 年龄段
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                              labels=[0, 1, 2, 3, 4]).astype(int)
        
        # 7. 票价分段
        df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3]).astype(int)
        
        # 8. 头衔提取
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 
                     'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1, 
                     'Countess': 4, 'Ms': 1, 'Lady': 4, 'Jonkheer': 4, 
                     'Don': 4, 'Dona': 4, 'Mme': 2, 'Capt': 4, 'Sir': 4}
        df['Title'] = df['Title'].map(title_map).fillna(4).astype(int)
        
        # 9. 舱房甲板
        df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
        deck_map = {'U': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
        df['CabinDeck'] = df['CabinDeck'].map(deck_map).fillna(0).astype(int)
        
        # 10. 特征交叉
        df['Pclass_Sex'] = df['Pclass'] * 2 + df['Sex']
        
        return df
    
    train = preprocess(train, is_train=True)
    test = preprocess(test, is_train=False)
    
    # 选择特征
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                'FamilySize', 'IsAlone', 'AgeBin', 'FareBin', 
                'Title', 'CabinDeck', 'Pclass_Sex']
    for col in train.columns:
        if col.startswith('Embarked_'):
            features.append(col)
    
    print(f"\n使用特征: {features}")
    
    # 训练模型
    X = train[features].fillna(0)
    y = train['Survived']
    
    # 模型1: 随机森林
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    
    # 模型2: 梯度提升
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring='accuracy')
    
    print("\n" + "=" * 60)
    print("模型交叉验证结果")
    print("=" * 60)
    print(f"随机森林: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
    print(f"梯度提升: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")
    
    # 模型融合
    voting = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    voting_scores = cross_val_score(voting, X, y, cv=cv, scoring='accuracy')
    print(f"模型融合: {voting_scores.mean():.4f} (+/- {voting_scores.std():.4f})")
    
    # 训练最终模型
    voting.fit(X, y)
    rf.fit(X, y)  # 单独训练RF用于特征重要性
    
    # 特征重要性
    print("\n特征重要性 (Random Forest):")
    importances = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importances.head(8).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 预测
    X_test = test[features].fillna(0)
    predictions = voting.predict(X_test)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions.astype(int)
    })
    
    output_path = '/lhcos-data/projects/titanic/submission_final.csv'
    submission.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("提交文件生成完成")
    print("=" * 60)
    print(f"文件路径: {output_path}")
    print(f"样本数: {len(submission)}")
    print(f"预测生存率: {submission['Survived'].mean():.2%}")
    print(f"\n前15行预览:")
    print(submission.head(15).to_string(index=False))
    print("\n" + "=" * 60)
    print(f"预期 Kaggle 得分: {voting_scores.mean():.4f} (约 {voting_scores.mean():.2%})")
    print("=" * 60)
    
    return submission

if __name__ == '__main__':
    submission = main()
