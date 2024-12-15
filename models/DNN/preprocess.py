import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

train_data_path = os.path.join(os.path.dirname(__file__), '../../data/train_data.csv')
stage1_test_data_path = os.path.join(os.path.dirname(__file__), '../../data/same_season_test_data.csv')
stage2_test_data_path = os.path.join(os.path.dirname(__file__), '../../data/2024_test_data.csv')
preprocess_dir = os.path.join(os.path.dirname(__file__), './preprocess/')

df_train = pd.read_csv(train_data_path)
df_test_stage1 = pd.read_csv(stage1_test_data_path)
df_test_stage2 = pd.read_csv(stage2_test_data_path)

X_train = df_train.drop(columns=['home_team_win', 'id', 'date'])
y_train = df_train['home_team_win'].astype(int)

X_test1 = df_test_stage1.drop(columns=['id'])
X_test2 = df_test_stage2.drop(columns=['id'])

categorical_cols = ['home_team_abbr', 'away_team_abbr', 'season',
                    'home_team_season', 'away_team_season', 'home_pitcher', 'away_pitcher']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test1_preprocessed = preprocessor.transform(X_test1)
X_test2_preprocessed = preprocessor.transform(X_test2)

with open(os.path.join(preprocess_dir, 'X_train.pkl'), 'wb') as f:
    pickle.dump(X_train_preprocessed, f)

with open(os.path.join(preprocess_dir, 'y_train.pkl'), 'wb') as f:
    pickle.dump(y_train, f)

with open(os.path.join(preprocess_dir, 'X_test1.pkl'), 'wb') as f:
    pickle.dump(X_test1_preprocessed, f)

with open(os.path.join(preprocess_dir, 'X_test2.pkl'), 'wb') as f:
    pickle.dump(X_test2_preprocessed, f)

print(f'Preprocessed data saved to: {preprocess_dir}')
