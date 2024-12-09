import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from libsvm.svmutil import *

train_data_path = os.path.join(os.path.dirname(__file__), '../../data/train_data.csv')
test_data_path = os.path.join(os.path.dirname(__file__), '../../data/same_season_test_data.csv')
df = pd.read_csv(train_data_path)

df.dropna(subset=['home_team_win'], inplace=True)

X = df.drop(columns=['home_team_win', 'id', 'date', 'home_pitcher', 'away_pitcher'])
y = df['home_team_win'].astype(int)

categorical_cols = ['home_team_abbr', 'away_team_abbr', 'season', 'home_team_season', 'away_team_season']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

X_train_list = X_train_preprocessed.toarray().tolist()
X_test_list = X_test_preprocessed.toarray().tolist()
y_train_list = y_train.tolist()
y_test_list = y_test.tolist()

svm_model = svm_train(y_train_list, X_train_list, '-s 0 -t 0 -c 1')

y_pred_list, p_acc, _ = svm_predict(y_test_list, X_test_list, svm_model)

accuracy = accuracy_score(y_test_list, y_pred_list)
print(f'Accuracy: {accuracy:.4f}')

test_df = pd.read_csv(test_data_path)

columns_to_drop = ['id', 'home_pitcher', 'away_pitcher']
if 'date' in test_df.columns:
    columns_to_drop.append('date')

X_test_final = test_df.drop(columns=columns_to_drop)

X_test_final_preprocessed = preprocessor.transform(X_test_final)

X_test_final_list = X_test_final_preprocessed.toarray().tolist()

y_test_final_pred_list, _, _ = svm_predict([0] * len(X_test_final_list), X_test_final_list, svm_model)

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'home_team_win': [True if pred == 1 else False for pred in y_test_final_pred_list]
})

submission_path = os.path.join(os.path.dirname(__file__), 'submissions/linear_SVM_1.csv')
submission_df.to_csv(submission_path, index=False)

print("Submission file generated at:", submission_path)
