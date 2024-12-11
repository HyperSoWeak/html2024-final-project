import numpy as np
import pickle
from scipy.stats import uniform, randint
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import xgboost as xgb

def load_data():
    with open('../../preprocess/processing/processed_data', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)

def train():
    x, y = load_data()   
    xgb_model = xgb.XGBClassifier(
        tree_method="hist"
    )
    clf = GridSearchCV(
        xgb_model,
        {"max_depth": [2, 3], 
         "n_estimators": [16, 17],
         'eval_metric': ['auc', 'ams@0'],
         'eta': [0.25, 0.5]
        },
        verbose=2,
        n_jobs=-1,
    )
    clf.fit(x, y)
    print(clf.best_score_)
    print(clf.best_params_)

if __name__ == '__main__':
    train()