import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

def load_data():
    with open('../../preprocess/processing/recover_processed_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)

def train():
    x, y = load_data()   
    xgb_model = xgb.XGBClassifier(eval_metric='auc', tree_method='hist')
    # {'alpha': 0.1, 'eta': 0.25, 'gamma': 3, 'lambda': 600, 'max_depth': 9, 'n_estimators': 50}
    clf = GridSearchCV(
        xgb_model,
        {
            "max_depth": [9], 
            "n_estimators": [50],
            'eta': [0.25],
            'gamma': [3],
            'lambda': [600],
            'alpha': [0.1],
            # 'min_child_weight': [10],
        },
        verbose=2,
        n_jobs=-1,
        cv=10
    )
    clf.fit(x, y)
    print(f'E_val: {clf.best_score_}')
    print(clf.best_params_)
    best = clf.best_estimator_
    print(f'E_in: {best.score(x, y)}')
    

if __name__ == '__main__':
    train()