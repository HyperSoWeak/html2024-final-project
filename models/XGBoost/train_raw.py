import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import csv
import sys

def load_data():
    train = np.array(pd.read_csv('../../data/train_data.csv', sep=',', header=None))
    header = train[0]
    train = train[1:]
    x, y = train.shape
    ground_truth = train[:, 5]
    train[:, 8:39] = train[:, 8:39].astype(float)
    train[:, 41:] = train[:, 41:].astype(float)
    np.set_printoptions(threshold=sys.maxsize)
    for i in range(x):
        if str(train[i][4]) == 'True':
            train[i][4] = 1
        elif str(train[i][4]) == 'False':
            train[i][4] = 0
        else:
            train[i][4] = float('nan')
        if str(ground_truth[i]) == 'True':
            ground_truth[i] = 1
        else:
            ground_truth[i] = 0
    train = np.delete(train, [3, 5], axis=1)
    header = np.delete(header, [0, 3, 5])
    data = pd.DataFrame(data=train[:,1:],    
                        index=train[:,0],    
                        columns=header)
    cat = ['home_team_abbr', 'away_team_abbr', 'is_night_game', 'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season']
    for col in cat:
        data[col] = data[col].astype("category")
    for col in header:
        if col not in cat:
            data[col] = data[col].astype(float)
    test = np.array(pd.read_csv('../../data/2024_test_data.csv', sep=',', header=None))[1:]
    test_x, test_y = test.shape
    for i in range(test_x):
        if str(test[i][3]) == 'True':
            test[i][3] = 1
        elif str(test[i][3]) == 'False':
            test[i][3] = 0
        else:
            test[i][3] = float('nan')
    test_data = pd.DataFrame(data=test[:,1:],    
                            index=test[:,0],    
                            columns=header)
    for col in cat:
        test_data[col] = test_data[col].astype("category")
    for col in header:
        if col not in cat:
            test_data[col] = test_data[col].astype(float)

    return data, np.array(ground_truth).astype(float), test_data

def train():
    x, y, test_x = load_data()
    xgb_model = xgb.XGBClassifier(eval_metric='auc', tree_method='hist', enable_categorical=True, max_cat_to_onehot=7, missing=float('nan'))
    clf = GridSearchCV(
        xgb_model,
        {
            "max_depth": [3], 
            "n_estimators": [50],
            'eta': [0.3],
            'gamma': [1],
            'lambda': [100],
            'alpha': [0],
            'min_child_weight': [30],
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
    res = best.predict(test_x)
    with open("./stage_2_predict.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "home_team_win"])
        for i in range(0, len(res)):
            if (res[i]):
                writer.writerow([i, "True"])
            else:
                writer.writerow([i, "False"])

if __name__ == '__main__':
    train()