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
    np.set_printoptions(threshold=sys.maxsize)
    for i in range(x):
        if str(ground_truth[i]) == 'True':
            ground_truth[i] = 1
        else:
            ground_truth[i] = 0
    train = np.delete(train, [0, 1, 2, 3, 4, 5, 6, 7, 39, 40], axis=1)
    test = np.array(pd.read_csv('../../data/2024_test_data.csv', sep=',', header=None))[1:]
    test = np.delete(test, [0, 1, 2, 3, 4, 5, 37, 38], axis=1)

    return train.astype(np.float32), np.array(ground_truth).astype(np.float32), test.astype(np.float32)

def load_preprocessed_data():
    with open('../../preprocess/processing/processed_data', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    with open('../../preprocess/processing/test2_recover', 'rb') as f:
        test_data = pickle.load(f)
    return train_data, ground_truth.astype(int), test_data


def train():
    x, y, test_x = load_preprocessed_data()
    xgb_model = xgb.XGBClassifier(eval_metric='auc', missing=np.nan, booster="gblinear")
    clf = GridSearchCV(
        xgb_model,
        {
            "n_estimators": [50, 60],
            'lambda': [1, 3],
            'alpha': [0, 1]
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
    res_in = best.predict(x)
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