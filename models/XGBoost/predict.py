import numpy as np
import pickle
import xgboost as xgb
import csv

def load_data():
    with open('../../preprocess/processing/recover_processed_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    with open('../../preprocess/processing/test2_recover', 'rb') as f:
        test_data = pickle.load(f)
    return train_data, ground_truth.astype(int), test_data

def train():
    x, y, x_test = load_data()   
    print(x.shape[1], x_test.shape[1])
    # {'alpha': 0.1, 'eta': 0.25, 'gamma': 3, 'lambda': 600, 'max_depth': 9, 'n_estimators': 50}
    model = xgb.XGBClassifier(
        eval_metric='auc', 
        tree_method='hist',
        max_depth=9,
        n_estimators=50,
        eta=0.25,
        gamma=3,
        reg_lambda=600,
        reg_alpha=0.1
    )
    model.fit(x, y)
    res = model.predict(x_test)
    print(res)
    with open("./stage_1_predict.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "home_team_win"])
        for i in range(0, len(res)):
            if (res[i]):
                writer.writerow([i, "True"])
            else:
                writer.writerow([i, "False"])

if __name__ == '__main__':
    train()