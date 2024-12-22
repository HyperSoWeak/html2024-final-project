import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import csv

def load_data():
    train = np.array(pd.read_csv('../../data/train_data_recovered.csv', sep=',', header=None))
    header = train[0]
    train = train[1:]
    x, y = train.shape
    train[:, 8:39] = train[:, 8:39].astype(float)
    train[:, 41:] = train[:, 41:].astype(float)
    ground_truth = [1 if game_res else 0 for game_res in train[:, 5]]
    for i in range(x):
        train[i][4] = 1 if train[i][4] == 'True' else 0
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

    test = np.array(pd.read_csv('../../data/2024_test_data_recovered.csv', sep=',', header=None))[1:]
    test_x, test_y = test.shape
    for i in range(test_x):
        test[i][3] = 1 if train[i][3] == 'True' else 0
    test_data = pd.DataFrame(data=test[:,1:],    
                            index=test[:,0],    
                            columns=header)
    for col in cat:
        test_data[col] = test_data[col].astype("category")
    for col in header:
        if col not in cat:
            test_data[col] = test_data[col].astype(float)

    return data, ground_truth, test_data

def train():
    x, y, x_test = load_data()
    print(y)
    # {'alpha': 0.1, 'eta': 0.3, 'gamma': 11, 'lambda': 0.1, 'max_depth': 11, 'min_child_weight': 10, 'n_estimators': 150}   
    model = xgb.XGBClassifier(
        eval_metric='auc', 
        tree_method='hist', 
        enable_categorical=True, 
        max_cat_to_onehot=6,
        reg_alpha=0.1,
        eta=0.3,
        gamma=11,
        reg_lambda=0.1,
        max_depth=5,
        min_child_weight=10,
        n_estimators=100
    )
    model.fit(x, y)
    model.save_model("xbg_model.json")
    res = model.predict(x_test)
    print(np.count_nonzero(res==0))
    with open("./stage_2_predict.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "home_team_win"])
        for i in range(0, len(res)):
            if (not res[i]):
                writer.writerow([i, "False"])
            else:
                writer.writerow([i, "True"])

if __name__ == '__main__':
    train()