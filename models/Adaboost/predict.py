import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from multiprocessing import Pool
import csv

def load_data():
    with open('../../preprocess/processing/processed_data', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    with open('../../preprocess/processing/test2', 'rb') as f:
        test_data = pickle.load(f)
    return train_data, ground_truth.astype(int), test_data

def run_adaboost(x_train, y_train):
    dtree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, random_state=42)
    model = AdaBoostClassifier(n_estimators=150, algorithm="SAMME", estimator=dtree)
    model.fit(x_train, y_train)
    
    with open('./adaboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def test(x_test, model):
    res = model.predict(x_test)

    with open("./Results/stage_2_predict.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "home_team_win"])
        for i in range(0, len(res)):
            if (res[i]):
                writer.writerow([i, "True"])
            else:
                writer.writerow([i, "False"])
                
def load_model():
    with open("./adaboost_model.pkl") as f:
        model = pickle.load(f)
    return model

def train():
    x_train, y_train, x_test = load_data()
    model = run_adaboost(x_train, y_train)
    test(x_test, model)
    
if __name__ == '__main__':
    train()