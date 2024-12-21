import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data():
    with open('../../preprocess/processing/recover_processed_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)

def run_adaboost(x_train, y_train, x_val, y_val, start, end, diff):
    E_val = []
    E_in = []
    estimator_cnt = []
    for i in range(start, end + 1, diff):
        model = AdaBoostClassifier(n_estimators=i, algorithm="SAMME")
        model.fit(x_train, y_train)
        e_val = model.score(x_val, y_val)
        E_val.append(e_val)
        e_in = model.score(x_train, y_train)
        E_in.append(e_in)
        estimator_cnt.append(i)
        print(i, e_in, e_val)

    return E_in, E_val, estimator_cnt

def record(E_in, E_val, estimator_cnt):
    # with open('./adaboost_model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    
    plt.plot(estimator_cnt, E_in, 'b')
    plt.plot(estimator_cnt, E_val, 'r')
    plt.legend(['$Acc_{in}$', '$Acc_{val}$'])
    plt.xlabel("Number of Weak Models (Decision Stumps)")
    plt.ylabel("Accuracy")
    plt.show()

def train():
    x, y = load_data()
    ada = AdaBoostClassifier(algorithm="SAMME")
    E_in = []
    E_val = []
    index = list(range(100, 2000, 200))
    for i in index:
        params = {
            'n_estimators': [i]
        }
        grid_search = GridSearchCV(estimator=ada, param_grid=params, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
        grid_search.fit(x, y)
        E_val.append(grid_search.best_score_)
        E_in.append(grid_search.score(x, y))

    record(E_in, E_val, index)
    
if __name__ == '__main__':
    train()