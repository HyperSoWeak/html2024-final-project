import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from multiprocessing import Pool

def load_data():
    with open('../../preprocess/processing/processed_data', 'rb') as f:
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
    plt.xlabel("Number of Weak Models (Decision Tree, d=5)")
    plt.ylabel("Accuracy")
    plt.show()

def train():
    x, y = load_data()
    train_size = len(y)
    val_size = 300
    E_in_avg = [0] * 19
    E_val_avg = [0] * 19
    fold = 8
    for k in range(1, fold + 1):
        x_train = np.concatenate((x[:int(train_size * k / 8) - val_size], x[int(train_size * k / 8):]))
        y_train = np.concatenate((y[:int(train_size * k / 8) - val_size], y[int(train_size * k / 8):]))
        x_val = x[int(train_size * k / 8) - val_size:int(train_size * k / 8)]
        y_val = y[int(train_size * k / 8) - val_size:int(train_size * k / 8)]
        input = []
        input.append((x_train, y_train, x_val, y_val, 2, 8, 1))
        input.append((x_train, y_train, x_val, y_val, 9, 13, 1))
        input.append((x_train, y_train, x_val, y_val, 14, 16, 1))
        input.append((x_train, y_train, x_val, y_val, 17, 20, 1))
        with Pool(4) as p:
            res = p.starmap(run_adaboost, input)
        for i in range(4):
            for j in range(0, len(res[i][0])):
                E_in_avg[res[i][2][j] - 2] += res[i][0][j]
            for j in range(0, len(res[i][1])):
                E_val_avg[res[i][2][j] - 2] += res[i][1][j]
    for i in range(len(E_in_avg)):
        E_in_avg[i] /= fold
        E_val_avg[i] /= fold
    record(E_in_avg, E_val_avg, range(2, 21, 1))
    
if __name__ == '__main__':
    train()