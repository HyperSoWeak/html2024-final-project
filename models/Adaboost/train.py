import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from multiprocessing import Pool

def load_data():
    with open('../../preprocess/processing/processed_data', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)

def run_adaboost(x_train, y_train, x_val, y_val, n_range):
    E_val = []
    E_in = []
    estimator_cnt = []
    for i in n_range:
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
    plt.legend(['$E_{in}$', '$E_{out}$'])
    plt.xlabel("Number of Weak Models (Decision tree)")
    plt.ylabel("Accuracy")
    plt.show()

def train():
    x, y = load_data()
    val_size = 300
    x_train = x[:-val_size]
    y_train = y[:-val_size]
    x_val = x[-val_size:]
    y_val = y[-val_size:]
    input = []
    input.append((x_train, y_train, x_val, y_val, (11000, 18000)))
    input.append((x_train, y_train, x_val, y_val, (12000, 17000)))
    input.append((x_train, y_train, x_val, y_val, (13000, 16000)))
    input.append((x_train, y_train, x_val, y_val, (14000, 15000)))
    with Pool(4) as p:
        res = p.starmap(run_adaboost, input)
    E_in = []
    E_val = []
    estimator_cnt = []
    for i in range(4):
        E_in += res[i][0]
        E_val += res[i][1]
        estimator_cnt += res[i][2]
    record(E_in, E_val, estimator_cnt)
    
if __name__ == '__main__':
    train() # so far best val: 9600, but actual best e_out: 2600 