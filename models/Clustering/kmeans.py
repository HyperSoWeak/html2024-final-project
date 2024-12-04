import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from tqdm import tqdm
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def load_data():
    with open('../../preprocess/processing/processed_data', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)

def calculate_e_val(ground_truth, cluster_labels, num_clusters):
    best_e_val = float('inf')
    N = ground_truth.shape[0]
    binary_labels = [0] * num_clusters
    
    for assignment in range(num_clusters):
        binary_labels[assignment] = 1
        predicted_labels = np.array([binary_labels[label] for label in cluster_labels])
        e_val1 = np.sum(predicted_labels != ground_truth) / N
        
        binary_labels[assignment] = 0
        predicted_labels = np.array([binary_labels[label] for label in cluster_labels])
        e_val0 = np.sum(predicted_labels != ground_truth) / N
    
        if e_val1 < e_val0:
            binary_labels[assignment] = 1
        else:
            binary_labels[assignment] = 0
    
    predicted_labels = np.array([binary_labels[label] for label in cluster_labels])
    best_e_val = np.sum(predicted_labels != ground_truth) / N
    
    return best_e_val, binary_labels

X, y = load_data()
k_values = range(2, 1229)
best_k = None
min_e_val = float('inf')
e_val_list = []
e_in_list = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for k in tqdm(k_values, desc='calulating E_in(k)'):
    e_val_sum = 0
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        train_fold, val_fold = X[train_idx], X[val_idx]
        y_fold = y[val_idx]
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(train_fold)
        
        val_cluster_labels = kmeans.predict(val_fold)
        e_val, _ = calculate_e_val(y_fold, val_cluster_labels, k)
        e_val_sum += e_val
    
    avg_e_val = e_val_sum / kf.n_splits
    e_val_list.append(avg_e_val)
    
    if avg_e_val < min_e_val:
        min_e_val = avg_e_val
        best_k = k
    
    print(f'k={k}, Average E_val={avg_e_val:.4f}')
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    # with open(f'kmeans_model_k{k}.pkl', 'rb') as f:
    #     kmeans = pickle.load(f)
    e_in, best_assignment = calculate_e_val(y, kmeans.labels_, k)
    
    binary_labels = np.array([best_assignment[label] for label in kmeans.labels_])
    kmeans.binary_labels = binary_labels
    
    e_in_list.append(e_in)
    
    with open(f'kmeans_model_k{k}.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
print(f'\nBest k: {best_k}, Minimum E_val: {min_e_val:.4f}')

plt.figure(figsize=(10, 6))
# plt.plot(k_values, e_val_list, linestyle='-', color='b', label='validation loss')
plt.plot(k_values, e_in_list, linestyle='-', color='g', label='training loss')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Error (E_val, E_in)')
plt.title('Number of Cluster vs Error')
plt.grid(True)
plt.legend()
plt.savefig('kmeans_curve.png')
plt.show()