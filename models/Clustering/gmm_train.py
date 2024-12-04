import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

previous_directory = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def load_data():
    with open('../../preprocess/processing/processed_data', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)

def calculate_e_in(ground_truth, cluster_labels, num_clusters):
    best_e_in = float('inf')
    N = ground_truth.shape[0]
    binary_labels = [0] * num_clusters
    
    for assignment in tqdm(range(num_clusters), desc=f'evaluating ein on k={num_clusters}'):
        binary_labels[assignment] = 1
        predicted_labels = np.array([binary_labels[label] for label in cluster_labels])
        e_in1 = np.sum(predicted_labels != ground_truth) / N
        
        binary_labels[assignment] = 1
        predicted_labels = np.array([binary_labels[label] for label in cluster_labels])
        e_in0 = np.sum(predicted_labels != ground_truth) / N
        
        if e_in1 < e_in0:
            binary_labels[assignment] = 1
        else:
            binary_labels[assignment] = 0
    
    predicted_labels = np.array([binary_labels[label] for label in cluster_labels])
    best_e_in = np.sum(predicted_labels != ground_truth) / N
    
    return best_e_in, binary_labels

def gmm_clustering_and_assignment(train_data, ground_truth, num_clusters, log_file):
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    gmm.fit(train_data)
    
    cluster_labels = gmm.predict(train_data)
    
    e_in, best_assignment = calculate_e_in(ground_truth, cluster_labels, num_clusters)
    log_file.write(f"Best E_in for k={num_clusters}: {e_in}\n")
    
    gmm.binary_labels = best_assignment
    with open(f'gmm_models/gmm_model_k{num_clusters}.pkl', 'wb') as f:
        pickle.dump(gmm, f)
    
    return gmm, e_in

def main():
    X, y = load_data()
    start, stop = int(sys.argv[1]), int(sys.argv[2])
    k_values = range(start, stop)
    best_k = None
    best_e_in = float('inf')
    e_in_list = []
    
    with open("clustering_results.txt", "w") as log_file, open("e_in_values.txt", "w") as e_in_file:
        for k in k_values:
            log_file.write(f"Clustering with k={k}...\n")
            print(f"Clustering with k={k}...")
            gmm, e_in = gmm_clustering_and_assignment(X, y, k, log_file)
            e_in_file.write(f"k={k}, E_in={e_in}\n")
            
            e_in_list.append(e_in)
            
            if e_in < best_e_in:
                best_e_in = e_in
                best_k = k
        
        log_file.write(f"Best k: {best_k} with E_in: {best_e_in}\n")
        print(f"Best k: {best_k} with E_in: {best_e_in}\n")
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, e_in_list, linestyle='-', color='g', label='training loss')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('E_in')
    plt.title('Number of GMM Cluster vs Error')
    plt.grid(True)
    plt.legend()
    plt.savefig('gmm_curve.png')
    plt.show()

if __name__ == '__main__':
    main()
    os.chdir(previous_directory)