import numpy as np
import pickle
from sklearn.cluster import MeanShift
import os

previous_directory = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def load_data():
    with open('../../preprocess/processing/processed_data', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)

def calculate_e_in(data, ground_truth, model):
    cluster_labels = model.predict(data)
    
    unique_clusters = np.unique(cluster_labels)
    binary_labels = {}
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        majority_label = int(np.sum(ground_truth[cluster_indices]) > cluster_indices.shape[0] / 2)
        binary_labels[cluster] = majority_label
    
    predicted_labels = np.array([binary_labels[label] for label in cluster_labels], dtype=int)
    
    e_in = np.sum(predicted_labels != ground_truth) / ground_truth.shape[0]
    return e_in, binary_labels

def train_mean_shift(train_data, bandwidth=None):
    mean_shift = MeanShift(bandwidth=bandwidth)
    mean_shift.fit(train_data)
    return mean_shift

def save_model(model, binary_labels, filename):
    model.binary_labels = binary_labels
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def main():
    X, y = load_data()
    mean_shift_model = train_mean_shift(X, bandwidth=None)
    e_in, binary_labels = calculate_e_in(X, y, mean_shift_model)
    print(f"E_in (in-sample error): {e_in}")
    
    save_model(mean_shift_model, binary_labels, "mean_shift_model_recover.pkl")

if __name__ == '__main__':
    main()
    os.chdir(previous_directory)