import numpy as np
import pickle
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def load_model(k):
    with open(f'kmeans_model_k{k}.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    return kmeans_model

def predict_test_data(test_data, model):
    cluster_labels = model.predict(test_data)
    binary_labels = np.array([model.binary_labels[label] for label in cluster_labels], dtype=int)
    return binary_labels

with open('../../preprocess/processing/test1', 'rb') as f:
    test_data = pickle.load(f)

test_k = 1228
model = load_model(test_k)
model.cluster_centers_ = model.cluster_centers_.astype(np.float32)

predicted_y = (predict_test_data(test_data, model) > 0.5)
print(predicted_y)

with open('prediction_1', 'wb') as f:
    pickle.dump(predicted_y, f)