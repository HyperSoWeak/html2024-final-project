import numpy as np
import pickle
import os
import pandas as pd

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def load_model():
    with open(f'mean_shift_models/mean_shift_model_recover.pkl', 'rb') as f:
        mean_shift_model = pickle.load(f)
    return mean_shift_model

def predict_test_data(test_data, model):
    cluster_labels = model.predict(test_data)
    binary_labels = np.array([model.binary_labels[label] for label in cluster_labels], dtype=int)
    return binary_labels

with open('../../preprocess/processing/test2_recover', 'rb') as f:
    test_data = pickle.load(f)

model = load_model()

predicted_y = (predict_test_data(test_data, model) > 0.5)
print(predicted_y)

output_filename = "mean_shift_predictions_recover.pkl"
with open(output_filename, 'wb') as f:
    pickle.dump(predicted_y, f)

df = pd.DataFrame(predicted_y)
df.to_csv(f'mean_shift_predictions_recover.csv')
