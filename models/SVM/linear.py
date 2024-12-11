import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from libsvm.svmutil import *

file_name = name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
train_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/processed_data')
ground_truth_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/ground_truth')
test_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/test1')
submission_path = os.path.join(os.path.dirname(__file__), f'submissions/{file_name}.csv')

def load_data():
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)

def load_test():
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    return test_data

def choose_best(X_train, y_train, X_test, y_test, C_values):
    best_acc = 0
    best_C = C_values[0]
    
    for C in C_values:
        print(f"Training with C={C}")
        svm_model = svm_train(y_train, X_train, f'-s 0 -t 0 -c {C} -h 0')
        
        _, accuracy, _ = svm_predict(y_test, X_test, svm_model)
        
        acc = accuracy[0]
        
        if acc > best_acc:
            best_acc = acc
            best_C = C
    
    print(f"Best C: {best_C} with accuracy: {best_acc:.4f}%")
    return best_C

if __name__ == '__main__':
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    best_C = choose_best(X_train, y_train, X_test, y_test, C_values)
    
    print(f"Training final model with best C={best_C}")
    svm_model_final = svm_train(y, X, f'-s 0 -t 0 -c {best_C} -h 0')
    
    X_test = load_test()
    y_test_pred, _, _ = svm_predict([0] * len(X_test), X_test, svm_model_final)
    
    submission_df = pd.DataFrame({
        'id': range(len(y_test_pred)),
        'home_team_win': [True if pred == 1 else False for pred in y_test_pred]
    })
    
    submission_df.to_csv(submission_path, index=False)
    print("Submission file generated at:", submission_path)
