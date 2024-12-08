import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/processed_data')
ground_truth_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/ground_truth')
test_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/test1')

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

if __name__ == '__main__':
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],           # Number of trees
        'max_depth': [None, 10, 20, 30],           # Max depth of each tree
        'min_samples_split': [2, 5, 10],           # Min samples required to split a node
        'min_samples_leaf': [1, 2, 4],             # Min samples required at each leaf
        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
        'bootstrap': [True, False]                 # Whether to use bootstrap sampling
    }

    print("Starting Grid Search...")

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    best_rf = grid_search.best_estimator_

    y_pred = best_rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy with Best Params: {accuracy * 100:.2f}%")

    X_test = load_test()
    y_test_pred = best_rf.predict(X_test)

    submission_df = pd.DataFrame({
        'id': range(len(y_test_pred)),
        'home_team_win': [True if pred == 1 else False for pred in y_test_pred]
    })

    submission_path = os.path.join(os.path.dirname(__file__), 'submissions/random-forest-optimized.csv')
    submission_df.to_csv(submission_path, index=False)

    print("Submission file generated at:", submission_path)
