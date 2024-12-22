import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/recover_processed_data.pkl')
ground_truth_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/ground_truth')
model_path = os.path.join(os.path.dirname(__file__), 'models/svc-rbf-rec.pkl')


def load_data():
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)


if __name__ == '__main__':
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Performing Grid Search for SVM with RBF kernel...')

    svm = SVC(kernel='rbf', random_state=42)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }

    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)

    best_svm = grid_search.best_estimator_

    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%')

    best_svm.fit(X, y)

    y_pred = best_svm.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f'Validation Accuracy after full training: {accuracy * 100:.2f}%')

    with open(model_path, 'wb') as f:
        pickle.dump(best_svm, f)

    print(f'Model saved at: {model_path}')
