import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# File paths for loading data and saving the model
train_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/recover_processed_data.pkl')
ground_truth_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/ground_truth')
model_path = os.path.join(os.path.dirname(__file__), 'models/rf-rec.pkl')


def load_data():
    """Load the training data and ground truth labels."""
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)


if __name__ == '__main__':
    # Load the data
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Performing Grid Search for Random Forest...')

    # Initialize the RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt'],
        'bootstrap': [True],
        'criterion': ['gini']
    }

    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best RandomForestClassifier model from grid search
    best_rf = grid_search.best_estimator_

    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%')

    # Train the best model on the full training data
    best_rf.fit(X, y)

    # Make predictions on the validation set
    y_pred = best_rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f'Validation Accuracy after full training: {accuracy * 100:.2f}%')

    # Save the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(best_rf, f)

    print(f'Model saved at: {model_path}')
