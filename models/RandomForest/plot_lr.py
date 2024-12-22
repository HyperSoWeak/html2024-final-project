import os
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import numpy as np

# Paths for the data and plot files
train_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/recover_processed_data.pkl')
ground_truth_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/ground_truth')
plot_path_rf = os.path.join(os.path.dirname(__file__), 'learning_curve_random_forest.png')


def load_data():
    """ Load the training data and ground truth labels. """
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)


def plot_learning_curve(train_sizes, train_scores, test_scores, plot_path):
    """ Plot learning curve for the training and validation accuracy. """
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Accuracy', color='blue')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation Accuracy', color='red')
    plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), color='blue', alpha=0.2)
    plt.fill_between(train_sizes, test_scores.mean(axis=1) - test_scores.std(axis=1),
                     test_scores.mean(axis=1) + test_scores.std(axis=1), color='red', alpha=0.2)
    plt.title('Learning Curve for Random Forest')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(plot_path)  # Save the plot as a PNG file
    plt.close()  # Close the plot to prevent it from being shown


if __name__ == '__main__':
    # Load full data
    X, y = load_data()

    # ---- Random Forest ----
    print('Training Random Forest model using full data...')
    rf = RandomForestClassifier(
        bootstrap=True,
        criterion='gini',
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=5,
        min_samples_split=2,
        n_estimators=1000,
        random_state=42
    )

    # Train the model on the full dataset
    rf.fit(X, y)

    # Generate and plot learning curve for Random Forest
    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy', n_jobs=-1
    )
    plot_learning_curve(train_sizes, train_scores, test_scores, plot_path_rf)

    print(f'Learning curve saved: {plot_path_rf}')
