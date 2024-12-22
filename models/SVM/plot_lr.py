import os
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import numpy as np

# Paths for the data and plot files
train_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/recover_processed_data.pkl')
ground_truth_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/ground_truth')
plot_path_linear = os.path.join(os.path.dirname(__file__), 'learning_curve_linear_SVM.png')
plot_path_gaussian = os.path.join(os.path.dirname(__file__), 'learning_curve_gaussian_SVM.png')


def load_data():
    """ Load the training data and ground truth labels. """
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)


def plot_learning_curve(train_sizes, train_scores, test_scores, kernel_name, plot_path):
    """ Plot learning curve for the training and validation accuracy. """
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label=f'Training Accuracy ({kernel_name})', color='blue')
    plt.plot(train_sizes, test_scores.mean(axis=1), label=f'Validation Accuracy ({kernel_name})', color='red')
    plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), color='blue', alpha=0.2)
    plt.fill_between(train_sizes, test_scores.mean(axis=1) - test_scores.std(axis=1),
                     test_scores.mean(axis=1) + test_scores.std(axis=1), color='red', alpha=0.2)
    plt.title(f'Learning Curve for {kernel_name} Kernel SVM')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(plot_path)  # Save the plot as a PNG file
    plt.close()  # Close the plot to prevent it from being shown


if __name__ == '__main__':
    # Load full data
    X, y = load_data()

    # ---- Linear Kernel SVM ----
    print('Training SVM with Linear kernel using full data...')
    svm_linear = SVC(kernel='linear', C=10, tol=0.001, random_state=42)

    # Train the model on the full dataset
    svm_linear.fit(X, y)

    # Generate and plot learning curve for Linear Kernel
    train_sizes, train_scores, test_scores = learning_curve(
        svm_linear, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy', n_jobs=-1
    )
    plot_learning_curve(train_sizes, train_scores, test_scores, 'Linear', plot_path_linear)

    # ---- Gaussian Kernel SVM ----
    print('Training SVM with Gaussian (RBF) kernel using full data...')
    svm_gaussian = SVC(kernel='rbf', C=10, gamma=0.001, random_state=42)

    # Train the model on the full dataset
    svm_gaussian.fit(X, y)

    # Generate and plot learning curve for Gaussian Kernel
    train_sizes, train_scores, test_scores = learning_curve(
        svm_gaussian, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy', n_jobs=-1
    )
    plot_learning_curve(train_sizes, train_scores, test_scores, 'Gaussian', plot_path_gaussian)

    print(f'Learning curves saved: {plot_path_linear} (Linear), {plot_path_gaussian} (Gaussian)')
