import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/recover_processed_data.pkl')
ground_truth_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/ground_truth')
test_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/recover_test1.pkl')

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

    print('Training Random Forest...')

    # best params from hyperparameter tuning
    rf = RandomForestClassifier(
        random_state=42,
        bootstrap=True,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=4,
        min_samples_split=10,
        n_estimators=200
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    # print('Random Forest Trained')

    X_test = load_test()
    y_test_pred = rf.predict(X_test)

    submission_df = pd.DataFrame({
        'id': range(len(y_test_pred)),
        'home_team_win': [True if pred == 1 else False for pred in y_test_pred]
    })

    file_name = name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
    submission_path = os.path.join(os.path.dirname(__file__), f'submissions/{file_name}.csv')
    submission_df.to_csv(submission_path, index=False)

    print("Submission file generated at:", submission_path)