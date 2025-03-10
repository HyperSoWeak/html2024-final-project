import os
import pickle
import pandas as pd

test_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/test2_recover')
model_name = 'svc-rbf-rec'
model_path = os.path.join(os.path.dirname(__file__), f'models/{model_name}.pkl')


def load_test():
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    return test_data


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    model = load_model(model_path)

    X_test = load_test()
    y_test_pred = model.predict(X_test)

    submission_df = pd.DataFrame({
        'id': range(len(y_test_pred)),
        'home_team_win': [True if pred == 1 else False for pred in y_test_pred]
    })

    submission_path = os.path.join(os.path.dirname(__file__), f'submissions/{model_name}_stage2_submissions.csv')
    submission_df.to_csv(submission_path, index=False)

    print("Submission file generated at:", submission_path)
