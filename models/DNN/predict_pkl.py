import os
import pickle
import pandas as pd

X_test_path = os.path.join(os.path.dirname(__file__), 'preprocess/X_test1.pkl')
model_name = 'dnn_model'
model_path = os.path.join(os.path.dirname(__file__), f'models/{model_name}.pkl')


def load_test():
    with open(X_test_path, 'rb') as f:
        test_data = pickle.load(f)
    return test_data


def load_model():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    model = load_model()

    X_test = load_test()
    y_test_pred = model.predict(X_test)
    y_test_pred = (y_test_pred > 0.5).astype(int)

    submission_df = pd.DataFrame({
        'id': range(len(y_test_pred)),
        'home_team_win': [True if pred == 1 else False for pred in y_test_pred]
    })

    submission_path = os.path.join(os.path.dirname(__file__), f'submissions/{model_name}_submissions.csv')
    submission_df.to_csv(submission_path, index=False)

    print("Submission file generated at:", submission_path)
