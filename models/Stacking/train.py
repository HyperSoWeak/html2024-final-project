import csv
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def load_data():
    with open('../../preprocess/processing/processed_data', 'rb') as f:
        train_data = pickle.load(f)
    with open('../../preprocess/processing/ground_truth', 'rb') as f:
        ground_truth = pickle.load(f)
    with open('../../preprocess/processing/test1', 'rb') as f:
        test_data = pickle.load(f)
    return train_data, ground_truth.astype(int), test_data

def train():
    x, y, x_test = load_data()
    estimators = [('ada', AdaBoostClassifier(random_state=42, algorithm="SAMME", n_estimators=2500)),
                  ('svm', SVC(random_state=42)),
                  ('lr', LogisticRegression(random_state=42, max_iter=1000))]
    rf = RandomForestClassifier(
        random_state=42,
        bootstrap=True,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=4,
        min_samples_split=10,
        n_estimators=200
    )
    # params = {"ada__n_estimators": [500, 1500, 2500],
    #           "lr__max_iter": [300, 600, 900]}
    st = StackingClassifier(estimators=estimators, final_estimator=rf, n_jobs=4, verbose=2)
    # grid = GridSearchCV(estimator=st, param_grid=params, verbose=2, cv=5, n_jobs=4)
    st.fit(x, y)
    res = st.predict(x_test)
    with open("./stacking.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "home_team_win"])
        for i in range(0, len(res)):
            if (res[i] == 1):
                writer.writerow([i, "True"])
            else:
                writer.writerow([i, "False"])

if __name__ == '__main__':
    train()