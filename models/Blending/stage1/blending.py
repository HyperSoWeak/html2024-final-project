import csv
import math
import numpy as np

res = [0] * 6185
model_cnt = 4
acc = [0.55422, 0.58166, 0.57779, 0.58650, 0.52969, 0.55390]
ranking = [3, 5, 4, 6, 1, 2]
name = ["./gaussian_SVM_1.csv", "./xgboost.csv", "./random-forest-optimized.csv", "./dnn_model_submissions.csv", "./mean_shift_predictions.csv", "./lr.csv"]

def w(x):
    return 3.85 ** 3.85 ** 3.85 ** x

for j in range(model_cnt):
    with open(name[j], "r", newline='') as f1:
        rows = list(csv.reader(f1))

        for i in range(1, len(rows)):
            res[i - 1] += w(acc[j]) * (1 if rows[i][1] == "True" else -1)

with open("./blending_res_2.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "home_team_win"])
    for i in range(0, len(res)):
        if (res[i] > 0):
            writer.writerow([i, "True"])
        else:
            writer.writerow([i, "False"])