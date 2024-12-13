import csv
import math
import numpy as np

res = [0] * 6185
model_cnt = 6
acc = [0.55422, 0.58779, 0.58166, 0.55390, 0.57779, 0.58553]
name = ["./gaussian_SVM_1.csv", "./blending.csv", "./xgboost.csv", "./lr.csv", "./random-forest-optimized.csv", "./dnn-recovered.csv"]

def w(x):
    return x

for j in range(model_cnt):
    with open(name[j], "r", newline='') as f1:
        rows = list(csv.reader(f1))

        for i in range(1, len(rows)):
            res[i - 1] += w(acc[j]) * (1 if rows[i][1] == "True" else -1)

with open("./blending_res.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "home_team_win"])
    for i in range(0, len(res)):
        if (res[i] > 0):
            writer.writerow([i, "True"])
        else:
            writer.writerow([i, "False"])