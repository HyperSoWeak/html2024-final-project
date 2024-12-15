import csv
import math
import numpy as np

res = [0] * 2428
model_cnt = 4
acc = [0.58388, 0.57724, 0.57308, 0.55481]
name = ["./dnn-recovered_stage2.csv", "./xgboost.csv", "./rbf_svm.csv", "./lin_reg.csv"]

def w(x):
    return 3.65 ** 3.65 ** 3.65 ** x

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