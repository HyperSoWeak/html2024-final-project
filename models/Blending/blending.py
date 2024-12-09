import csv

res = [0] * 6185

with open("./gaussian_SVM_1.csv", "r", newline='') as f1:
    rows = list(csv.reader(f1))

    for i in range(1, len(rows)):
        res[i - 1] += 0.55422 * (1 if rows[i][1] == "True" else -1)

with open("./mean_shift_predictions.csv", "r", newline='') as f1:
    rows = list(csv.reader(f1))

    for i in range(1, len(rows)):
        res[i - 1] += 0.52969 * (1 if rows[i][1] == "True" else -1)

with open("./adaboost.csv", "r", newline='') as f1:
    rows = list(csv.reader(f1))

    for i in range(1, len(rows)):
        res[i - 1] += 0.56036 * (1 if rows[i][1] == "True" else -1)

with open("./lr.csv", "r", newline='') as f1:
    rows = list(csv.reader(f1))

    for i in range(1, len(rows)):
        res[i - 1] += 0.55390 * (1 if rows[i][1] == "True" else -1)

with open("./random-forest-optimized.csv", "r", newline='') as f1:
    rows = list(csv.reader(f1))

    for i in range(1, len(rows)):
        res[i - 1] += 0.57779 * (1 if rows[i][1] == "True" else -1)

with open("./dnn-1.csv", "r", newline='') as f1:
    rows = list(csv.reader(f1))

    for i in range(1, len(rows)):
        res[i - 1] += 0.52001 * (1 if rows[i][1] == "True" else -1)

with open("./blending.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "home_team_win"])
    for i in range(0, len(res)):
        if (res[i] > 0):
            writer.writerow([i, "True"])
        else:
            writer.writerow([i, "False"])