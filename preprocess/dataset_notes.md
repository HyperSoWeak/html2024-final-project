# dataset notes

## ttoi
- dictionary mapping team to its winning probability

## ptoi
- dictionary mapping pitcher to its winning probability

## ground_truth
- ground_truth isolated from training data

## preprocessed_data
sorted by date increasingly and without date
without index (column 0)
deleted season names (column 36, 37)
0. winning probability of home team (float)
1. winning probability of away team (float)
2. is night game (int) r = 0.02 to result
3. winning probability of home pitcher (float)
4. winning probability of away pitcher (float)
- columns performed PCA halfing the dimensions $d \rightarrow \lfloor \frac{1}{2} d \rfloor$
    - 10, 11, 12, 14, 20, 21, 22, 42, 48, 50, 54, 55, 57, 58, 60, 61, 66, 69, 70, 90, 91, 93, 96, 105
    - 15, 16, 17, 19, 24, 25, 26, 45, 51, 53, 72, 73, 75, 76, 78, 79, 84, 87, 88, 108, 109, 111, 114, 123
    - 28, 30, 126, 127, 132, 141
    - 32, 34, 144, 145, 150, 159
    - 43, 46, 49, 52
    - 56, 59, 62
    - 74, 77, 80


## testdata preprocessing
1. run the `test_preprocess.py`