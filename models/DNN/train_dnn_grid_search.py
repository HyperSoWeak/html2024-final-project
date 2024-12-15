import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

X_train_path = os.path.join(os.path.dirname(__file__), 'preprocess/X_train.pkl')
y_train_path = os.path.join(os.path.dirname(__file__), 'preprocess/y_train.pkl')
model_save_path = os.path.join(os.path.dirname(__file__), 'models/dnn_model.keras')


def load_data():
    with open(X_train_path, 'rb') as f:
        X_train = pickle.load(f)
    with open(y_train_path, 'rb') as f:
        y_train = pickle.load(f)
    return X_train, y_train


def build_dnn_model(input_dim, layers_config, learning_rate, dropout_rate):
    model = Sequential()

    model.add(Input(shape=(input_dim,)))

    for units in layers_config:
        model.add(Dense(units, activation='tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    X_train, y_train = load_data()

    layers_configs = [
        [128, 64, 32],
        [128, 64, 32, 16],
        [256, 128, 64],
    ]
    batch_size_options = [32, 64]
    dropout_rates = [0.3, 0.5]
    n_splits = 3
    learning_rate = 0.001

    best_val_accuracy = float('-inf')
    best_model = None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for layers_config in layers_configs:
        for batch_size in batch_size_options:
            for dropout_rate in dropout_rates:
                print(f'Training with layers {layers_config}, batch size {batch_size}, dropout rate {dropout_rate}...')

                fold_accuracies = []

                for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
                    print(f'Training fold {fold + 1}/{n_splits}...')

                    X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
                    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

                    model = build_dnn_model(X_train.shape[1], layers_config, learning_rate, dropout_rate)

                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

                    model.fit(
                        X_fold_train, y_fold_train,
                        validation_data=(X_fold_val, y_fold_val),
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1
                    )

                    y_fold_val_pred = (model.predict(X_fold_val) > 0.5).astype(int)
                    fold_accuracy = accuracy_score(y_fold_val, y_fold_val_pred)
                    fold_accuracies.append(fold_accuracy)

                    print(f'Fold {fold + 1} accuracy: {fold_accuracy:.4f}')

                mean_fold_accuracy = np.mean(fold_accuracies)
                print(f'Mean cross-validation accuracy: {mean_fold_accuracy:.4f}')

                if mean_fold_accuracy > best_val_accuracy:
                    best_val_accuracy = mean_fold_accuracy
                    best_model = model
                    print(f'New best model found with validation accuracy {best_val_accuracy:.4f}')

    best_model.save(model_save_path)
    print(f'Best model saved at: {model_save_path}')
