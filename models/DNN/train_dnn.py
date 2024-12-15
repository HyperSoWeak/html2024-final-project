import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

X_train_path = os.path.join(os.path.dirname(__file__), 'preprocess/X_train.pkl')
y_train_path = os.path.join(os.path.dirname(__file__), 'preprocess/y_train.pkl')
model_path = os.path.join(os.path.dirname(__file__), 'models/dnn_model.pkl')


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
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    layers_config = [64, 64, 32, 16]
    batch_size = 64
    dropout_rate = 0.5
    learning_rate = 0.001

    model = build_dnn_model(X_train.shape[1], layers_config, learning_rate, dropout_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    print("Training DNN Model...")

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f'Model saved at: {model_path}')
