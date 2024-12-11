import os
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

train_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/processed_data')
ground_truth_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/ground_truth')
test_data_path = os.path.join(os.path.dirname(__file__), '../../preprocess/processing/test1')

def load_data():
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    return train_data, ground_truth.astype(int)

def load_test():
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    return test_data

def create_dnn_model(input_dim):
    model = Sequential()

    # Input Layer
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))  # Increased dropout to reduce overfitting
    model.add(BatchNormalization())

    # Hidden Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

    # Compile the model with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = create_dnn_model(input_dim=X_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    print("Training DNN Model...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
              callbacks=[early_stopping, reduce_lr], verbose=2)

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    X_test = load_test()
    X_test = scaler.transform(X_test)
    y_test_pred = model.predict(X_test)

    y_test_pred = (y_test_pred > 0.5).astype(int)

    submission_df = pd.DataFrame({
        'id': range(len(y_test_pred)),
        'home_team_win': [True if pred == 1 else False for pred in y_test_pred]
    })

    file_name = name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
    submission_path = os.path.join(os.path.dirname(__file__), f'submissions/{file_name}.csv')
    submission_df.to_csv(submission_path, index=False)

    print("Submission file generated at:", submission_path)
