import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

train_data_path = os.path.join(os.path.dirname(__file__), '../../data/train_data_recovered.csv')
test_data_path = os.path.join(os.path.dirname(__file__), '../../data/2024_test_data_recovered.csv')


def load_data():
    df = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    df.dropna(subset=['home_team_win'])

    X = df.drop(columns=['home_team_win', 'id', 'date'])
    y = df['home_team_win'].astype(int)

    X_test = df_test.drop(columns=['id'])

    categorical_cols = ['home_team_abbr', 'away_team_abbr', 'season',
                        'home_team_season', 'away_team_season', 'home_pitcher', 'away_pitcher']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # Numerical transformer with StandardScaler and Imputer
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())  # Use StandardScaler here
    ])

    # Categorical transformer with OneHotEncoder and Imputer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1126)

    # Preprocess the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_val_preprocessed, X_test_preprocessed, y_train, y_val


def create_dnn_model(input_dim):
    model = Sequential()

    # Input Layer
    model.add(Dense(128, input_dim=input_dim, activation='tanh'))
    model.add(Dropout(0.4))  # Increased dropout to reduce overfitting
    model.add(BatchNormalization())

    # Hidden Layers
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

    # Compile the model with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val = load_data()

    model = create_dnn_model(input_dim=X_train.shape[1])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    print("Training DNN Model...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
              callbacks=[early_stopping, reduce_lr], verbose=2)

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Evaluate on the test data
    y_test_pred = model.predict(X_test)

    # Convert predictions to binary (0 or 1)
    y_test_pred = (y_test_pred > 0.5).astype(int)

    # Create submission file
    submission_df = pd.DataFrame({
        'id': range(len(y_test_pred)),
        'home_team_win': [True if pred == 1 else False for pred in y_test_pred]
    })

    file_name = os.path.splitext(os.path.basename(__file__))[0]
    submission_path = os.path.join(os.path.dirname(__file__), f'submissions/{file_name}.csv')
    submission_df.to_csv(submission_path, index=False)

    print("Submission file generated at:", submission_path)
