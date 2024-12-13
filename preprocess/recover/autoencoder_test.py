import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from sklearn.impute import SimpleImputer

train_data_path = os.path.join(os.path.dirname(__file__), '../../data/train_data.csv')
test_data_path = os.path.join(os.path.dirname(__file__), '../../data/2024_test_data.csv')
save_data_path = os.path.join(os.path.dirname(__file__), '../../data/2024_test_data_recovered.csv')

df = pd.read_csv(test_data_path)

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_columns]

imputer = SimpleImputer(strategy='mean')
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric_imputed), columns=df_numeric_imputed.columns)

X_train, X_test = train_test_split(df_scaled, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoder = Dense(int(input_dim / 2), activation='relu')(input_layer)

decoder = Dense(int(input_dim / 2), activation='sigmoid')(encoder)
output_layer = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(input_layer, output_layer)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2)

df_scaled_pred = autoencoder.predict(df_scaled)

df_pred = pd.DataFrame(scaler.inverse_transform(df_scaled_pred), columns=df_scaled.columns)

df_filled = df_numeric.copy()
df_filled[df_numeric.isnull()] = df_pred[df_numeric.isnull()]

# Round to integer
int_columns = ['home_team_rest', 'away_team_rest', 'home_pitcher_rest', 'away_pitcher_rest', 'season']
df_filled[int_columns] = df_filled[int_columns].round().astype(int)

df[numeric_columns] = df_filled
df.to_csv(save_data_path, index=False)

print(f"Filled data saved to {save_data_path}")
