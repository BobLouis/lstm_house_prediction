import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the CSV data
print("Loading dataset...")
df = pd.read_csv("dataset.csv", encoding='utf-8')

# Drop the Address column as specified
df = df.drop('Address', axis=1)

# Print the first few rows to see the dataset structure
print("\nDataset preview:")
print(df.head())



# Check date format in the first few rows
print("\nDate format examples:")
for i in range(min(5, len(df))):
    if i < len(df):
        print(f"Row {i}: {df['Date'].iloc[i]}")

# Convert the Date column to datetime format - try different formats
try:
    # Try Taiwanese calendar format (113 = 2024 in Gregorian)
    df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(
                                 str(x).replace(str(x).split('/')[0], 
                                               str(int(str(x).split('/')[0]) + 1911)), 
                                 format='%Y/%m/%d', errors='coerce'))
    print("\nUsing Taiwanese calendar conversion")
except Exception as e:
    print(f"Error with Taiwanese calendar conversion: {e}")
    # Fallback to direct parsing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    print("Using direct date parsing")

# Check how many valid dates we have
print(f"\nValid dates after conversion: {df['Date'].notna().sum()} out of {len(df)}")

# Drop rows with missing date values
df = df.dropna(subset=['Date'])

# Sort by date
df = df.sort_values('Date')

# Print data information after date processing
print("\nDataset shape after date processing:", df.shape)

# Handle percentage values in 'MainBuildingRatio'
if 'MainBuildingRatio' in df.columns:
    # Check if MainBuildingRatio contains percentage symbols
    if df['MainBuildingRatio'].dtype == 'object':
        # Convert percentage strings to float
        df['MainBuildingRatio'] = df['MainBuildingRatio'].str.replace('%', '').astype(float) / 100
        print("\nConverted MainBuildingRatio from percentage to decimal")

# Fill any remaining missing values with the mean of their respective columns
numeric_cols = df.select_dtypes(include=['number']).columns
for column in numeric_cols:
    df[column] = df[column].fillna(df[column].mean())
    
# For non-numeric columns (except date), fill with mode
for column in df.columns:
    if column != 'Date' and column not in numeric_cols:
        df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 0)

# Convert categorical columns to numeric using one-hot encoding
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"\nEncoding categorical columns: {categorical_cols}")
    df = pd.get_dummies(df, columns=categorical_cols)

print("\nDataset shape after encoding:", df.shape)
print("\nFeatures after encoding:", df.columns.tolist())

# Check for any remaining NaN values
print("\nRemaining NaN values after processing:")
print(df.isnull().sum())

# Select the target variable (TotalPrice) and features
X = df.drop(['TotalPrice', 'Date'], axis=1)
y = df['TotalPrice']


X_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)

# Normalize the target variable
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

# Create sequences for LSTM
def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# Adjust time steps if needed based on data size
time_steps = min(12, max(1, len(X) // 10))  # Ensure we have enough data for sequences
print(f"\nUsing time_steps = {time_steps}")

# Prepare the sequences
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
print("\nSequence shapes:")
print("X_seq shape:", X_seq.shape)
print("y_seq shape:", y_seq.shape)

# Only proceed if we have data
if len(X_seq) > 0:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    print("\nTrain-test split shapes:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)


# Only proceed if we have data
if len(X_seq) > 0:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    print("\nTrain-test split shapes:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Print model summary
    print("\nModel Summary:")
    model.summary()

    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=2
    )

    # Evaluate the model
    print("\nEvaluating the model...")
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {loss}')

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)

    # Inverse transform to get actual values
    y_pred_actual = y_scaler.inverse_transform(y_pred)
    y_test_actual = y_scaler.inverse_transform(y_test)

    # Calculate the Mean Absolute Error in the original scale
    mae = np.mean(np.abs(y_pred_actual - y_test_actual))
    print(f'Mean Absolute Error: {mae:.2f} (in original price scale)')

    # Save the model
    model.save('lstm_house_price_model.h5')
    print("\nModel saved as 'lstm_house_price_model.h5'")

    # Function to predict future house prices
    def predict_future_price(input_data, time_steps, model, X_scaler, y_scaler):
        # Ensure input_data has the right shape and is scaled
        if len(input_data) < time_steps:
            print("Not enough data points for prediction")
            return None
        
        # Select the last time_steps data points and scale them
        input_sequence = input_data[-time_steps:].values
        input_sequence_scaled = X_scaler.transform(input_sequence)
        
        # Reshape for LSTM [samples, time steps, features]
        input_sequence_reshaped = input_sequence_scaled.reshape(1, time_steps, input_sequence_scaled.shape[1])
        
        # Get prediction
        predicted_scaled = model.predict(input_sequence_reshaped)
        
        # Inverse transform to get actual price
        predicted_price = y_scaler.inverse_transform(predicted_scaled)
        
        return predicted_price[0][0]

    # Example of using the prediction function
    print("\nExample prediction for the next period:")
    if len(X) > time_steps:
        example_input = X.iloc[-time_steps:].copy()
        predicted_price = predict_future_price(example_input, time_steps, model, X_scaler, y_scaler)
        print(f"Predicted house price: {predicted_price:.2f}")
else:
    print("\n⚠️ ERROR: Not enough data available after processing to create sequences.")
    print("Please check if your dates are correctly parsed and if you have enough data rows.")


# Calculate different time windows for analysis
short_window = min(30, len(y_test_actual))
medium_window = min(90, len(y_test_actual))

# Create DataFrame for easier plotting
results_df = pd.DataFrame({
    'Actual': y_test_actual.flatten(),
    'Predicted': y_pred_actual.flatten(),
    'Error': (y_test_actual - y_pred_actual).flatten(),
    'Percent_Error': ((y_test_actual - y_pred_actual) / y_test_actual * 100).flatten()
})

# Plot full test set predictions
plt.figure(figsize=(14, 7))
plt.plot(results_df.index, results_df['Actual'], label='Actual Prices', linewidth=2)
plt.plot(results_df.index, results_df['Predicted'], label='Predicted Prices', linewidth=2, linestyle='--')
plt.fill_between(results_df.index, results_df['Actual'], results_df['Predicted'], alpha=0.3, color='lightgrey')
plt.title('House Price Prediction Results - Full Test Set', fontsize=16)
plt.xlabel('Test Sample Index', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Plot short-term window for detailed view
plt.figure(figsize=(14, 7))
plt.plot(results_df.index[:short_window], results_df['Actual'][:short_window], 
         label='Actual', marker='o', markersize=6, linewidth=2)
plt.plot(results_df.index[:short_window], results_df['Predicted'][:short_window], 
         label='Predicted', marker='x', markersize=6, linewidth=2, linestyle='--')
plt.fill_between(results_df.index[:short_window], 
                results_df['Actual'][:short_window], 
                results_df['Predicted'][:short_window], 
                alpha=0.3, color='lightgrey')
plt.title(f'House Price Prediction - Detailed View (First {short_window} Samples)', fontsize=16)
plt.xlabel('Test Sample Index', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Plot medium-term window
if len(y_test_actual) > 30:
    plt.figure(figsize=(14, 7))
    plt.plot(results_df.index[:medium_window], results_df['Actual'][:medium_window], 
             label='Actual', linewidth=2)
    plt.plot(results_df.index[:medium_window], results_df['Predicted'][:medium_window], 
             label='Predicted', linewidth=2, linestyle='--')
    plt.fill_between(results_df.index[:medium_window], 
                    results_df['Actual'][:medium_window], 
                    results_df['Predicted'][:medium_window], 
                    alpha=0.3, color='lightgrey')
    plt.title(f'House Price Prediction - Medium-term View (First {medium_window} Samples)', fontsize=16)
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    