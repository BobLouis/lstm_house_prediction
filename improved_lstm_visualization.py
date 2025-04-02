import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

def plot_prediction_analysis(y_true, y_pred, title='Price Prediction Analysis'):
    """
    Plot comprehensive analysis of price predictions.
    
    Args:
        y_true: Actual price values
        y_pred: Predicted price values
        title: Title for the main plot
    """
    # Flatten arrays if needed
    y_true = y_true.flatten() if hasattr(y_true, 'flatten') else y_true
    y_pred = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    
    # Create results dataframe
    results = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Error': y_true - y_pred,
        'AbsError': np.abs(y_true - y_pred),
        'PercentError': (y_true - y_pred) / y_true * 100
    })
    
    # Calculate error metrics
    mae = np.mean(results['AbsError'])
    mape = np.mean(np.abs(results['PercentError']))
    rmse = np.sqrt(np.mean(np.square(results['Error'])))
    
    # 1. Main prediction plot
    plt.figure(figsize=(14, 8))
    
    # Plot actual vs predicted
    plt.plot(results.index, results['Actual'], 'b-', label='Actual', alpha=0.7)
    plt.plot(results.index, results['Predicted'], 'r--', label='Predicted', alpha=0.7)
    
    # Add key metrics to the plot
    plt.text(0.02, 0.95, f'MAE: {mae:.2f}\nMAPE: {mape:.2f}%\nRMSE: {rmse:.2f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Prediction accuracy by price range
    # Create price bins based on actual values
    bins = [0, 1000, 5000, 10000, 20000, np.inf]
    labels = ['<1k', '1k-5k', '5k-10k', '10k-20k', '>20k']
    results['PriceBin'] = pd.cut(results['Actual'], bins=bins, labels=labels)
    
    # Calculate MAPE by price bin
    bin_metrics = results.groupby('PriceBin').agg(
        Count=('Actual', 'count'),
        MeanPrice=('Actual', 'mean'),
        MAPE=('PercentError', lambda x: np.mean(np.abs(x))),
        MAE=('AbsError', 'mean')
    ).reset_index()
    
    # Plot MAPE by price bin
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='PriceBin', y='MAPE', data=bin_metrics, palette='viridis')
    
    # Add value labels on bars
    for i, v in enumerate(bin_metrics['MAPE']):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
        
    # Add sample count
    for i, (count, mae) in enumerate(zip(bin_metrics['Count'], bin_metrics['MAE'])):
        ax.text(i, 5, f'n={count}\nMAE={mae:.1f}', ha='center', fontsize=9)
    
    plt.title('Mean Absolute Percentage Error by Price Range', fontsize=16)
    plt.xlabel('Price Range', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Return the results dataframe for further analysis
    return results

def create_improved_lstm_model(input_shape, learning_rate=0.001):
    """
    Create an improved LSTM model with techniques to better handle outliers.
    """
    model = Sequential([
        # First LSTM layer with more units
        LSTM(128, activation='tanh', return_sequences=True, 
             input_shape=input_shape, 
             recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Second LSTM layer
        LSTM(64, activation='tanh', return_sequences=False),
        BatchNormalization(),
        
        # Dense layers with dropout for regularization
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(1)
    ])
    
    # Use Huber loss which is less sensitive to outliers than MSE
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
    
    return model

def log_transform_target(y_data, inverse=False, epsilon=1.0):
    """
    Apply or inverse log transformation to target variable.
    """
    if inverse:
        return np.exp(y_data) - epsilon
    else:
        return np.log(y_data + epsilon)

def price_based_sampling(X, y, price_threshold=10000, high_price_multiplier=5):
    """
    Oversample high-priced properties to give them more weight in training.
    """
    # Find indices of high-priced properties
    high_price_indices = np.where(y >= price_threshold)[0]
    
    if len(high_price_indices) == 0:
        return X, y
    
    # Duplicate high-priced properties
    X_high = np.repeat(X[high_price_indices], high_price_multiplier, axis=0)
    y_high = np.repeat(y[high_price_indices], high_price_multiplier, axis=0)
    
    # Combine with original data
    X_combined = np.vstack([X, X_high])
    y_combined = np.vstack([y, y_high])
    
    # Shuffle the combined data
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    
    return X_combined[indices], y_combined[indices]

def analyze_high_value_properties(df, price_column='TotalPrice', threshold=10000):
    """
    Analyze high-value properties to understand their characteristics.
    """
    # Separate high and normal value properties
    high_value = df[df[price_column] >= threshold].copy()
    normal_value = df[df[price_column] < threshold].copy()
    
    # Print summary statistics
    print(f"Total properties: {len(df)}")
    print(f"High-value properties (>={threshold}): {len(high_value)} ({len(high_value)/len(df)*100:.1f}%)")
    print(f"Normal-value properties (<{threshold}): {len(normal_value)} ({len(normal_value)/len(df)*100:.1f}%)")
    
    # Calculate mean values for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) > 0:
        print("\nMean values for numeric features:")
        comparison = pd.DataFrame({
            'High-value': high_value[numeric_cols].mean(),
            'Normal-value': normal_value[numeric_cols].mean(),
            'Ratio': high_value[numeric_cols].mean() / normal_value[numeric_cols].mean()
        })
        print(comparison.sort_values('Ratio', ascending=False))
    
    # Plot price distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df[price_column], bins=50, kde=True)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.title('Price Distribution with High-Value Threshold', fontsize=16)
    plt.xlabel('Price', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return high_value, normal_value 