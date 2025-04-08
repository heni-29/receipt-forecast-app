import streamlit as st 
import pandas as pd
import numpy as np
import torch

def load_data(path):
    df = pd.read_csv(path)
    df.rename(columns={"# Date": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    monthly_data = df.resample("M").sum()
    monthly_data.reset_index(inplace=True)
    return monthly_data

def build_features(dates, start_time=0):
    months = pd.Series(dates).dt.month.values
    time = np.arange(start_time, start_time + len(dates))

    # One-hot encode month
    month_one_hot = np.zeros((len(months), 12))
    month_one_hot[np.arange(len(months)), months - 1] = 1

    # Features: [bias, time, one-hot month]
    X = np.column_stack((np.ones(len(dates)), time, time**0.8, month_one_hot))
    return X

def train_model(X_np, y_np, lr=1e-6, epochs=10000):
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

    # Initialize weights
    w = torch.randn(X.shape[1], 1, dtype=torch.float32, requires_grad=True)

    for epoch in range(epochs):
        # Forward pass
        y_pred = X @ w
        loss = torch.mean((y_pred - y) ** 2)

        # Backward pass
        loss.backward()

        # Update weights manually
        with torch.no_grad():
            w -= lr * w.grad
            w.grad.zero_()

        # Optional: print loss
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.2f}")

    return w

def predict(X_np, weights):
    X = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        y_pred = X @ weights
    return y_pred.numpy()

def main():
    # Load and prepare data
    df = load_data("data_daily.csv")
    X = build_features(df["Date"])
    y = df["Receipt_Count"].values

    # Train model using gradient descent
    weights = train_model(X, y)

    # Forecast for 2022
    future_dates = pd.date_range("2022-01-31", periods=12, freq="ME")
    X_future = build_features(future_dates, start_time=X.shape[0])
    y_future_pred = predict(X_future, weights)

    # Output results
    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Predicted_Receipt_Count": y_future_pred.flatten().astype(int)
    }).set_index("Month")

    forecast_df.to_csv("2022_monthly_forecast_pytorch.csv")
    print("Forecast saved to 2022_monthly_forecast_pytorch.csv")
    print(forecast_df)
    torch.save(weights, "model_weights.pth")
    print("model_weights.pth saved!")

if __name__ == "__main__":
    main()