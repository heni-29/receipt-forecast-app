import streamlit as st
import pandas as pd
import numpy as np
import torch

# Load trained weights
weights = torch.load("model_weights.pth")

def build_features(dates, start_time=0):
    months = pd.Series(dates).dt.month.values
    time = np.arange(start_time, start_time + len(dates))

    month_one_hot = np.zeros((len(months), 12))
    month_one_hot[np.arange(len(months)), months - 1] = 1

    X = np.column_stack((np.ones(len(dates)), time, time**0.8,  month_one_hot))

    return X

def predict(X_np, weights):
    X = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        y_pred = X @ weights
    return y_pred.numpy()

def main():
    st.title("Monthly Receipt Forecast App")
    st.markdown("This app predicts the number of scanned receipts for each month in 2022 using a model trained from scratch with PyTorch.")

    # Load 2021 data
    df = pd.read_csv("data_daily.csv")
    df.rename(columns={"# Date": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    monthly_data = df.resample("M").sum()
    monthly_data.reset_index(inplace=True)

    # Display 2021 data
    st.subheader("2021 Monthly Receipt Totals")
    st.dataframe(monthly_data[["Date", "Receipt_Count"]])

    # Generate predictions for 2022
    future_dates = pd.date_range("2022-01-31", periods=12, freq="ME")
    X_future = build_features(future_dates, start_time=len(monthly_data))
    y_future = predict(X_future, weights).flatten().astype(int)

    forecast_df = pd.DataFrame({
        "Month": future_dates.strftime("%B"),
        "Predicted_Receipt_Count": y_future
    })

    # User selects a month
    st.subheader("Select a month to see prediction:")
    selected_month = st.selectbox("Month", forecast_df["Month"])
    prediction = forecast_df[forecast_df["Month"] == selected_month]["Predicted_Receipt_Count"].values[0]
    st.success(f"Predicted Receipts for {selected_month} 2022: **{prediction:,}**")

    # Plotting
    st.subheader("Receipt Count Trend")
    combined_df = pd.DataFrame({
    "Date": pd.concat([monthly_data["Date"], pd.Series(future_dates)]).reset_index(drop=True),
    "Receipts": np.concatenate([
        monthly_data["Receipt_Count"].values,
        y_future
    ]),
    "Year": ["2021"] * len(monthly_data) + ["2022"] * len(y_future)
})
    combined_df.sort_values(by="Date", inplace=True)
    st.subheader("Full Receipt Forecast Trend (2021–2022)")
    st.line_chart(data=combined_df, x="Date", y="Receipts")

if __name__ == "__main__":
    main()
