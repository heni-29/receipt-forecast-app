# Receipt Forecast App

This project predicts monthly scanned receipt counts for 2022 using a linear regression model built from scratch with PyTorch. The app is built with Streamlit and fully containerized with Docker.

## Run via Docker (No setup required)

```bash
docker pull heni29/receipt-forecast-app
docker run -p 8501:8501 heni29/receipt-forecast-app
