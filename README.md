# Receipt Forecast App

This project predicts monthly scanned receipt counts for 2022 using a linear regression model built from scratch with PyTorch. The app is built with Streamlit and fully containerized with Docker.

## Docker Hub Link
https://hub.docker.com/r/heni29/receipt-forecast-app

## Model Details

Built with PyTorch
Manual training using gradient descent
Features: time trend + one-hot encoded months

## Technologies Used

Python 3.9
PyTorch
Pandas, NumPy
Streamlit
Docker

## Run via Docker (No setup required)

```bash
docker pull heni29/receipt-forecast-app
docker run -p 8501:8501 heni29/receipt-forecast-app
Then open: http://localhost:8501 in your browser
