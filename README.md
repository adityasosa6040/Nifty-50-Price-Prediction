# Nifty-50-Price-Prediction
This Python code uses a deep learning model called an LSTM (Long Short-Term Memory) network to forecast the Nifty 50 stock index. It handles everything from downloading the latest financial data to visualizing future predictions.

# Nifty 50 Stock Price Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict the closing price of the Nifty 50 stock market index (`^NSEI`). The model is built with TensorFlow and Keras, and it fetches historical data from Yahoo Finance.

## 📜 Overview

The notebook walks through the complete process of creating a time-series forecasting model:
1.  **Data Collection**: Fetches historical Nifty 50 data from 2010 to the present day using the `yfinance` library.
2.  **Data Preprocessing**: Cleans and scales the data using `MinMaxScaler` to prepare it for the LSTM model.
3.  **Model Building**: Constructs a stacked LSTM model with multiple layers and Dropout for regularization.
4.  **Training & Evaluation**: Trains the model on 80% of the historical data and evaluates its performance on the remaining 20% using metrics like RMSE, MAE, and R² score.
5.  **Forecasting**: Predicts the Nifty 50 closing price for the next 30 days based on the last 60 days of available data.

## 📈 Visualizations

### Historical Prediction vs. Actual Data
This chart shows the model's predictions on both the training (solid red) and testing (dashed red) data against the actual closing prices (blue).

![Nifty50 Price Prediction](<img width="1178" height="547" alt="Quant Model Nifty" src="https://github.com/user-attachments/assets/541c7313-fd11-4f4d-8af9-3a02d58c0a13" />
)<img width="1178" height="547" alt="Quant Model Nifty" src="https://github.com/user-attachments/assets/a03ed76e-9186-4c6f-97b8-6ce8d2c066e3" />


### Future 30-Day Forecast
This chart displays the historical data along with the model's 30-day forecast into the future.

![Nifty50 30-Day Forecast](<img width="1178" height="547" alt="Quant Model Prediction" src="https://github.com/user-attachments/assets/e0571113-5b6c-43bc-8ea0-ea770742740f" />
)<img width="1178" height="547" alt="Quant Model Prediction" src="https://github.com/user-attachments/assets/58e3ed0e-2dfe-4c4c-aaea-e7a031cdf62a" />


## ⚙️ Model Architecture

The model is a Sequential model built with Keras, consisting of four LSTM layers followed by a Dense output layer. Dropout layers are included to prevent overfitting.

-   **Layers**:
    -   LSTM (50 units, `return_sequences=True`)
    -   Dropout (0.2)
    -   LSTM (50 units, `return_sequences=True`)
    -   Dropout (0.2)
    -   LSTM (50 units, `return_sequences=True`)
    -   Dropout (0.2)
    -   LSTM (50 units)
    -   Dropout (0.2)
    -   Dense (1 unit)
-   **Optimizer**: Adam
-   **Loss Function**: Mean Squared Error

## 🛠️ Technologies Used

-   Python 3
-   TensorFlow & Keras
-   Scikit-learn
-   Pandas
-   NumPy
-   Matplotlib
-   yfinance

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/adityasosa6040/Nifty-50-Price-Prediction.git](https://github.com/adityasosa6040/Nifty-50-Price-Prediction.git)
    cd Nifty-50-Price-Prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    Create a `requirements.txt` file with the following content:
    ```txt
    numpy
    pandas
    matplotlib
    yfinance
    tensorflow
    scikit-learn
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Quant_Model_Nifty.ipynb
    ```
