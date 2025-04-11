# Stock Price Prediction - TSLA using LSTM

This project aims to predict the opening price of Tesla (TSLA) stock using a Long Short-Term Memory (LSTM) neural network.

## Project Description

The project utilizes historical stock data for Tesla, fetched from Yahoo Finance, to train an LSTM model. The goal is to predict the next day's opening stock price based on the opening prices of the previous 50 days.

-   **Data Source:** Yahoo Finance (`yfinance` library)
-   **Stock Ticker:** TSLA (Tesla)
-   **Data Range Used:** 2010-06-29 to 2019-05-28 (Daily Interval)
-   **Feature Predicted:** 'Open' price
-   **Model:** Stacked LSTM network built with Keras.
-   **Sequence Length:** 50 days

## Libraries Used

The following Python libraries are used in the project:

-   `yfinance`: For fetching stock data.
-   `numpy`: For numerical operations.
-   `pandas`: For data manipulation and analysis.
-   `seaborn`: For data visualization.
-   `matplotlib`: For plotting graphs.
-   `scikit-learn`: For data preprocessing (MinMaxScaler) and evaluation metrics (MAE, MSE).
-   `keras` (TensorFlow backend): For building and training the LSTM model.

## Workflow

1.  **Data Loading:** Fetches TSLA stock data using `yfinance`.
2.  **Preprocessing:**
    *   Selects the 'Open' price column.
    *   Splits the data into training (80%) and testing (20%) sets.
    *   Scales the data using `MinMaxScaler`.
    *   Creates input sequences (previous 50 days' open prices) and corresponding target values (next day's open price).
3.  **Model Building:**
    *   Constructs a Sequential Keras model.
    *   Includes four LSTM layers (96 units each) with Dropout layers (rate 0.2) in between to prevent overfitting.
    *   Adds a Dense output layer with one neuron for the prediction.
4.  **Model Training:**
    *   Compiles the model using Mean Squared Error (MSE) loss and the Adam optimizer.
    *   Trains the model for 70 epochs, using a portion of the training data for validation.
    *   Saves the best performing model based on validation loss using `ModelCheckpoint`.
5.  **Prediction & Evaluation:**
    *   Loads the best saved model (`tesla.stock_prediction.hdf5`).
    *   Makes predictions on the test dataset.
    *   Inverse transforms the predictions and actual values back to their original scale.
    *   Visualizes the predicted prices against the true stock prices.
    *   Calculates the Mean Absolute Error (MAE) to evaluate model performance.

## Results

The model was trained and evaluated on the test set.

-   **Visualization:** The final plot shows the predicted stock prices (blue) against the true stock prices (red) for the test period.
-   **Mean Absolute Error (MAE):** The MAE on the test set was approximately **7.79**. This indicates the average absolute difference between the predicted opening price and the actual opening price.

*(Note: The `tesla.stock_prediction.hdf5` file containing the trained model weights is generated during the notebook execution.)*