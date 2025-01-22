# Intraday Market Navigator

## Project Overview
The **Intraday Market Navigator** is a machine learning-powered system designed to enhance decision-making for traders and investors in intraday stock trading. The project aims to minimize risks while maximizing profits through accurate and timely stock price predictions, with a specific focus on Reliance Industries Ltd. (RIL).

## Team
- **Vaibhav Vikas Ranjan** (221020457)
- **Juttuka Saaketh** (221010222)
- **Sanjana Sori** (221010241)

### Institution
Department of DSAI & ECE, Dr. Shyama Prasad Mukherjee International Institute of Information Technology, Naya Raipur.

---

## Features
1. **Predictive Model Development**: Utilizes advanced machine learning algorithms like SVM, LSTM, CNN, and Logistic Regression for intraday stock price forecasting.
2. **Data Analysis & Preprocessing**: Collects and processes minute-by-minute stock prices along with fundamental and technical analysis.
3. **User-Friendly Interface**: Enables traders to access actionable predictions for buy/sell signals.
4. **Portfolio Performance Tracking**: Evaluates the impact of model predictions on trading outcomes.

---

## Objectives
- Provide accurate intraday stock market predictions.
- Equip investors with tools to make informed trading decisions.
- Develop strategies for High-Frequency Trading (HFT) bots.

---

## Methodology
### 1. Data Collection and Preprocessing
- Historical intraday stock price data sourced from platforms like BSE, CBOE, ICE, and NYSE.
- Attributes include Date, Time, Open, Close, Low, High, and Volume.

### 2. Machine Learning Models
#### Baseline Model:
- **Logistic Regression**: Initial model for binary classification.

#### Advanced Models:
- **Support Vector Machines (SVM)**: Linear, Polynomial, and RBF kernels.
- **Naive Bayes**: Probabilistic classifier for independent features.
- **Random Forest**: Ensemble method combining decision trees.
- **Recurrent Neural Networks (RNNs)**: Multi-layer LSTMs and GRU architectures.
- **Convolutional Neural Networks (CNNs)**: Deep learning model for pattern recognition.

### 3. Feature Selection
- Lasso regularization and Extra Tree Classifier to identify relevant features and improve model efficiency.

### 4. Model Evaluation
- Metrics include training and test accuracy.
- Best-performing model: **SVM (Linear Kernel)** with 66.58% test accuracy.

### 5. Portfolio Simulation
- Simulates trading outcomes with a starting portfolio of Rs. 1,00,000.

---

## Results
- **SVM (Linear Kernel)** outperformed other models, demonstrating its robustness for intraday trading.
- Portfolio performance evaluated using real-time trading scenarios.

---

## Future Directions
1. Transition to live data acquisition using APIs like AlphaVantage for real-time stock prices.
2. Expand dataset size to improve neural network performance.
3. Explore additional machine learning models and features.

---

## References
1. Naik and Mohan: Study on intraday stock prediction using DNN.
2. LSTM and Random Forest: Multi-feature approach for prediction.
3. Machine Learning models for SPDR S&P 500 price movements.
4. MKSVR model integrating market news and stock prices.
5. Various academic papers and resources listed in the report.



```python
# Sample Code for Data Preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('reliance_stock_data.csv')

# Preprocessing
scaler = StandardScaler()
data[['Open', 'Close', 'Low', 'High', 'Volume']] = scaler.fit_transform(data[['Open', 'Close', 'Low', 'High', 'Volume']])

# Feature Engineering
# Example: Creating Moving Averages
data['SMA'] = data['Close'].rolling(window=5).mean()
data['EMA'] = data['Close'].ewm(span=5, adjust=False).mean()

# Save preprocessed data
data.to_csv('preprocessed_data.csv', index=False)

print("Data preprocessing complete and saved to 'preprocessed_data.csv'.")

# Sample Code for SVM Model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Split data
X = data[['Open', 'High', 'Low', 'Volume', 'SMA', 'EMA']]
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"SVM Model Accuracy: {accuracy:.2f}")




[Report.pdf](https://github.com/user-attachments/files/16839487/Report.pdf)
