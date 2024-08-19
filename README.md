# Stock Price Prediction Using Machine Learning

## Overview

This project focuses on predicting stock prices by analyzing eight years of historical stock data, particularly studying the correlation between the US stock market and international index stocks. Multiple machine learning models were employed, with Orthogonal Matching Pursuit (OMP) emerging as the top-performing model based on its minimal Mean Squared Error (MSE). This project aims to demonstrate the potential of OMP in delivering accurate stock price predictions.

## Features

- **Data Analysis**: Analyzed eight years of stock data, exploring the correlation between the US stock market and international indices.
- **Model Comparison**: Compared various machine learning models, including Decision Tree Regressor, Neural Network, Orthogonal Matching Pursuit (OMP), Gradient Boosting, and LSTM networks.
- **Model Performance**: Determined OMP as the most effective model with the lowest MSE of 1.24e-05.
- **Prediction Accuracy**: Demonstrated the correlation between the performance of OMP and its capability to accurately predict stock prices.

## Technology Stack

- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-Learn, TensorFlow/Keras (for neural networks and LSTM), Matplotlib/Seaborn (for data visualization)
- **Models Used**:
  - Decision Tree Regressor
  - Neural Network
  - Orthogonal Matching Pursuit (OMP)
  - Gradient Boosting
  - LSTM Networks

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Raja-Apoorva/Stock-prediction-ML.git
   cd Stock-prediction-ML
   ```

## Usage

- **Data Analysis**: Use the analysis scripts to explore correlations in the stock data.
- **Model Training**: Train various models by executing the provided scripts, and compare their performance.
- **Prediction**: Use the trained OMP model to make stock price predictions and evaluate its accuracy.

## Contributing

If you wish to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the creators of Scikit-Learn, TensorFlow, and other open-source libraries used in this project.
- Gratitude to the financial data providers for the stock market data used in this analysis.
