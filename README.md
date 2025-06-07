# Project: Evaluating Mean Reversion Characteristics on the US stock market
Source Code Development for Mean Reversion

# Project Objective

To test whether certain US stocks or stock pairs exhibit statistically significant mean-reverting behavior, and to design a trading framework that takes advantage of these inefficiencies.

# How to install 
```shell
pip install -r requirements.txt
```



# Repository Structure
├── utils.py # Utility functions for data download and mean reversion testing
├── notebooks/
│ ├── Mean_Reversion_Individual_Stock_Notebook.ipynb # Tests ADF, Hurst exponent, OU half-life on individual stocks
│ └── Mean_Reversion_Pair_Trading_Notebook.ipynb # Identifies cointegrated pairs and backtests the pair trading strategy
├── requirements.txt # Python dependencies
└── README.md # Project overview and setup instructions

# Clone this repository

```bash
git clone https://github.com/lequee/MScFE_690_Capstone.git
```
