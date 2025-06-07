# Project: Evaluating Mean Reversion Characteristics on the US stock market


## Project Objective
To test whether certain US stocks or stock pairs exhibit statistically significant mean-reverting behavior, and to design a trading framework that takes advantage of these inefficiencies.

## How to install 
```shell
pip install -r requirements.txt
```

## Repository Structure
```shell
├── README.md                                                  # Project overview and setup instructions
├── data_prices
│   ├── etf_industry_correlation_2010_2019.csv
│   ├── metadata.csv
│   ├── prices_2015_2019.csv
│   ├── ...
├── modelling_code
│   ├── Mean_Reversion_Individual_Stock_Notebook.ipynb         # Tests ADF, Hurst exponent, OU half-life on individual stocks
│   ├── Mean_Reversion_Pair_Trading_Notebook.ipynb             # Identifies cointegrated pairs and backtests the pair trading strategy
│   └── tools
│       ├── _init_.py
│       └── utils.py                                           # Utility functions for data download and mean reversion testing
├── modelling_result
│   ├── mean_reversion_individual_full_report_2020_2024.csv
│   ├── plots_individual_2010_2014.png
│   └── ...
└── requirements.txt                                           # Python dependencies
```

## Clone this repository
```shell
git clone https://github.com/lequee/MScFE_690_Capstone.git
```
