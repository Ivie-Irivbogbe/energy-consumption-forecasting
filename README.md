# Analytical Forecasting of Energy Consumption Patterns

## Description
This project involves developing a forecast model for U.S. natural gas consumption for the next 24 months. The analysis uses historical consumption data and economic indicators such as transportation indexes and vehicle miles traveled. Forecasting methods applied include ETS, Holt-Winters, and VAR models to capture trends and seasonality.

## Objectives
- Develop a forecast model for U.S. natural gas consumption for the next 24 months.
- Utilize historical consumption data to predict future trends.
- Incorporate economic indicators such as transportation indexes and vehicle miles traveled.
- Apply ETS and Holt-Winters models to forecast natural gas consumption, capturing trends and seasonality.
- Use VAR to analyze consumption with economic indicators for informed resource planning.
- Provide actionable insights for stakeholders for informed decision-making for energy production infrastructure development and meeting environmental targets.

## Data Description
- **Source:** Federal Reserve Economic Data (FRED)
  - [Natural Gas Consumption](https://fred.stlouisfed.org/series/NATURALGASD11)
  - [Consumer Price Index - Transportation](https://fred.stlouisfed.org/series/CPIETRANS)
  - [Vehicle Miles Traveled](https://fred.stlouisfed.org/series/TRFVOLUSM227SFWA)

- **Dataset:**
  - `CombinedData.py`: Python script for data processing and analysis.
  - `CombinedData.csv`: Combined dataset used for analysis.
  - `CPIETRANS.csv`: Consumer Price Index - Transportation data.
  - `TRFVOLUSM227SFWA.csv`: Vehicle Miles Traveled data.
  - `NATURALGASD11.csv`: Natural Gas Consumption data.
  - `Final Code.py`: Final analysis code.
  - `Final Presentation.pptx`: PowerPoint presentation with visualizations of the analysis.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Ivie-Irivbogbe/energy-consumption-forecasting.git
