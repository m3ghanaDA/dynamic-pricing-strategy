# 🚖 Dynamic Pricing Strategy Using Python & Machine Learning

## 📌 Project Overview

This project demonstrates how **Dynamic Pricing** can be implemented using **Data Science and Machine Learning**. The objective is to optimize ride pricing by adjusting costs based on real-time **demand and supply conditions**, rather than relying solely on ride duration.

The project analyzes ride-sharing data, applies a dynamic pricing algorithm, evaluates profitability, and trains a machine learning model to predict optimized ride prices.

---

## 🎯 Business Problem

Traditional pricing models often use a fixed pricing strategy based only on ride duration. However, market conditions such as:

- Number of riders (Demand)
- Number of available drivers (Supply)
- Vehicle type
- Ride duration

can significantly impact the optimal ride price.

The goal is to develop a **dynamic pricing model** that:

- Increases prices during high-demand periods
- Increases prices when driver availability is low
- Maximizes revenue and profitability
- Maintains a balance between demand and supply

---

## 📊 Dataset Features

The dataset contains ride-related information including:

| Feature | Description |
|----------|-------------|
| Number_of_Riders | Total riders requesting rides |
| Number_of_Drivers | Available drivers |
| Vehicle_Type | Economy or Premium |
| Expected_Ride_Duration | Estimated ride duration |
| Historical_Cost_of_Ride | Original ride price |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Random Forest Regression
- Jupyter Notebook

---

## 📈 Exploratory Data Analysis (EDA)

The project includes:

### Descriptive Statistics
- Data overview
- Summary statistics

### Visualizations
- Ride Duration vs Historical Cost
- Vehicle Type vs Ride Cost Distribution
- Correlation Matrix Heatmap
- Profitability Analysis
- Actual vs Predicted Pricing

---

## ⚙️ Dynamic Pricing Logic

### Demand Multiplier

Demand levels are calculated using percentile thresholds:

- High Demand → 75th Percentile
- Low Demand → 25th Percentile

Higher rider demand results in increased ride prices.

### Supply Multiplier

Driver availability is used to estimate supply:

- Lower driver availability → Higher prices
- Higher driver availability → Lower prices

### Adjusted Ride Cost

The final ride price is calculated using:

```python
Adjusted Price =
Historical Ride Cost × Demand Multiplier × Supply Multiplier
```

This allows prices to dynamically adapt to market conditions.

---

## 💰 Profitability Analysis

After implementing the pricing strategy, the project:

- Calculates profit percentage for each ride
- Identifies profitable rides
- Identifies loss-making rides
- Visualizes profit distribution using a donut chart

Formula used:

```python
Profit Percentage =
((Adjusted Cost - Historical Cost) / Historical Cost) * 100
```

---

## 🤖 Machine Learning Model

### Data Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Outlier treatment using IQR
- Feature transformation
- Categorical encoding

Vehicle Type Encoding:

```python
Premium = 1
Economy = 0
```

---

## 🌲 Model Training

The project uses:

### Random Forest Regressor

Features:

```python
[
    Number_of_Riders,
    Number_of_Drivers,
    Vehicle_Type,
    Expected_Ride_Duration
]
```

Target:

```python
adjusted_ride_cost
```

Train-Test Split:

```python
80% Training
20% Testing
```

---

## 🔮 Price Prediction

The trained model can predict ride prices using custom user inputs.

Example:

```python
number_of_riders = 50
number_of_drivers = 25
vehicle_type = "Economy"
expected_ride_duration = 30
```

Output:

```python
Predicted Price: ₹XX.XX
```

---

## 📊 Model Evaluation

The model performance is evaluated by comparing:

- Actual Prices
- Predicted Prices

Visualization:

- Scatter Plot
- Ideal Prediction Line

This helps assess how closely the model predicts dynamic ride pricing.

---

## 📂 Project Structure

```bash
Dynamic-Pricing-Strategy/
│
├── dynamic pricing.ipynb
├── dynamic_pricing.csv
├── README.md
│
├── images/
│   ├── correlation_heatmap.png
│   ├── ride_duration_vs_cost.png
│   ├── profitability_chart.png
│   └── actual_vs_predicted.png
│
└── requirements.txt
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/m3ghanaDA/Dynamic-Pricing-Strategy.git
cd Dynamic-Pricing-Strategy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

---

## 📦 Requirements

```txt
pandas
numpy
plotly
scikit-learn
jupyter
```

Install all requirements:

```bash
pip install pandas numpy plotly scikit-learn jupyter
```

---

## 📌 Key Learnings

- Dynamic Pricing Fundamentals
- Demand & Supply-Based Pricing
- Feature Engineering
- Exploratory Data Analysis
- Machine Learning Regression
- Random Forest Modeling
- Business Profitability Analysis
- Data-Driven Decision Making

---

## 🔮 Future Improvements

- Real-time API Integration
- Time-Based Demand Forecasting
- Weather Data Integration
- Competitor Price Monitoring
- Deep Learning Pricing Models
- MLOps Deployment Pipeline
- Interactive Dashboard using Streamlit

---

## 📜 Conclusion

This project demonstrates how **Data Science and Machine Learning** can be leveraged to build a dynamic pricing system that adjusts ride costs based on demand and supply conditions. Such pricing strategies help businesses maximize revenue, improve resource allocation, and provide a more adaptive pricing mechanism compared to traditional static pricing models.

---

## 👩‍💻 Author

**Meghana D A**

- MSc Data Science (Distinction) – University of Essex
- Data Scientist | Machine Learning Enthusiast | Data Analyst
- Python | SQL | Power BI | Machine Learning | Deep Learning

### Connect with Me

- LinkedIn: https://www.linkedin.com/in/your-profile
- GitHub: https://github.com/your-github-username

---

⭐ If you found this project useful, please consider giving it a star!
