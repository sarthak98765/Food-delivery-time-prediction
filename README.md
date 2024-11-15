# 🍴 Food Delivery Time Prediction

A machine learning project to predict food delivery time based on three key features:  
1. **Age of the delivery partner**  
2. **Ratings of the delivery partner**  
3. **Distance between the restaurant and the delivery location**

---

## 📋 Table of Contents
- [About](#about)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Development](#model-development)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#conclusion)

---

## 📖 About
This project analyzes and predicts the time taken for food delivery using various parameters. The predictive model leverages both exploratory data analysis and a Long Short-Term Memory (LSTM) neural network for accurate delivery time estimation.

---

## ✨ Features
1. **Age of Delivery Partner**: Young delivery partners tend to take less time for deliveries.
2. **Ratings of Delivery Partner**: Higher ratings correlate with faster delivery times.
3. **Distance**: Delivery time increases linearly with distance.
4. **Visualization**: Visual relationships between features and time using `Plotly` and `Seaborn`.

---

## 💻 Technologies Used
- **Python**  
  - Libraries: `pandas`, `numpy`, `seaborn`, `plotly`, `keras`, `tensorflow`, `matplotlib`
- **Machine Learning**  
  - LSTM Neural Network using `Keras` and `TensorFlow`.

---

## 🔎 Exploratory Data Analysis (EDA)
### Relationship Between Distance and Time Taken
![Scatter Plot - Distance vs Time Taken](#)  
- **Insights**: Delivery time consistently increases with distance, but most deliveries occur within 25-30 minutes.

### Relationship Between Time Taken and Age
![Scatter Plot - Age vs Time Taken](#)  
- **Insights**: Younger delivery partners take less time for deliveries.

### Relationship Between Time Taken and Ratings
![Scatter Plot - Ratings vs Time Taken](#)  
- **Insights**: Delivery partners with higher ratings deliver faster.

### Vehicle and Order Type Analysis
![Box Plot - Vehicle and Order Type](#)  
- **Insights**: No significant difference in delivery time based on vehicle type or order type.

---

## ⚙️ Model Development
### LSTM Neural Network
The LSTM neural network is used to predict delivery time based on three features:  
1. **Delivery Partner Age**  
2. **Delivery Partner Ratings**  
3. **Distance**  

### Model Architecture
- Two LSTM layers
- Fully connected Dense layers
- Optimizer: `Adam`
- Loss Function: `Mean Squared Error`

### Model Training
- Batch Size: `1`  
- Epochs: `9`

---

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sarthak98765/Food-delivery-time-prediction.git
   cd Food-Delivery-Time-Prediction

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly keras tensorflow

3. Add your dataset file (deliverytime.txt) to the project directory.

## 🚀 Usage
1. Run the script:
   ```bash
   python delivery_time_prediction.py

2. Enter the required inputs:
   - **Age of Delivery Partner**
   - **Ratings of Delivery Partner**
   - **Distance**
3. View the predicted delivery time in minutes.

---

## 🔚 Conclusion
The features contributing the most to food delivery time are:
1. **Age of the delivery partner**
2. **Ratings of the delivery partner**
3. **Distance between the restaurant and the delivery location**

This project demonstrates how machine learning can be effectively applied to solve real-world problems in the food delivery industry.
