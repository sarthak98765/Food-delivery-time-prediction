import pandas as pd
import numpy as np
import plotly.express as px
#https://www.datacamp.com/cheat-sheet/plotly-express-cheat-sheet
# Importing data 
df = pd.read_csv("deliverytime.txt")

df
df.shape
df.info()
df.isnull().sum()
# Calculating Distance
# Set the earth's radius (in kilometers)
R = 6371

# Convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi/180)

# Function to calculate the distance between two points using the haversine formula
def distcalculate(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon = deg_to_rad(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
  
# Calculate the distance between each pair of points
df['distance'] = np.nan

for i in range(len(df)):
    df.loc[i, 'distance'] = distcalculate(df.loc[i, 'Restaurant_latitude'], 
                                        df.loc[i, 'Restaurant_longitude'], 
                                        df.loc[i, 'Delivery_location_latitude'], 
                                        df.loc[i, 'Delivery_location_longitude'])
df
# EDA
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.scatterplot(data=df,x='distance',y='Time_taken(min)').set(title='Relationship Between Distance and Time Taken') 
                    

#Ordinary Least Squares (OLS) trendline function. Requires statsmodels to be installed.
#This trendline function causes fit results to be stored within the figure, accessible via the plotly.
figure = px.scatter(data_frame = df, 
                    x="distance",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    trendline="ols", 
                    title = "Relationship Between Distance and Time Taken")
figure.show()
#CONCLUSION:-1)There is a consistent relationship between the time taken and the distance travelled to deliver the food.
            #2)It means that most delivery partners deliver food within 25-30 minutes, regardless of distance.
figure1 = px.scatter(data_frame = df, 
                    x="Delivery_person_Age",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time Taken and Age")
figure1.show()
#CONCLUSION:-1)There is a linear relationship between the time taken to deliver the food and the age of the delivery partner.
            #2)It means young delivery partners take less time to deliver the food compared to the elder partners.
figure3 = px.scatter(data_frame = df, 
                    x="Delivery_person_Ratings",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time Taken and Ratings")
figure3.show()
#CONCLUSION:-1)There is an inverse linear relationship between the time taken to deliver the food and the ratings of the delivery partner.
            #2)It means delivery partners with higher ratings take less time to deliver the food compared to partners with low ratings.
fig = px.box(df, 
             x="Type_of_vehicle",
             y="Time_taken(min)", 
             color="Type_of_order")
fig.show()
#CONCLUSION:-SO there is not much difference between the time taken by delivery partners depending on the vehicle 
            #they are driving and the type of food they are delivering
# Final Conclusion
So the features that contribute most to the food delivery time based on our analysis are:

1)age of the delivery partner    
2)ratings of the delivery partner     
3)distance between the restaurant and the delivery location
# Food Delivery Time prediction model
pip install keras
pip install tensorflow
#splitting data
from sklearn.model_selection import train_test_split
x = np.array(df[["Delivery_person_Age", 
                   "Delivery_person_Ratings", 
                   "distance"]])
y = np.array(df[["Time_taken(min)"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

# creating the LSTM neural network model
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()
# training the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=9)
print("Food Delivery Time Prediction")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a, b, c]])

print("Predicted Delivery Time in Minutes = ", model.predict(features))
