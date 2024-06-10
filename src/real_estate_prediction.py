# %% [markdown]
# # Introduction
# 
# One of the main aspects of a data scientist's work involves exploratory data analysis and data cleaning. To exercise this skill, I utilized a dataset wich provides data of the date of purchase, house age, location, distance to the nearest MRT station, and house price per unit area in Taiwan.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
dataframe = pd.read_csv("../database/real_estate.csv")
dataframe.head()

# %% [markdown]
# # Data Cleaning
# Before carrying out an profund analysis, it is necessary to clean the data.
# On this step, some columns will be droped since it won't be use and others will be renamed for a better understanding of the data.

# %%
dataframe.drop(columns=["No", "X1 transaction date"], inplace=True)
dataframe.rename(columns={"X2 house age": "house_age", "X3 distance to the nearest MRT station": "dist_nearest_mrt_station", 
                          "X4 number of convenience stores": "num_convenience_stores", "X5 latitude": "latitude", "X6 longitude": "longitude",
                            "Y house price of unit area": "house_price_unit_area"}, inplace=True)

# %% [markdown]
# For a better understand of the data distribuition, the common metrics will be evaluated

# %%
dataframe.describe()

# %% [markdown]
# The data sems to be very clear and ready for use.

# %% [markdown]
# # Exploratory Data Analysis
# For an initial analysis, we will seek to understand how our data would be organized spatially (according to latitude and longitude)

# %%
dataframe.plot(kind='scatter', x='longitude', y='latitude', alpha=0.5)

# %% [markdown]
# With the purpose of gain some insight's it will be plot a longitude x latitude graph with color variation according to house price

# %%
dataframe.plot(kind="scatter", x="longitude", y="latitude", alpha=0.5, c=dataframe["house_price_unit_area"], cmap=plt.get_cmap('jet'))

# %% [markdown]
# As a result, it can be seen that there is a interest variation in prices according to location. More expensive houses seems to be located more on the northwest side.
# 
# Let's investigate further.

# %%
correlation = dataframe.corr()
correlation["house_price_unit_area"].sort_values(ascending=False)

# %% [markdown]
# It can be seen in the table above that there is a significant correlation between the price and the distance to the nearest metro station where when the value of the distance to the nearest metro station decrease, the house prince of unit area increase. 
# 
# Beside that, there is a moderate correlation between the number of convenience stores and the house price.

# %% [markdown]
# # Simple Linear Regression
# 
# Now that the data already was studied and understand, a machine learning model will be implemented to try predict the house prices based on one feature, distance to the nearest metro station. For this, it will be use a simple linear regression algorithm.

# %% [markdown]
# ## Training Set
# 
# It is necessary to separate a training set and a test set. The training set should be use to train the learning algorithm and the test set will be for test if the algorithm is perfoming as it should.

# %%
x = dataframe[['house_age', 'dist_nearest_mrt_station', 'num_convenience_stores', 'latitude', 'longitude']]
y = dataframe['house_price_unit_area']

# %% [markdown]
# ## Z-Score Normalization
# I am dealing with different range of data. To prevent this from have a big impact on my prediction, I will implement the z-score normalization.

# %%
def z_score_normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    standarlized_column = (x - mean) / std
    
    return standarlized_column

# %%
x_train = z_score_normalize(dataframe['dist_nearest_mrt_station'])

# %% [markdown]
# Since the main goal is to build a simple linear regression model to fit this data. A first step on this is to define a cost function. This function tells us how well the model is performing so that we can try to improve it. Essentially, this cost function calculates the difference, known as the error, between ŷ (the predicted value) and y (the actual target value). 
#  
# For a linear regression with one variable, the prediction of the model will be a linear function.

# %%
def cost_function(x: pd.Series, y: pd.Series, w: float, b: float) -> float:
    number_of_samples = len(x)

    total_cost = 0
    sum_cost = 0

    for i in range(number_of_samples):
        f = (w*x[i]) + b
        current_cost = (f - y[i])**2

        sum_cost += current_cost
    
    total_cost = sum_cost/(2*number_of_samples)

    return total_cost

# %% [markdown]
# 
# In Linear Regression, the goal is to find the values of w and b that minimize the value of J. Achieving the minimum value for J is an indication that the model fits the data relatively well.
#  
# To acomplish this purpose, it will be implemented a Gradient Descendent algorithm. For this I will be inicializing w and b as zero and make changes to their values until the appropriate value is reached where the minimum possible value of the cost function is achieved. 
# 

# %%
def compute_gradient(x: pd.Series, y: pd.Series, w: float, b: float):
    number_of_samples = len(x)

    dj_dw = 0
    dj_db = 0

    for id_sample in range(number_of_samples):
        f_wb = (w*x[id_sample])+b

        dj_dw_i = (f_wb - y[id_sample])*x[id_sample]
        dj_db_i = (f_wb - y[id_sample])

        dj_dw += dj_dw_i
        dj_db += dj_db_i
    
    dj_dw = dj_dw/number_of_samples
    dj_db = dj_db/number_of_samples

    return dj_dw, dj_db 

# %% [markdown]
# To find the optimal parameters (w, b) of the linear regression, it will be use a batch gradient descent. 
#  
# In the following cell, the parameters (w, b) will be update by alpha (learning rate) and the calculus of the gradient.

# %%
def gradient_descent(x: pd.Series, y: pd.Series, initial_w: float, initial_b :float, cost_function, compute_gradient, alpha: float, num_iters: int):

    J_history = []
    w_history = []
    w = initial_w
    b = initial_b 

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b) 

        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db) 

        # Save cost J at each iteration
        if i<100000:      #Prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        
        if i%(num_iters/10) == 0: # Print cost every at intervals 10 times or as many iterations if < 10
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
    
    plt.plot(range(len(J_history)), J_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function J')
    plt.title('Cost Function J vs. Iteration')
    plt.grid(True)
    plt.show()
    
    return w, b, J_history, w_history

# %% [markdown]
# Now I will try to predict the house price per unit area by the distance to the nearest metro station. For this, I will generate a simple array.
# 
# The frist step on this task is to find the optimal values for cost function parameters w and b.

# %%
initial_w = 0.
initial_b = 0.

num_iterations = 1500
alpha = 0.01

w, b, _, _ = gradient_descent(x_train, y, initial_w, initial_b, cost_function, compute_gradient, alpha, num_iterations)

print(f"w, b foung by gradient descent: {w}, {b}")

# %%
m = len(x_train)
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

# %%
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y, marker='x', c='r') 

# Set the title
plt.title("House Price vs. Distance to MRT")
# Set the y-axis label
plt.ylabel('House Price of Unit Area')
# Set the x-axis label
plt.xlabel('Distance to Nearest MRT in 1.000m')

# %% [markdown]
# For provide another analysis, I will make a prediction of the house price based on the number of conveniences stores around.

# %%
x_train = z_score_normalize(dataframe['num_convenience_stores'])

initial_w = 0.
initial_b = 0.

num_iterations = 1500
alpha = 0.01

w, b, _, _ = gradient_descent(x_train, y, initial_w, initial_b, cost_function, compute_gradient, alpha, num_iterations)

print(f"w, b foung by gradient descent: {w}, {b}")

# %%
m = len(x_train)
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

# %%
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y, marker='x', c='r') 

# Set the title
plt.title("House Price vs. Number of Convenience Stores")
# Set the y-axis label
plt.ylabel('House Price of Unit Area')
# Set the x-axis label
plt.xlabel('Number of Convenience Stores')

# %% [markdown]
# # Conclusion
# Based on the explored data, it is evident that there is a significant correlation between the distant to the nearest metro and the house prince. The value of the house tends to decrease as it is farther from the metro station and to encrease with more convenience stores around.


