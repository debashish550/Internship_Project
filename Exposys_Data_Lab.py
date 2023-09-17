#!/usr/bin/env python

# coding: utf-8
#####            Versions
# pandas                        1.5.3
# matplotlib                    3.7.0
# seaborn                       0.12.2
# numpy                         1.23.5
# scikit-learn                  1.2.1

# In[16]: DATA EXPLORATION 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('50_Startups.csv')

# Display basic statistics of the dataset
print(data.describe())

# Pairplot to visualize relationships between numeric variables
sns.pairplot(data, diag_kind='kde')
plt.show()

# Correlation heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Box plots for each numeric feature
numeric_columns = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=col, data=data, orient='v', width=0.5)
    plt.title(f'Boxplot of {col}')
    plt.show()

# Histograms of each numeric feature
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], bins=20, kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Scatter plots of the independent variables vs. profit
sns.scatterplot(x="R&D Spend", y="Profit", data = data)
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("R&D Spend vs. Profit")
plt.show()

sns.scatterplot(x="Administration", y="Profit", data= data)
plt.xlabel("Administration")
plt.ylabel("Profit")
plt.title("Administration vs. Profit")
plt.show()

sns.scatterplot(x="Marketing Spend", y="Profit", data= data)
plt.xlabel("Marketing Spend")
plt.ylabel("Profit")
plt.title("Marketing Spend vs. Profit")


# In[44]: without using any data preprocessing, hyperparameter tuning, feature selection techniques:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset 
data = pd.read_csv('50_Startups.csv')


# Separate the target variable ('Profit') from the features
X = data.drop(columns=['Profit'])
y = data['Profit']


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary to store results
results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Linear Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Lasso Regression
lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Lasso Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Ridge Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Support Vector Regression
svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["SVR"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Decision Tree Regression
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Decision Tree Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Random Forest Regression
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Random Forest Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Neural Network Regression
nn = MLPRegressor()
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Neural Network Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Display results
for model, metrics in results.items():
    print(f"{model} Metrics:")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"R2: {metrics['R2']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print("\n")


# In[45]:using data preprocessing techniques (Min-Max scaling, Z-score scaling):


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset 
data = pd.read_csv('50_Startups.csv')


# Separate the target variable ('Profit') from the features
X = data.drop(columns=['Profit'])
y = data['Profit']

# Normalize the features using Min-Max scaling
min_max_scaler = MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X)

# Standardize the features using Z-score scaling
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X_normalized)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

# Create a dictionary to store results
results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Linear Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Lasso Regression
lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Lasso Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Ridge Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Support Vector Regression
svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["SVR"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Decision Tree Regression
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Decision Tree Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Random Forest Regression
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Random Forest Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Neural Network Regression
nn = MLPRegressor()
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Neural Network Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Display results
for model, metrics in results.items():
    print(f"{model} Metrics:")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"R2: {metrics['R2']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print("\n")


# In[29]:using data preprocessing techniques (Min-Max scaling, Z-score scaling) and hyperparameter tuning (Cross validation, Grid Search)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset 
data = pd.read_csv('50_Startups.csv')

# Separate the target variable ('Profit') from the features
X = data.drop(columns=['Profit'])
y = data['Profit']

# Normalize the features using Min-Max scaling
min_max_scaler = MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X)

# Standardize the features using Z-score scaling
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X_normalized)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

# Create a dictionary to store results
results = {}

# List of models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(),
    "Ridge Regression": Ridge(),
    "SVR": SVR(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(),
    "Neural Network Regression": MLPRegressor()
}

# Loop through each model
for model_name, model in models.items():
    # Define hyperparameters and their possible values for grid search
    param_grid = {}
    if model_name == "Linear Regression":
        param_grid = {'fit_intercept': [True, False]}
    
    # Create GridSearchCV instance
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Fit the model with cross-validation
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate the best model using cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    # Calculate metrics
    mse = -cv_scores.mean()
    r2 = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2').mean()
    
    # Fit the best model on the test set
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Store the results
    results[model_name] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}
    
# Display results
for model, metrics in results.items():
    print(f"{model} Metrics:")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"R2: {metrics['R2']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print("\n")


# In[43]: using data preprocessing techniques (Min-Max scaling, Z-score scaling) and hyperparameter tuning (Cross validation, Grid Search) and 
# dropping the ‘administration’ feature:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load  dataset 
data = pd.read_csv('50_Startups.csv')

# Drop the 'Administration' column
data = data.drop(columns=['Administration'])

# Separate the target variable ('Profit') from the features
X = data.drop(columns=['Profit'])
y = data['Profit']

# Normalize the features using Min-Max scaling
min_max_scaler = MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X)

# Standardize the features using Z-score scaling
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X_normalized)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

# Create a dictionary to store results
results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Linear Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Lasso Regression
lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Lasso Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Ridge Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Support Vector Regression
svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["SVR"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Decision Tree Regression
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Decision Tree Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Random Forest Regression
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Random Forest Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Neural Network Regression
nn = MLPRegressor()
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
results["Neural Network Regression"] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

# Display results
for model, metrics in results.items():
    print(f"{model} Metrics:")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"R2: {metrics['R2']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print("\n")







