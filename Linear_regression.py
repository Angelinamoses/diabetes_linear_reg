# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print("all good")

# %% [markdown]
# ## Loading the dataset

# %%
from sklearn.datasets import load_diabetes

# %%
diabetes = load_diabetes()

# %%
type(diabetes)

# %%
diabetes.keys()

# %%
## Let's check the description of the dataset
print(diabetes.DESCR)

# %%
print(diabetes)

# %%
print(diabetes.target)

# %% [markdown]
# ## Preparing the Dataset

# %%
dataset = pd.DataFrame(diabetes.data,columns= diabetes.feature_names)

# %%
dataset.head()

# %%
print(diabetes.target[:5])

# %%
dataset['target']= diabetes.target

# %%
dataset.head()

# %%
dataset.info()

# %%
## Summarizing The Stats of the data
dataset.describe()

# %%
## Checking the missing values
dataset.isnull().sum()

# %% [markdown]
# ### Exploratory Data Analysis

# %%
## Correlation
dataset.corr()

# %%
import seaborn as sns
sns.pairplot(dataset)

# %%
plt.scatter(dataset['age'],dataset['target'])
plt.xlabel("age")
plt.ylabel("target")

# %%
import seaborn as sns
sns.regplot(x="bmi", y="target",data=dataset)

# %%
sns.regplot(x="bp", y="target",data=dataset)

# %%
sns.regplot(x="age", y="target",data=dataset)

# %%
sns.regplot(x="s6", y="target",data=dataset)

# %%
## Independent and Dependent features
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

# %%
x.head()

# %%
y

# %%
## Train Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

# %%
x_train

# %%
x_test

# %%
y_train

# %%
y_test

# %%
## Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

# %%
x_train=scaler.fit_transform(x_train)

# %%
x_test=scaler.transform(x_test)

# %%
x_train

# %%
x_test

# %% [markdown]
# ## Model Training

# %%
from sklearn.linear_model import LinearRegression

# %%
# Create the regression model
regression=LinearRegression()

# %%
# Fitting the model with training data
regression.fit(x_train,y_train)

# %%
## print the coefficients and the intercept
print(regression.coef_)

# %%
print(regression.intercept_)

# %%
## on which parameters the model has been trained
regression.get_params()

# %%
### Prediction with Test Data
reg_pred=regression.predict(x_test)

# %%
reg_pred

# %% [markdown]
# # Assumptions

# %%
## plot a scatter plot for the prediction
plt.scatter(y_test,reg_pred)

# %%
## Residuals
residuals=y_test-reg_pred

# %%
residuals

# %%
## Plotting this Residuals
sns.displot(residuals,kind="kde")

# %%
## Scatter plot with respect to prediction and residuals
# Uniform distribution
plt.scatter(reg_pred,residuals)

# %%
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))

# %% [markdown]
#  ## R square and adjusted R square
# 
#  Formula
# 
#  # R^2 = 1 - SSR/SST
# 
#  R^2 = coefficient od determination SSR = sum of squares of residuals SST = total sum of squares

# %%
from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)

# %% [markdown]
# ## Adjusted R2 = 1 - [(1-R2)*(n-1)/(n-k-1)]
# where:
# 
# ## The R2 of the model n: The number of observations k: The number of predictor variables

# %%
#display adjusted R-Squared
1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)

# %% [markdown]
# ## New Data Prediction

# %%
new_data=diabetes.data[0].reshape(1,-1)

# %%
regression.predict(diabetes.data[0].reshape(1,-1))

# %%
##transformation of new data
scaler.transform(diabetes.data[0].reshape(1,-1))

# %%
regression.predict(scaler.transform(diabetes.data[0].reshape(1,-1)))

# %% [markdown]
# ## Pickling The Model file for Deployment

# %%
import pickle

# %%
with open('regmodel.pkl','wb')as f:
    pickle.dump(regression,f)

# %%
pickled_model=pickle.load(open('regmodel.pkl','rb'))

# %%
## Prediction
pickled_model.predict(scaler.transform(diabetes.data[0].reshape(1,-1)))

# %%


# %%


# %%


# %%


# %%


# %%

import pickle
import numpy as np

with open("regmodel.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded 🔥")

sample = np.array([[0.05, -0.02, 0.03, 0.01, -0.04, 0.02, 0.01, -0.03, 0.04, 0.02]])

prediction = model.predict(sample)

print("Prediction:", prediction)

features = list(map(float, input("Enter 10 values separated by space: ").split()))

prediction = model.predict([features])

print("Prediction:", prediction)