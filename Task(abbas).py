import pandas as pd
from sklearn.model_selection import train_test_split

#Import Dataset
df = pd.read_csv('student_scores - student_scores.csv')
print(df)
#Checking the null values
print(df.isnull().sum())
# Splitting the dataset into dependent and independent variable
y=df.pop('Hours')
y=pd.DataFrame(y,columns=['Hours'])
x=df
print(y)


#split data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=30)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model=lr.fit(x_train, y_train)

# You can also test with your own data
hours = 9.25
own_pred = model.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))