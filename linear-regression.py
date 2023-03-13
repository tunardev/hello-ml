import numpy as np
import pandas as pd
import sklearn.linear_model as ml
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import sklearn.metrics as sm

data = pd.read_csv('data.csv')

experience = data['experience'].values.reshape(-1, 1)
salary = data['salary'].values.reshape(-1, 1)

model = ml.LinearRegression()

x_train, x_test, y_train, y_test = ms.train_test_split(experience, salary, test_size=0.22, random_state=0)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('experiences: \n', x_test)
print('new salaries: \n', y_pred)

score = sm.r2_score(y_test, y_pred)
print('R2 Score: ', score)

plt.scatter(experience, salary, color='red')
plt.scatter(x_test, y_pred, color='blue')
plt.show()