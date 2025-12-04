import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

X, y = make_regression(n_samples=1000, n_features=3, noise=5, random_state=42)

df = pd.DataFrame(X, columns=['Attendance', 'Assignments', 'Internal_Test'])
df['Final_Score'] = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeRegressor(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

ada_model = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4), n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)

mse_ada = mean_squared_error(y_test, y_pred_ada)
mae_ada = mean_absolute_error(y_test, y_pred_ada)

print("--- Decision Tree Results ---")
print(f"MSE: {mse_dt:.2f}")
print(f"MAE: {mae_dt:.2f}")

print("\n--- AdaBoost Results ---")
print(f"MSE: {mse_ada:.2f}")
print(f"MAE: {mae_ada:.2f}")

metrics = ['MSE', 'MAE']
dt_scores = [mse_dt, mae_dt]
ada_scores = [mse_ada, mae_ada]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, dt_scores, width, label='Decision Tree', color='red')
plt.bar(x + width/2, ada_scores, width, label='AdaBoost', color='green')

plt.ylabel('Error (Lower is Better)')
plt.title('Decision Tree vs AdaBoost Performance')
plt.xticks(x, metrics)
plt.legend()
plt.show()
