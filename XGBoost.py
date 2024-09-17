import numpy as np
import pandas as pd




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
   for filename in filenames:
       print(os.path.join(dirname, filename))


df = pd.read_csv('/Users/unmol.sharma/Desktop/ Math IA downloads/indian_rda_based_diet_recommendation_system.csv')
df
df.info()
df.isna().sum()




df.loc[0, 'VegNovVeg'] = 0




df['VegNovVeg'] = df['VegNovVeg'].astype('int64')


import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm




features = ['Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 'Potassium', 'Carbohydrates', 'Fibre', 'Sugars']




for feature in features:
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))


   sns.histplot(df[feature], ax=axes[0], kde=True)
   axes[0].set_title(f'Distribution Plot of {feature}')


   sm.qqplot(df[feature], line='s', ax=axes[1])
   axes[1].set_title(f'QQ Plot of {feature}')


   sns.boxplot(x=df[feature], ax=axes[2])
   axes[2].set_title(f'Box Plot of {feature}')


   plt.tight_layout()




   plt.show()


# Data is Right Skewed and outliers exist in upper bound most
## Apply Yeo-johnsons transfomration from power transformer sklearn to normalize the distribution a bit


from sklearn.preprocessing import PowerTransformer


selected_features = ['Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 'Potassium', 'Carbohydrates', 'Fibre', 'Sugars']




pt = PowerTransformer(method='yeo-johnson')




df[selected_features] = pt.fit_transform(df[selected_features])








features = ['Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 'Potassium', 'Carbohydrates', 'Fibre', 'Sugars']




for feature in features:
  
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))


   sns.histplot(df[feature], ax=axes[0], kde=True)
   axes[0].set_title(f'Distribution Plot of {feature}')


   sm.qqplot(df[feature], line='s', ax=axes[1])
   axes[1].set_title(f'QQ Plot of {feature}')


   sns.boxplot(x=df[feature], ax=axes[2])
   axes[2].set_title(f'Box Plot of {feature}')


  
   plt.tight_layout()




   plt.show()
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor, plot_tree, plot_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score




X = df.drop(columns=['Food_items', 'Calories'])
Ey = df['Calories']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_regressor = XGBRegressor()




param_grid = {
   'max_depth': [3, 5, 7, 10],
   'learning_rate': [0.01, 0.05, 0.1, 0.2],
   'n_estimators': [100, 200, 300, 400]
}
print("Install graphviz god dammit")
xgb_grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
xgb_grid_search.fit(X_train, y_train)


# Plot the first decision tree from the best model
best_xgb_model = xgb_grid_search.best_estimator_
plt.figure(figsize=(50, 10))  # You can adjust the size to fit your needs
plot_tree(best_xgb_model, num_trees=0, rankdir='LR')  # 'LR' is for left to right tree layout
plt.show()




# Plotting feature importance
fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(best_xgb_model, ax=ax)
plt.show()


# Get best parameters for XGBoost Regressor
best_params = xgb_grid_search.best_params_


# Predict on the test set
y_pred = xgb_grid_search.predict(X_test)




mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)




print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
print("Mean Absolute Error (MAE):", mae)
print("Explained Variance Score (EVS):", evs)






# Actual vs Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test, color='blue', label='Actual', alpha=0.5)
plt.scatter(y_test, y_pred, color='red', label='Predicted', alpha=0.5)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Actual vs Predicted Calories')
plt.legend()
plt.grid(True)
plt.show()
