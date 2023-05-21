# Programmer : Jsako
# Commented out IPython magic to ensure Python compatibility.

# Start imports ...
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
    precision_recall_curve,
    f1_score,
    precision_score
)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
# End imports .

# Data .
d = pd.read_csv('dataset-kidney-stone.csv')
d.info()

# Examine the distribution of each feature to see if it is skewed or has any outliers .
for i in d.columns:
  plt.boxplot(d[i])
  plt.title(i)
  plt.show()

# Data distribution and skewness .
for i in d.columns:
  sns.displot(d, x = i, kde = True)

"""It is made evident that only two features have few distinctive outliers . This being the case , outlier removal will not be performed as it is deemed unnecessary ."""

# Standardize the data to make sure all the variables are in the same format .
x = d.drop(['target', 'Unnamed: 0'], axis = 1)
y = d['target']

ss = StandardScaler()
x[['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']] = ss.fit_transform(d[['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']])

x[['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']] = x[['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']].apply(lambda y: y/y.max(), axis = 0)

x

# Split the dataset into training and testing sets using a reasonable ratio (80:20) .
# This will allow us to train our model on a portion of the data and evaluate its performance on unseen data .
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Using extreme gradient boosting (XGBoost) to predict the risk of kidney stones based on the input features .
# Using a Python library such as xgboost to build and train the model .
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(x_train, y_train)

y_pred = xgb_clf.predict(x_test)
xgboosts1, xgboosts2, _ = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print('ROC-AUC Score : ', roc_auc)
# Specified metrics .
print('Mean Absolute Error : ', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error : ', mean_squared_error(y_test, y_pred))
print('R^2 Score : ', r2_score(y_test, y_pred))

plt.plot(xgboosts1, xgboosts2, linestyle = '--', color = '#E6E6FA', label = 'XGB')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Set up a random search CV object and define the hyperparameter grid to search over , etc. .
rg = xgb.XGBClassifier()
pg = {
    'learning_rate': uniform(0, 1),
    'max_depth': randint(1, 10),
    'n_estimators': randint(100, 1000)
}

rs = RandomizedSearchCV(
    rg,
    pg,
    n_iter = 10,
    cv = 5,
    random_state = 42,
    n_jobs = -1
)
rs.fit(x_train, y_train)

bp = rs.best_params_

bpm = xgb.XGBClassifier(**bp, random_state = 42)
bpm.fit(x, y)

yp_train = bpm.predict(x_train)
yp_test = bpm.predict(x_test)
xgboosts1_test, xgboosts2_test, _ = roc_curve(y_test, yp_test)
roc_auc = roc_auc_score(y_test, yp_test)
# Test .
print(classification_report(y_test, yp_test))
print('Testing ROC-AUC Score : ', roc_auc)
# Specified metrics .
print('Testing Mean Absolute Error : ', mean_absolute_error(y_test, yp_test))
print('Testing Mean Squared Error : ', mean_squared_error(y_test, yp_test))
print('Testing R^2 Score : ', r2_score(y_test, yp_test))

xgb_bps, xgb_bpt, _ = roc_curve(y_train, yp_train)
roc_auc = roc_auc_score(y_train, yp_train)
# Train .
print(classification_report(y_train, yp_train))
print('Training ROC-AUC Score : ', roc_auc)
# Specified metrics .
print('Training Mean Absolute Error : ', mean_absolute_error(y_train, yp_train))
print('Training Mean Squared Error : ', mean_squared_error(y_train, yp_train))
print('Training R^2 Score : ', r2_score(y_train, yp_train))

plt.plot(xgboosts1_test, xgboosts2_test, linestyle = '--', color = '#D2B48C', label = 'XGB Test')
plt.plot(xgb_bps, xgb_bpt, linestyle = '--', color = '#008080', label = 'XGB Train')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

"""6. According to the results, XGBoost is very useful for estimating the likelihood that kidney stones may form. In this study, RandomizedSearchCV was used to optimize the hyperparameters of the XGBoost classifier. Then I fitted the classifier using the optimized parameters. The XGBoost model demonstrated exceptional performance with best parameters, indicating its potential as a trustworthy tool for estimating the risk of kidney stones. Additionally, the usefulness of the method is demonstrated by the fact that the optimal hyperparameters found can be used for future predictions. These findings lead to the conclusion that XGBoost is very useful for estimating the likelihood of kidney stone formation. Healthcare professionals can utilize XGBoost as a potent and precise technique to forecast the risk of kidney stones by using the ideal hyperparameters discovered."""
