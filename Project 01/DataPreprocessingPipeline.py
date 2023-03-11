# Programmer : Jsako
# Commented out IPython magic to ensure Python compatibility.
# Part ( 1 ) ...

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

stroke_data = pd.read_csv('/content/healthcare-dataset-stroke-data.csv')

stroke_data.describe()

"""**myObservations :**

>

*   The average BMI is approximately 29 . A BMI of 29 falls in the overweight category according to the World Health Organization (WHO) classification .
*   Of all patients recorded in the dataset , the oldest aged individuals are 82 years old . This may be identified via the *max* row and *age* column .
*   The hypertension , heart_disease , and stroke attributes are 0 for the 25% , 50% , and 75% titles . From this information , we may conclude that more patients had not encountered them .
*   The count title indicates 5110.000000 for each attribute other than for BMI which is 4909.000000 .
*   The lowest recorded average glucose level rounds to 55 . This is lower than the normal range , which is typically between 70 and 99 .
"""

# Part ( 2 ) ...

for i in stroke_data.columns[1:]:
  if (stroke_data[i].dtype == 'object') or (stroke_data[i].dtype == 'int64'):
    sns.countplot(data = stroke_data, x = i, hue = 'stroke')
    plt.title('The number of the samples with {} based on stroke .'.format(i))
    plt.show()

# Part ( 3 ) ...

sns.set(rc = {'figure.figsize':(18,10)})
sns.FacetGrid(stroke_data, hue = 'stroke', height = 8).map(sns.distplot, "age").add_legend()
plt.title("Univariate distribution plot for patients' age .")
plt.show()

"""**Description of Results :**

>

It may be concluded here that the likelyhood of a stroke occuring increases as does age .
"""

# Part ( 4 ) ...

sns.set(rc = {'figure.figsize':(18,10)})
seaborn_plot = sns.violinplot(x = 'stroke', y = 'age', data = stroke_data)
seaborn_plot.set_xlabel("Stroke", fontsize = 15)
seaborn_plot.set_ylabel("Age of Patient", fontsize = 15)

"""It may be concluded here that the likelyhood of a stroke occuring is at its peak when the age of the patient ranges from 75 to 80 years old ."""

# Part ( 5 ) ...

"""Based on what is seen of the number of individuals who have and have not had a stroke , it seems as though this dataset may be imbalanced . This is evident by the miniscule range in age associated to those who are more likely to have a stroke ."""

# Part ( 6 ) ...

fig, ax = plt.subplots(figsize = (7, 7))
heatmap = sns.heatmap(stroke_data[['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke']].corr(), vmax = 1, annot = True)
heatmap.set_title('Correlation Heatmap')

stroke_data[1:].corr()

"""Based on the definition of when things are positively correlated , or not correlated , the variables that have the strongest correlation are age and BMI . It may also be noted that age , avg_glucose_level , BMI , hypertension , and heart_disease have a strong correlation as well ."""

# Part ( 7 , 8 , 9 ) ...

plt.figure(figsize = (10,7))
sns.boxplot(data = stroke_data, x = stroke_data["bmi"], color = 'orange');

plt.figure(figsize = (10,7))
sns.boxplot(data = stroke_data, x = stroke_data["avg_glucose_level"], color = 'orange');

bmi_outliers = stroke_data.loc[stroke_data['bmi'] > 50]
avg_glucose_level = stroke_data.loc[stroke_data['avg_glucose_level'] > 175]

stroke_data["bmi"] = stroke_data["bmi"].apply(lambda x: 50 if x > 50 else x)
stroke_data["avg_glucose_level"] = stroke_data["avg_glucose_level"].apply(lambda x: 175 if x > 175 else x)

stroke_data.dtypes

stroke_data['bmi'] = stroke_data['bmi'].fillna(stroke_data['bmi'].mean())

le = LabelEncoder()
i = 0
for col_name in stroke_data.columns[i:]:
  if (stroke_data[col_name].dtype == 'object'):
    stroke_data[col_name] = le.fit_transform(stroke_data[col_name])

stroke_data.head(3)

le.inverse_transform(stroke_data['Residence_type'].unique())
np.array(['formerly smoked', 'Unknown'], dtype = object)
