import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Read the dataset from the URL
from js import fetch
import io

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'

response = requests.get(URL)
boston_url = io.BytesIO(response.content)
boston_df = pd.read_csv(boston_url)

# Q1: Boxplot for "Median value of owner-occupied homes"
ax = sns.boxplot(y='MEDV', data=boston_df)
ax.set_title('Owner-occupied homes')
plt.show()

# Q2: Bar plot for the Charles river variable
ax2 = sns.countplot(x='CHAS', data=boston_df)
ax2.set_title('Number of homes near the Charles River')
plt.show()

# Q3: Boxplot for MEDV vs AGE variable
boston_df['Age_Group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, float('inf')], labels=['35 years and younger', 'between 35 and 70 years', '70 years and older'])
ax3 = sns.boxplot(x='Age_Group', y='MEDV', data=boston_df)
ax3.set_title('Median value of owner-occupied homes per Age Group')
plt.show()

# Q4: Scatter plot for Nitric oxide concentrations vs proportion of non-retail business acres per town
ax4 = sns.scatterplot(x='INDUS', y='NOX', data=boston_df)
ax4.set_title('Nitric oxide concentration per proportion of non-retail business acres per town')
plt.show()

# Q5: Histogram for the pupil to teacher ratio variable
ax5 = sns.histplot(x='PTRATIO', data=boston_df)
ax5.set_title('Pupil to teacher ratio per town')
plt.show()

# Q6: T-test for median value of houses bounded by the Charles river or not
scipy.stats.ttest_ind(boston_df[boston_df['CHAS'] == 0]['MEDV'], boston_df[boston_df['CHAS'] == 1]['MEDV'])

# Q7: ANOVA for Median values of houses for each proportion of owner-occupied units built prior to 1940 (AGE)
lm = ols('MEDV ~ AGE', data=boston_df).fit()
table = sm.stats.anova_lm(lm)
print(table)

# Q8: Pearson Correlation for Nitric oxide concentrations and proportion of non-retail business acres per town
scipy.stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])

# Q9: Regression analysis for the impact of an additional weighted distance to the five Boston employment centres on the median value of owner-occupied homes
x = boston_df['DIS']
y = boston_df['MEDV']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())
