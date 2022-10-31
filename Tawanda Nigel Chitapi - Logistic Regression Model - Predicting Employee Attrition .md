### Project: Understanding Employee Attrition with Machine Learning 
### Activity: Feature Selection & Logistic Regression Modelling - Model Evaluation on Up Sampled Data
### Model: Logistic Regression
### Author: Tawanda Nigel Chitapi
### Email: nigel.chitapi@gmail.com    
### Date: September 05, 2022
### Institution: BrainStation


```python
# import packages

# the data science trinity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn
import seaborn as sns

# model selection tools
from sklearn.model_selection import train_test_split

# scaler
from sklearn.preprocessing import StandardScaler

# linear models
from sklearn.linear_model import LogisticRegression

# metrics
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_curve, roc_auc_score

# SMOTE
from imblearn.over_sampling import SMOTE
```

### The goal of this notebook is achieve the following:

- Perform Feature Selection to pick only those features that are best predictors of the target variable

- Check if target feature classes are balanced, if they are not balanced, we ought to perform some data upsampling

- Fit data into a Logistic Regression Model and determine accuracies

- Evaluate overall model performance 



```python
# load our data 

employee_df = pd.read_csv('data/employee_attrition.csv')
```


```python
# sanity check to see if our data loaded successfully

employee_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmployeeID</th>
      <th>Age</th>
      <th>Attrition</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>HourlyRate</th>
      <th>...</th>
      <th>Other</th>
      <th>Technical Degree</th>
      <th>Admin</th>
      <th>Administrative</th>
      <th>Nurse</th>
      <th>Other.1</th>
      <th>Therapist</th>
      <th>Divorced</th>
      <th>Married</th>
      <th>Single</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1313919</td>
      <td>41</td>
      <td>0</td>
      <td>1102</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>94</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200302</td>
      <td>49</td>
      <td>0</td>
      <td>279</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>61</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1060315</td>
      <td>37</td>
      <td>1</td>
      <td>1373</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>92</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1272912</td>
      <td>33</td>
      <td>0</td>
      <td>1392</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>56</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1414939</td>
      <td>27</td>
      <td>0</td>
      <td>591</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>40</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>




```python
# sanity check, to ensure all data types are numerical 
employee_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1676 entries, 0 to 1675
    Data columns (total 50 columns):
     #   Column                    Non-Null Count  Dtype
    ---  ------                    --------------  -----
     0   EmployeeID                1676 non-null   int64
     1   Age                       1676 non-null   int64
     2   Attrition                 1676 non-null   int64
     3   DailyRate                 1676 non-null   int64
     4   DistanceFromHome          1676 non-null   int64
     5   Education                 1676 non-null   int64
     6   EmployeeCount             1676 non-null   int64
     7   EnvironmentSatisfaction   1676 non-null   int64
     8   Gender                    1676 non-null   int64
     9   HourlyRate                1676 non-null   int64
     10  JobInvolvement            1676 non-null   int64
     11  JobLevel                  1676 non-null   int64
     12  JobSatisfaction           1676 non-null   int64
     13  MonthlyIncome             1676 non-null   int64
     14  MonthlyRate               1676 non-null   int64
     15  NumCompaniesWorked        1676 non-null   int64
     16  Over18                    1676 non-null   int64
     17  OverTime                  1676 non-null   int64
     18  PercentSalaryHike         1676 non-null   int64
     19  PerformanceRating         1676 non-null   int64
     20  RelationshipSatisfaction  1676 non-null   int64
     21  StandardHours             1676 non-null   int64
     22  Shift                     1676 non-null   int64
     23  TotalWorkingYears         1676 non-null   int64
     24  TrainingTimesLastYear     1676 non-null   int64
     25  WorkLifeBalance           1676 non-null   int64
     26  YearsAtCompany            1676 non-null   int64
     27  YearsInCurrentRole        1676 non-null   int64
     28  YearsSinceLastPromotion   1676 non-null   int64
     29  YearsWithCurrManager      1676 non-null   int64
     30  Non-Travel                1676 non-null   int64
     31  Travel_Frequently         1676 non-null   int64
     32  Travel_Rarely             1676 non-null   int64
     33  Cardiology                1676 non-null   int64
     34  Maternity                 1676 non-null   int64
     35  Neurology                 1676 non-null   int64
     36  Human Resources           1676 non-null   int64
     37  Life Sciences             1676 non-null   int64
     38  Marketing                 1676 non-null   int64
     39  Medical                   1676 non-null   int64
     40  Other                     1676 non-null   int64
     41  Technical Degree          1676 non-null   int64
     42  Admin                     1676 non-null   int64
     43  Administrative            1676 non-null   int64
     44  Nurse                     1676 non-null   int64
     45  Other.1                   1676 non-null   int64
     46  Therapist                 1676 non-null   int64
     47  Divorced                  1676 non-null   int64
     48  Married                   1676 non-null   int64
     49  Single                    1676 non-null   int64
    dtypes: int64(50)
    memory usage: 654.8 KB


### Now we will dive into our Logistic Regression modelling journey

## Logistic Regression

Logistic regression is one of the most basic (yet effective) tools we have for classifying categorical data.


With *linear* regression, we model data using linear equations of the form: 

$$f(X) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_d x_d$$

This equation outputs numerical predictions from numerical input attributes. Each $\beta_i$ represents how much our model weighs $x_i$ in predicting the output values. 

We will modify this function to get *categorical* predictions 

With logistic regression, we take advantage of the **sigmoid curve**:
$$ s(X) = \frac{1}{1+e^{-x}} $$

where $e$ is the value $e = 2.71828182845...$ and is often called [**Euler's Number**]


```python
# defining the sigmoid curve manually
def sigmoid_curve(x):
    return 1 / (1 + np.exp(-x)) # exp(x) is the function to calculate e^x

# for many x values, we will calculate s(x)
x_points = np.arange(-7, 7, 0.01)

plt.figure()
plt.plot(x_points, sigmoid_curve(x_points))
plt.grid()
plt.xlim(-7,7)
plt.show()
```


    
![png](output_8_0.png)
    


We take the linear function...
$$ f(X) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_d x_d $$

... and feed it into the sigmoid curve...
$$ s(X) = \frac{1}{1+e^{-f(X)}} $$

... to create our final predictive model:
$$ s(X) = \frac{1}{1+e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_d x_d)}} $$


Now our model has two nice properties:

   1. It is bounded between 0 and 1 (just like a probability)
   2. It is smooth, it never makes a sharp jump
    
* The first property means that we can interpret $s(X)$ as the probability that $X$ belongs to a particular class (say class $1$). By extension, this also gives us the probability that $X$ belongs to the other class, $1-s(X)$. 

* The second property is not as directly useful to us, but it makes it easy for Python to solve several underlying equations when fitting the model, (e.g. the gradient of the function).

Once we have our model's estimate we can make a decision on which class it belongs to. 

Since $X$ must belong to one of the two classes, we assign it to the class our model believes is more likely. This is to say that we assign it to class $1$ if $s(X) \geq 1-s(X)$, otherwise we assign it to class $0$. 

For example, in our case, we want to predict if an employee will attrition or not. Attrition is class 1 and staying on the job is class 0. Our model tells us: 

|    Class  |  0  |  1  | 
| --------- |:---:|:---:|
|Probability|0.73 |0.27 | 

These are called **soft predictions.** This means that there a 0.73 probability that the employee will **stay on the job** and a 0.27 probability that the employee **will attrtion**. These two should addup to **1.**  

We now have to make the **hard prediction** of saying if that the employee will **stay on the job. Since Class 0 is more likely than Class 1, we can make a decision to predict that the individual will stay on the job.** 

Equivalently, in a two-class scenario, we can look at the probability of Class 1 happening and predict Class 1 if the probability is larger than 0.5.

# First we will distinguish the Independent Features "X" and the Dependent Feature "Y"


```python
# independent features
X = employee_df.drop(['Attrition'], axis=1)

# dependent feature (Attrition)
y = employee_df['Attrition']
```


```python
# check the independent features

X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmployeeID</th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>...</th>
      <th>Other</th>
      <th>Technical Degree</th>
      <th>Admin</th>
      <th>Administrative</th>
      <th>Nurse</th>
      <th>Other.1</th>
      <th>Therapist</th>
      <th>Divorced</th>
      <th>Married</th>
      <th>Single</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1313919</td>
      <td>41</td>
      <td>1102</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>94</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200302</td>
      <td>49</td>
      <td>279</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>61</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1060315</td>
      <td>37</td>
      <td>1373</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>92</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1272912</td>
      <td>33</td>
      <td>1392</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>56</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1414939</td>
      <td>27</td>
      <td>591</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>40</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1671</th>
      <td>1117656</td>
      <td>26</td>
      <td>471</td>
      <td>24</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>66</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1672</th>
      <td>1152327</td>
      <td>46</td>
      <td>1125</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>94</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>1812428</td>
      <td>20</td>
      <td>959</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>83</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1674</th>
      <td>1812429</td>
      <td>39</td>
      <td>466</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>65</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1675</th>
      <td>1152329</td>
      <td>27</td>
      <td>511</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>89</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1676 rows × 49 columns</p>
</div>




```python
# check the dependent feature 'attrition'

y
```




    0       0
    1       0
    2       1
    3       0
    4       0
           ..
    1671    1
    1672    0
    1673    0
    1674    0
    1675    0
    Name: Attrition, Length: 1676, dtype: int64



### Now we will move on to perform Feature Selection 

- Our Independent data contains 49 columns, 49 columns are too wide a dimension and puts our model at the risk of the curse of dimensionality especially given that we only have 1676 rows.

- In order to best predict our target variable class, we need to conduct some feature selection and select only those variables that best predict the target variable class.

- To achieve this we will use the Fisher - Chi-Squared test and assess the derived p-values to select on the best predictors which are features with p-values less than the significance threshold of 0.05



```python
# import the chi2 feature selection package

from sklearn.feature_selection import chi2
```


```python
# create the chi2 variable and apply on the X and y variables
f_p_values = chi2(X, y)

f_p_values
```




    (array([1.21143359e+03, 2.18075488e+02, 9.80042735e+02, 1.34752206e+02,
            9.14675779e-01, 0.00000000e+00, 7.62324266e+00, 4.26023102e-01,
            1.37660534e+01, 8.64185842e+00, 4.33138845e+01, 4.99790687e+00,
            2.15238646e+05, 1.25027767e+04, 1.15322155e+00, 0.00000000e+00,
            1.36592523e+02, 1.26983331e-02, 7.82147747e-03, 2.99901161e-01,
            0.00000000e+00, 3.81226029e+01, 4.97310855e+02, 2.98027378e+00,
            2.44735458e+00, 3.59223808e+02, 2.23346033e+02, 5.90028751e+01,
            2.07527896e+02, 1.08669784e+00, 1.07867314e+01, 1.71639456e+00,
            2.15864231e+00, 1.45973417e-01, 5.70868185e+00, 2.15412094e+00,
            2.11386774e-02, 1.56263236e+00, 2.29483170e+00, 6.51176112e-01,
            1.19062460e+00, 2.15572106e+00, 1.33078829e+01, 1.02725034e+00,
            9.96392145e+00, 1.71956394e+01, 1.09284509e+01, 1.20169006e+01,
            4.95438134e+01]),
     array([1.99742170e-265, 2.37780925e-049, 3.91110641e-215, 3.74056454e-031,
            3.38876833e-001, 1.00000000e+000, 5.76207893e-003, 5.13946832e-001,
            2.07043843e-004, 3.28526142e-003, 4.66263191e-011, 2.53779918e-002,
            0.00000000e+000, 0.00000000e+000, 2.82875781e-001, 1.00000000e+000,
            1.48051863e-031, 9.10278803e-001, 9.29527672e-001, 5.83944390e-001,
            1.00000000e+000, 6.64360504e-010, 3.65659657e-110, 8.42850144e-002,
            1.17723119e-001, 4.15525833e-080, 1.68481749e-050, 1.57441751e-014,
            4.75580191e-047, 2.97204021e-001, 1.02230239e-003, 1.90157807e-001,
            1.41769907e-001, 7.02413392e-001, 1.68812104e-002, 1.42187800e-001,
            8.84401805e-001, 2.11280207e-001, 1.29805281e-001, 4.19692502e-001,
            2.75203604e-001, 1.42039749e-001, 2.64292606e-004, 3.10805360e-001,
            1.59637668e-003, 3.37209504e-005, 9.46986434e-004, 5.27203008e-004,
            1.93990632e-012]))



The variable **f_p_values** refers to f-score & p-value values, the first array [0] represents f-scores and the second array [1] represent the p-values.


```python
# our significance threshold is 0.05 and so we will only select those features with p-values less than 0.05
p_values = f_p_values[1]< 0.05
```


```python
# create a series of p_values and match them to the relevant columns 

selected_features = pd.Series(p_values, index = X.columns)
```


```python
# sort the p_values in descending order 
selected_features.sort_values(ascending = False , inplace = True)
```


```python
# display features
selected_features 
```




    EmployeeID                   True
    OverTime                     True
    Married                      True
    Divorced                     True
    Therapist                    True
    Other.1                      True
    Administrative               True
    Neurology                    True
    Travel_Frequently            True
    YearsWithCurrManager         True
    YearsSinceLastPromotion      True
    YearsInCurrentRole           True
    YearsAtCompany               True
    Age                          True
    TotalWorkingYears            True
    Shift                        True
    Single                       True
    JobSatisfaction              True
    HourlyRate                   True
    DailyRate                    True
    DistanceFromHome             True
    MonthlyRate                  True
    MonthlyIncome                True
    JobLevel                     True
    JobInvolvement               True
    EnvironmentSatisfaction      True
    Medical                     False
    Other                       False
    Technical Degree            False
    Admin                       False
    PercentSalaryHike           False
    Life Sciences               False
    Nurse                       False
    EmployeeCount               False
    Education                   False
    Marketing                   False
    Maternity                   False
    Human Resources             False
    Gender                      False
    PerformanceRating           False
    Cardiology                  False
    Travel_Rarely               False
    Non-Travel                  False
    TrainingTimesLastYear       False
    NumCompaniesWorked          False
    Over18                      False
    StandardHours               False
    RelationshipSatisfaction    False
    WorkLifeBalance             False
    dtype: bool



- We will go on and create a new subset of X features, using only the features with **"True"** values, meaning they have p-values less than 0.05. These are our best predictors.




```python
# we will create a new variable for X, with our newly identified features, those with p-values < 0.05

X = X[['EmployeeID', 'OverTime', 'Married', 'Divorced','Therapist',
                'Other.1', 'Administrative', 'Travel_Frequently','YearsWithCurrManager',
                'YearsSinceLastPromotion', 'YearsInCurrentRole', 'YearsAtCompany','Age',
                'TotalWorkingYears', 'Shift', 'Single', 'JobSatisfaction', 'DailyRate', 
                'MonthlyIncome','JobLevel', 'EnvironmentSatisfaction', 'HourlyRate']]
```

## Before we finalize our feature selection we need to ensure we avoid any multi-collinearity of the independent features and so we will  assess correlation of our newly selected "X" features 


```python
# calculate correlation of X_train features
X.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmployeeID</th>
      <th>OverTime</th>
      <th>Married</th>
      <th>Divorced</th>
      <th>Therapist</th>
      <th>Other.1</th>
      <th>Administrative</th>
      <th>Travel_Frequently</th>
      <th>YearsWithCurrManager</th>
      <th>YearsSinceLastPromotion</th>
      <th>...</th>
      <th>Age</th>
      <th>TotalWorkingYears</th>
      <th>Shift</th>
      <th>Single</th>
      <th>JobSatisfaction</th>
      <th>DailyRate</th>
      <th>MonthlyIncome</th>
      <th>JobLevel</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EmployeeID</th>
      <td>1.000000</td>
      <td>-0.027952</td>
      <td>-0.017187</td>
      <td>0.000210</td>
      <td>0.011081</td>
      <td>0.022057</td>
      <td>0.018420</td>
      <td>-0.001679</td>
      <td>-0.013707</td>
      <td>0.031340</td>
      <td>...</td>
      <td>-0.039033</td>
      <td>-0.031294</td>
      <td>-0.003884</td>
      <td>0.018318</td>
      <td>-0.007392</td>
      <td>-0.002558</td>
      <td>-0.027595</td>
      <td>-0.030085</td>
      <td>-0.005930</td>
      <td>0.000007</td>
    </tr>
    <tr>
      <th>OverTime</th>
      <td>-0.027952</td>
      <td>1.000000</td>
      <td>-0.012405</td>
      <td>0.028293</td>
      <td>-0.015384</td>
      <td>0.049238</td>
      <td>-0.019162</td>
      <td>0.030692</td>
      <td>-0.049998</td>
      <td>-0.027633</td>
      <td>...</td>
      <td>0.030970</td>
      <td>0.019880</td>
      <td>0.009751</td>
      <td>-0.012152</td>
      <td>0.017259</td>
      <td>0.013474</td>
      <td>0.011969</td>
      <td>0.006171</td>
      <td>0.076885</td>
      <td>-0.015575</td>
    </tr>
    <tr>
      <th>Married</th>
      <td>-0.017187</td>
      <td>-0.012405</td>
      <td>1.000000</td>
      <td>-0.500837</td>
      <td>0.020346</td>
      <td>-0.034832</td>
      <td>0.036375</td>
      <td>-0.016296</td>
      <td>0.039595</td>
      <td>0.047891</td>
      <td>...</td>
      <td>0.087163</td>
      <td>0.056776</td>
      <td>0.212182</td>
      <td>-0.625263</td>
      <td>-0.006441</td>
      <td>0.033362</td>
      <td>0.058937</td>
      <td>0.055978</td>
      <td>-0.026605</td>
      <td>0.041855</td>
    </tr>
    <tr>
      <th>Divorced</th>
      <td>0.000210</td>
      <td>0.028293</td>
      <td>-0.500837</td>
      <td>1.000000</td>
      <td>0.015750</td>
      <td>-0.003429</td>
      <td>0.012051</td>
      <td>0.000069</td>
      <td>0.000778</td>
      <td>-0.016633</td>
      <td>...</td>
      <td>0.025896</td>
      <td>0.027044</td>
      <td>0.449617</td>
      <td>-0.362325</td>
      <td>-0.016153</td>
      <td>0.049534</td>
      <td>0.028530</td>
      <td>0.030563</td>
      <td>0.022820</td>
      <td>-0.015792</td>
    </tr>
    <tr>
      <th>Therapist</th>
      <td>0.011081</td>
      <td>-0.015384</td>
      <td>0.020346</td>
      <td>0.015750</td>
      <td>1.000000</td>
      <td>-0.243788</td>
      <td>-0.096766</td>
      <td>-0.005212</td>
      <td>0.087145</td>
      <td>0.003570</td>
      <td>...</td>
      <td>0.066814</td>
      <td>0.087889</td>
      <td>0.003639</td>
      <td>-0.036111</td>
      <td>-0.004456</td>
      <td>-0.016433</td>
      <td>0.090090</td>
      <td>0.148062</td>
      <td>0.053129</td>
      <td>-0.017635</td>
    </tr>
    <tr>
      <th>Other.1</th>
      <td>0.022057</td>
      <td>0.049238</td>
      <td>-0.034832</td>
      <td>-0.003429</td>
      <td>-0.243788</td>
      <td>1.000000</td>
      <td>-0.185603</td>
      <td>0.026205</td>
      <td>-0.146215</td>
      <td>-0.059912</td>
      <td>...</td>
      <td>-0.115285</td>
      <td>-0.172149</td>
      <td>-0.033613</td>
      <td>0.040601</td>
      <td>-0.017922</td>
      <td>0.008571</td>
      <td>-0.205051</td>
      <td>-0.268842</td>
      <td>-0.017158</td>
      <td>-0.007548</td>
    </tr>
    <tr>
      <th>Administrative</th>
      <td>0.018420</td>
      <td>-0.019162</td>
      <td>0.036375</td>
      <td>0.012051</td>
      <td>-0.096766</td>
      <td>-0.185603</td>
      <td>1.000000</td>
      <td>-0.065798</td>
      <td>0.168081</td>
      <td>0.116193</td>
      <td>...</td>
      <td>0.254053</td>
      <td>0.424881</td>
      <td>-0.007451</td>
      <td>-0.050036</td>
      <td>0.002253</td>
      <td>-0.029636</td>
      <td>0.581717</td>
      <td>0.509565</td>
      <td>-0.004736</td>
      <td>0.016232</td>
    </tr>
    <tr>
      <th>Travel_Frequently</th>
      <td>-0.001679</td>
      <td>0.030692</td>
      <td>-0.016296</td>
      <td>0.000069</td>
      <td>-0.005212</td>
      <td>0.026205</td>
      <td>-0.065798</td>
      <td>1.000000</td>
      <td>0.004547</td>
      <td>0.022498</td>
      <td>...</td>
      <td>-0.030146</td>
      <td>-0.019856</td>
      <td>-0.010557</td>
      <td>0.017486</td>
      <td>0.040752</td>
      <td>-0.012983</td>
      <td>-0.042703</td>
      <td>-0.033256</td>
      <td>-0.007935</td>
      <td>-0.016717</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>-0.013707</td>
      <td>-0.049998</td>
      <td>0.039595</td>
      <td>0.000778</td>
      <td>0.087145</td>
      <td>-0.146215</td>
      <td>0.168081</td>
      <td>0.004547</td>
      <td>1.000000</td>
      <td>0.518664</td>
      <td>...</td>
      <td>0.215909</td>
      <td>0.461688</td>
      <td>0.014314</td>
      <td>-0.043339</td>
      <td>-0.023042</td>
      <td>-0.025272</td>
      <td>0.350122</td>
      <td>0.379717</td>
      <td>-0.012417</td>
      <td>-0.021436</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>0.031340</td>
      <td>-0.027633</td>
      <td>0.047891</td>
      <td>-0.016633</td>
      <td>0.003570</td>
      <td>-0.059912</td>
      <td>0.116193</td>
      <td>0.022498</td>
      <td>0.518664</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.217212</td>
      <td>0.394941</td>
      <td>0.011702</td>
      <td>-0.036574</td>
      <td>-0.013595</td>
      <td>-0.034571</td>
      <td>0.337241</td>
      <td>0.343102</td>
      <td>0.005866</td>
      <td>-0.028642</td>
    </tr>
    <tr>
      <th>YearsInCurrentRole</th>
      <td>-0.014384</td>
      <td>-0.042361</td>
      <td>0.065051</td>
      <td>0.009901</td>
      <td>0.076948</td>
      <td>-0.133644</td>
      <td>0.162408</td>
      <td>-0.009533</td>
      <td>0.721543</td>
      <td>0.548235</td>
      <td>...</td>
      <td>0.222655</td>
      <td>0.457439</td>
      <td>0.039263</td>
      <td>-0.078977</td>
      <td>-0.001337</td>
      <td>0.019651</td>
      <td>0.364152</td>
      <td>0.387624</td>
      <td>0.012241</td>
      <td>-0.016776</td>
    </tr>
    <tr>
      <th>YearsAtCompany</th>
      <td>-0.018060</td>
      <td>-0.018642</td>
      <td>0.052403</td>
      <td>0.014625</td>
      <td>0.036099</td>
      <td>-0.128711</td>
      <td>0.267159</td>
      <td>0.007048</td>
      <td>0.771224</td>
      <td>0.616915</td>
      <td>...</td>
      <td>0.319012</td>
      <td>0.622495</td>
      <td>0.013028</td>
      <td>-0.069616</td>
      <td>0.005376</td>
      <td>-0.026892</td>
      <td>0.513977</td>
      <td>0.532529</td>
      <td>-0.000448</td>
      <td>-0.014742</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.039033</td>
      <td>0.030970</td>
      <td>0.087163</td>
      <td>0.025896</td>
      <td>0.066814</td>
      <td>-0.115285</td>
      <td>0.254053</td>
      <td>-0.030146</td>
      <td>0.215909</td>
      <td>0.217212</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.692512</td>
      <td>0.037117</td>
      <td>-0.117209</td>
      <td>-0.015848</td>
      <td>0.001441</td>
      <td>0.511378</td>
      <td>0.518333</td>
      <td>0.008945</td>
      <td>0.034671</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>-0.031294</td>
      <td>0.019880</td>
      <td>0.056776</td>
      <td>0.027044</td>
      <td>0.087889</td>
      <td>-0.172149</td>
      <td>0.424881</td>
      <td>-0.019856</td>
      <td>0.461688</td>
      <td>0.394941</td>
      <td>...</td>
      <td>0.692512</td>
      <td>1.000000</td>
      <td>0.009632</td>
      <td>-0.085522</td>
      <td>-0.021435</td>
      <td>0.009378</td>
      <td>0.772938</td>
      <td>0.780929</td>
      <td>0.000208</td>
      <td>0.005988</td>
    </tr>
    <tr>
      <th>Shift</th>
      <td>-0.003884</td>
      <td>0.009751</td>
      <td>0.212182</td>
      <td>0.449617</td>
      <td>0.003639</td>
      <td>-0.033613</td>
      <td>-0.007451</td>
      <td>-0.010557</td>
      <td>0.014314</td>
      <td>0.011702</td>
      <td>...</td>
      <td>0.037117</td>
      <td>0.009632</td>
      <td>1.000000</td>
      <td>-0.633883</td>
      <td>0.017740</td>
      <td>0.054407</td>
      <td>0.005847</td>
      <td>0.010478</td>
      <td>0.005283</td>
      <td>0.051862</td>
    </tr>
    <tr>
      <th>Single</th>
      <td>0.018318</td>
      <td>-0.012152</td>
      <td>-0.625263</td>
      <td>-0.362325</td>
      <td>-0.036111</td>
      <td>0.040601</td>
      <td>-0.050036</td>
      <td>0.017486</td>
      <td>-0.043339</td>
      <td>-0.036574</td>
      <td>...</td>
      <td>-0.117209</td>
      <td>-0.085522</td>
      <td>-0.633883</td>
      <td>1.000000</td>
      <td>0.021501</td>
      <td>-0.080588</td>
      <td>-0.089190</td>
      <td>-0.087837</td>
      <td>0.008074</td>
      <td>-0.030832</td>
    </tr>
    <tr>
      <th>JobSatisfaction</th>
      <td>-0.007392</td>
      <td>0.017259</td>
      <td>-0.006441</td>
      <td>-0.016153</td>
      <td>-0.004456</td>
      <td>-0.017922</td>
      <td>0.002253</td>
      <td>0.040752</td>
      <td>-0.023042</td>
      <td>-0.013595</td>
      <td>...</td>
      <td>-0.015848</td>
      <td>-0.021435</td>
      <td>0.017740</td>
      <td>0.021501</td>
      <td>1.000000</td>
      <td>0.032115</td>
      <td>-0.016550</td>
      <td>-0.012497</td>
      <td>0.001518</td>
      <td>-0.073942</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>-0.002558</td>
      <td>0.013474</td>
      <td>0.033362</td>
      <td>0.049534</td>
      <td>-0.016433</td>
      <td>0.008571</td>
      <td>-0.029636</td>
      <td>-0.012983</td>
      <td>-0.025272</td>
      <td>-0.034571</td>
      <td>...</td>
      <td>0.001441</td>
      <td>0.009378</td>
      <td>0.054407</td>
      <td>-0.080588</td>
      <td>0.032115</td>
      <td>1.000000</td>
      <td>0.011030</td>
      <td>0.009005</td>
      <td>0.010620</td>
      <td>0.027128</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>-0.027595</td>
      <td>0.011969</td>
      <td>0.058937</td>
      <td>0.028530</td>
      <td>0.090090</td>
      <td>-0.205051</td>
      <td>0.581717</td>
      <td>-0.042703</td>
      <td>0.350122</td>
      <td>0.337241</td>
      <td>...</td>
      <td>0.511378</td>
      <td>0.772938</td>
      <td>0.005847</td>
      <td>-0.089190</td>
      <td>-0.016550</td>
      <td>0.011030</td>
      <td>1.000000</td>
      <td>0.951572</td>
      <td>0.003372</td>
      <td>-0.008443</td>
    </tr>
    <tr>
      <th>JobLevel</th>
      <td>-0.030085</td>
      <td>0.006171</td>
      <td>0.055978</td>
      <td>0.030563</td>
      <td>0.148062</td>
      <td>-0.268842</td>
      <td>0.509565</td>
      <td>-0.033256</td>
      <td>0.379717</td>
      <td>0.343102</td>
      <td>...</td>
      <td>0.518333</td>
      <td>0.780929</td>
      <td>0.010478</td>
      <td>-0.087837</td>
      <td>-0.012497</td>
      <td>0.009005</td>
      <td>0.951572</td>
      <td>1.000000</td>
      <td>0.008277</td>
      <td>-0.018830</td>
    </tr>
    <tr>
      <th>EnvironmentSatisfaction</th>
      <td>-0.005930</td>
      <td>0.076885</td>
      <td>-0.026605</td>
      <td>0.022820</td>
      <td>0.053129</td>
      <td>-0.017158</td>
      <td>-0.004736</td>
      <td>-0.007935</td>
      <td>-0.012417</td>
      <td>0.005866</td>
      <td>...</td>
      <td>0.008945</td>
      <td>0.000208</td>
      <td>0.005283</td>
      <td>0.008074</td>
      <td>0.001518</td>
      <td>0.010620</td>
      <td>0.003372</td>
      <td>0.008277</td>
      <td>1.000000</td>
      <td>-0.057505</td>
    </tr>
    <tr>
      <th>HourlyRate</th>
      <td>0.000007</td>
      <td>-0.015575</td>
      <td>0.041855</td>
      <td>-0.015792</td>
      <td>-0.017635</td>
      <td>-0.007548</td>
      <td>0.016232</td>
      <td>-0.016717</td>
      <td>-0.021436</td>
      <td>-0.028642</td>
      <td>...</td>
      <td>0.034671</td>
      <td>0.005988</td>
      <td>0.051862</td>
      <td>-0.030832</td>
      <td>-0.073942</td>
      <td>0.027128</td>
      <td>-0.008443</td>
      <td>-0.018830</td>
      <td>-0.057505</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>22 rows × 22 columns</p>
</div>




```python
# plot correlation heatmap for better visualization

sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(X.corr().round(2), vmin=-1, vmax=1, cmap='coolwarm', annot=True, linewidths=0.5)
plt.show()
```


    
![png](output_26_0.png)
    


### We will use a threshold of ** -0.6 and +0.6** as a threshold to determine strong correlation, and any value below -0.6 will be determined as a strong correlation and any value above +0.6 as a strong correlation.

#### From the observation made from the heatmap, we will drop the following features to avoid multi-collinearity:

- YearsInCurrentRole
- YearsAtCompany
- YearsSinceLastPromotion
- YearsWithCurrManager
- MonthlyIncome
- TotalWorkingYears
- EmployeeID **(this feature does not tell us much about the employee characteristics, other than identifying them)**



```python
# drop the identified collinear features and define the newly update X variable
X = X.drop(['YearsInCurrentRole', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
           'MonthlyIncome', 'TotalWorkingYears', 'EmployeeID'], axis=1)
```


```python
# plot correlation heatmap to verify if the remaining features are not correlated

sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(X.corr().round(2), vmin=-1, vmax=1, cmap='coolwarm', annot=True, linewidths=0.5)
plt.show()
```


    
![png](output_29_0.png)
    


After our second iteration of checking for correlation, it turned out that the **Shift and Single** features are correlated and so we will go ahead and drop them from the selected features


```python
# drop the identified correlated features
X = X.drop(['Shift', 'Single'], axis=1)
```


```python
# plot correlation heatmap to verify if the remaining features are not strongly correlated

sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(X.corr().round(2), vmin=-1, vmax=1, cmap='coolwarm', annot=True, linewidths=0.5)
plt.show()
```


    
![png](output_32_0.png)
    


From the correlation heatmap above we can observe that the remaining features are do not have a correlation value **above +0.6 or below -0.6** With that we can go ahead and make the remaining features our official **X features**

## Now that the best predictive features have been selected, we will go ahead and split our data into train and test sets


```python
# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.3,
    stratify = y,
    random_state = 1
)
```


```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OverTime</th>
      <th>Married</th>
      <th>Divorced</th>
      <th>Therapist</th>
      <th>Other.1</th>
      <th>Administrative</th>
      <th>Travel_Frequently</th>
      <th>Age</th>
      <th>JobSatisfaction</th>
      <th>DailyRate</th>
      <th>JobLevel</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>334</td>
      <td>1</td>
      <td>1</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>2</td>
      <td>790</td>
      <td>2</td>
      <td>1</td>
      <td>40</td>
    </tr>
    <tr>
      <th>246</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>4</td>
      <td>832</td>
      <td>1</td>
      <td>3</td>
      <td>63</td>
    </tr>
    <tr>
      <th>768</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>54</td>
      <td>3</td>
      <td>1082</td>
      <td>3</td>
      <td>3</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1214</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>3</td>
      <td>581</td>
      <td>1</td>
      <td>3</td>
      <td>62</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>138</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>3</td>
      <td>959</td>
      <td>2</td>
      <td>1</td>
      <td>41</td>
    </tr>
    <tr>
      <th>570</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29</td>
      <td>3</td>
      <td>657</td>
      <td>2</td>
      <td>2</td>
      <td>66</td>
    </tr>
    <tr>
      <th>328</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>33</td>
      <td>4</td>
      <td>508</td>
      <td>2</td>
      <td>2</td>
      <td>46</td>
    </tr>
    <tr>
      <th>56</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>853</td>
      <td>3</td>
      <td>2</td>
      <td>71</td>
    </tr>
    <tr>
      <th>621</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>4</td>
      <td>1012</td>
      <td>1</td>
      <td>2</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
<p>1173 rows × 13 columns</p>
</div>



We have initiated an 70% train set and 30% test set split on our data 

- the split percentages are at our own discretion, the goal is to have sufficient enough data to train our model

#### Class imbalance is a common problem, that cause bias of our model accuracy and so It is important to check whether our dependent feature classes are balanced or not.


```python
# check the distribution of target variable (attrition) classes

print('Attrition class distribution')
display(pd.Series(y_train).value_counts().sort_index())
```

    Attrition class distribution



    0    1034
    1     139
    Name: Attrition, dtype: int64



```python
print (f'There is a significant class imbalance of {(139/1034)*100}% class 1, compared to {(1034/(1034+139))*100}% class 0.')
```

    There is a significant class imbalance of 13.44294003868472% class 1, compared to 88.15004262574595% class 0.


We will need to upsample the data in the y_train dataset in order to balance the classes. 

**We will simply duplicate the data in the monirity class (class 1), this will not add any new information to the model, to achieve this we will augment the data using "SMOTE"** (Synthetic Minority Oversampling Technique)


```python
# instantiate SMOTE sampler, fit it to the training data, then resample the data
X_train_sm, y_train_sm = SMOTE(random_state=1).fit_resample(X_train, y_train)
```


```python
# sanity check to see what SMOTE has done

print('Original class distribution')
display(pd.Series(y_train).value_counts().sort_index())

print('\nResampled class distribution')
display(pd.Series(y_train_sm).value_counts().sort_index())
```

    Original class distribution



    0    1034
    1     139
    Name: Attrition, dtype: int64


    
    Resampled class distribution



    0    1034
    1    1034
    Name: Attrition, dtype: int64


#### Our minority class, "class 1" has been upsampled from 139 to 1034. Now our classes are balanced.

## We will now go on to fit our data into a Logistic Regression Model using the "Logit function" of the statsmodel. This will yield coefficients  that help us determine the odds ratios and probabilities of each feature predictions.

Before we fit the model, we must alter our `X` data to add in a column of ones as the first column. This is done using the `add_constant` function. We need to do this so that the matrix calculations in the background allow for an intercept to be calculated. **Always** add a constant.


```python
# import stats model package 
import statsmodels.api as sm
```


```python
# Add Constants
X_withconstant = sm.add_constant(X_train_sm)

#check if constant has been successfully added
X_withconstant
```

    /Users/tawandanigelchitapi/opt/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/tsatools.py:142: FutureWarning: In a future version of pandas all arguments of concat except for the argument 'objs' will be keyword-only
      x = pd.concat(x[::order], 1)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>OverTime</th>
      <th>Married</th>
      <th>Divorced</th>
      <th>Therapist</th>
      <th>Other.1</th>
      <th>Administrative</th>
      <th>Travel_Frequently</th>
      <th>Age</th>
      <th>JobSatisfaction</th>
      <th>DailyRate</th>
      <th>JobLevel</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>2</td>
      <td>334</td>
      <td>1</td>
      <td>1</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>2</td>
      <td>790</td>
      <td>2</td>
      <td>1</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>4</td>
      <td>832</td>
      <td>1</td>
      <td>3</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>54</td>
      <td>3</td>
      <td>1082</td>
      <td>3</td>
      <td>3</td>
      <td>41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>3</td>
      <td>581</td>
      <td>1</td>
      <td>3</td>
      <td>62</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2063</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>1</td>
      <td>1094</td>
      <td>1</td>
      <td>3</td>
      <td>95</td>
    </tr>
    <tr>
      <th>2064</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>4</td>
      <td>844</td>
      <td>1</td>
      <td>3</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2065</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>4</td>
      <td>478</td>
      <td>1</td>
      <td>2</td>
      <td>59</td>
    </tr>
    <tr>
      <th>2066</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46</td>
      <td>1</td>
      <td>1165</td>
      <td>2</td>
      <td>2</td>
      <td>53</td>
    </tr>
    <tr>
      <th>2067</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>2</td>
      <td>308</td>
      <td>2</td>
      <td>3</td>
      <td>55</td>
    </tr>
  </tbody>
</table>
<p>2068 rows × 14 columns</p>
</div>



## Now fit the training data into the Logit function 


```python
# 1. Instantiate model
mylogreg = sm.Logit(y_train_sm, X_withconstant)

#2. Fit the model (this returns a separate object with the parameters)
mylogreg_results = mylogreg.fit()

#3. Display results
mylogreg_results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.301788
             Iterations 9





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Attrition</td>    <th>  No. Observations:  </th>  <td>  2068</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  2054</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>    13</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 27 Oct 2022</td> <th>  Pseudo R-squ.:     </th>  <td>0.5646</td> 
</tr>
<tr>
  <th>Time:</th>                <td>09:06:57</td>     <th>  Log-Likelihood:    </th> <td> -624.10</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -1433.4</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> 
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                   <td>    9.0001</td> <td>    0.553</td> <td>   16.289</td> <td> 0.000</td> <td>    7.917</td> <td>   10.083</td>
</tr>
<tr>
  <th>OverTime</th>                <td>    2.5144</td> <td>    0.175</td> <td>   14.344</td> <td> 0.000</td> <td>    2.171</td> <td>    2.858</td>
</tr>
<tr>
  <th>Married</th>                 <td>   -2.2115</td> <td>    0.166</td> <td>  -13.299</td> <td> 0.000</td> <td>   -2.537</td> <td>   -1.886</td>
</tr>
<tr>
  <th>Divorced</th>                <td>   -3.3579</td> <td>    0.281</td> <td>  -11.932</td> <td> 0.000</td> <td>   -3.909</td> <td>   -2.806</td>
</tr>
<tr>
  <th>Therapist</th>               <td>   -5.0926</td> <td>    1.111</td> <td>   -4.585</td> <td> 0.000</td> <td>   -7.270</td> <td>   -2.916</td>
</tr>
<tr>
  <th>Other.1</th>                 <td>   -0.8803</td> <td>    0.160</td> <td>   -5.517</td> <td> 0.000</td> <td>   -1.193</td> <td>   -0.568</td>
</tr>
<tr>
  <th>Administrative</th>          <td>   -1.8056</td> <td>    1.119</td> <td>   -1.613</td> <td> 0.107</td> <td>   -3.999</td> <td>    0.388</td>
</tr>
<tr>
  <th>Travel_Frequently</th>       <td>   -0.2098</td> <td>    0.203</td> <td>   -1.033</td> <td> 0.301</td> <td>   -0.608</td> <td>    0.188</td>
</tr>
<tr>
  <th>Age</th>                     <td>   -0.0712</td> <td>    0.010</td> <td>   -7.085</td> <td> 0.000</td> <td>   -0.091</td> <td>   -0.052</td>
</tr>
<tr>
  <th>JobSatisfaction</th>         <td>   -0.7321</td> <td>    0.072</td> <td>  -10.136</td> <td> 0.000</td> <td>   -0.874</td> <td>   -0.591</td>
</tr>
<tr>
  <th>DailyRate</th>               <td>   -0.0006</td> <td>    0.000</td> <td>   -2.949</td> <td> 0.003</td> <td>   -0.001</td> <td>   -0.000</td>
</tr>
<tr>
  <th>JobLevel</th>                <td>   -0.9115</td> <td>    0.118</td> <td>   -7.722</td> <td> 0.000</td> <td>   -1.143</td> <td>   -0.680</td>
</tr>
<tr>
  <th>EnvironmentSatisfaction</th> <td>   -0.7753</td> <td>    0.076</td> <td>  -10.193</td> <td> 0.000</td> <td>   -0.924</td> <td>   -0.626</td>
</tr>
<tr>
  <th>HourlyRate</th>              <td>   -0.0068</td> <td>    0.004</td> <td>   -1.790</td> <td> 0.073</td> <td>   -0.014</td> <td>    0.001</td>
</tr>
</table>



- The coefficients calculated refer to our logistic regression model, they control how the sigmoid curve looks (more stretched out horizontally, inverted, etc.).

- To understand the values of the parameters, we need to understand a key underlying concept first

## Odds ratio


We first need to understand the concept of the odds ratio. When we talk about the odds ratio, we refer to the ratio between the odds of an outcome happening, and the same outcome not happening.
    
We can calculate the ratio using the following formula:

$$\frac{p}{1-p}$$

where $p$ is the chance of the outcome happening (**leaving the job**) and $1-p$ is the probability of my outcome not happening (**not leaving the job**).
    
For example, if from the employee data there is a 75% probability of an employee leaving the job. This means the chances of staying on the job are 25%. The odds ratio of an employee leaving their job is

$$\frac{0.75}{0.25} = \frac{3}{1}$$

so my odds ratio is $\frac{3}{1}$. You may also see this written as 3:1 odds of leaving the job.


#### Bringing this all together

With the interpretation of logistic regression coefficients. The important thing to remember is that we don't interpret the coefficients in terms of their impact on the **probability** of getting class 1 (**attrition**) but rather the **odds ratio** of getting class 1.

#### Lets go through and interpret the coefficients of our logistic regression model. 


```python
mylogreg_results.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Attrition</td>    <th>  No. Observations:  </th>  <td>  2068</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  2054</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>    13</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 27 Oct 2022</td> <th>  Pseudo R-squ.:     </th>  <td>0.5646</td> 
</tr>
<tr>
  <th>Time:</th>                <td>09:06:57</td>     <th>  Log-Likelihood:    </th> <td> -624.10</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -1433.4</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> 
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                   <td>    9.0001</td> <td>    0.553</td> <td>   16.289</td> <td> 0.000</td> <td>    7.917</td> <td>   10.083</td>
</tr>
<tr>
  <th>OverTime</th>                <td>    2.5144</td> <td>    0.175</td> <td>   14.344</td> <td> 0.000</td> <td>    2.171</td> <td>    2.858</td>
</tr>
<tr>
  <th>Married</th>                 <td>   -2.2115</td> <td>    0.166</td> <td>  -13.299</td> <td> 0.000</td> <td>   -2.537</td> <td>   -1.886</td>
</tr>
<tr>
  <th>Divorced</th>                <td>   -3.3579</td> <td>    0.281</td> <td>  -11.932</td> <td> 0.000</td> <td>   -3.909</td> <td>   -2.806</td>
</tr>
<tr>
  <th>Therapist</th>               <td>   -5.0926</td> <td>    1.111</td> <td>   -4.585</td> <td> 0.000</td> <td>   -7.270</td> <td>   -2.916</td>
</tr>
<tr>
  <th>Other.1</th>                 <td>   -0.8803</td> <td>    0.160</td> <td>   -5.517</td> <td> 0.000</td> <td>   -1.193</td> <td>   -0.568</td>
</tr>
<tr>
  <th>Administrative</th>          <td>   -1.8056</td> <td>    1.119</td> <td>   -1.613</td> <td> 0.107</td> <td>   -3.999</td> <td>    0.388</td>
</tr>
<tr>
  <th>Travel_Frequently</th>       <td>   -0.2098</td> <td>    0.203</td> <td>   -1.033</td> <td> 0.301</td> <td>   -0.608</td> <td>    0.188</td>
</tr>
<tr>
  <th>Age</th>                     <td>   -0.0712</td> <td>    0.010</td> <td>   -7.085</td> <td> 0.000</td> <td>   -0.091</td> <td>   -0.052</td>
</tr>
<tr>
  <th>JobSatisfaction</th>         <td>   -0.7321</td> <td>    0.072</td> <td>  -10.136</td> <td> 0.000</td> <td>   -0.874</td> <td>   -0.591</td>
</tr>
<tr>
  <th>DailyRate</th>               <td>   -0.0006</td> <td>    0.000</td> <td>   -2.949</td> <td> 0.003</td> <td>   -0.001</td> <td>   -0.000</td>
</tr>
<tr>
  <th>JobLevel</th>                <td>   -0.9115</td> <td>    0.118</td> <td>   -7.722</td> <td> 0.000</td> <td>   -1.143</td> <td>   -0.680</td>
</tr>
<tr>
  <th>EnvironmentSatisfaction</th> <td>   -0.7753</td> <td>    0.076</td> <td>  -10.193</td> <td> 0.000</td> <td>   -0.924</td> <td>   -0.626</td>
</tr>
<tr>
  <th>HourlyRate</th>              <td>   -0.0068</td> <td>    0.004</td> <td>   -1.790</td> <td> 0.073</td> <td>   -0.014</td> <td>    0.001</td>
</tr>
</table>



### Model Interpretation 

We care about the model parameters and their associated p-values. 

If the p-values are low (lower than 0.05) than it means that the associated features are relevant in predicting the dependent variable, **Employee Attrition**, in this case.

Based on the p-values in the summary table above, we can see that the **Travel_Frequently, Administrative and HourlyRate** features are not relavent in predicting employee attrition since they have p-values above 0.05 

We will go ahead and drop them immediately.


```python
X_train_sm2 = X_train_sm.drop(['Travel_Frequently', 'Administrative', 'HourlyRate'], axis=1)
```


```python
# Add constants to new set of X_train features "X_train_sm2"
X_withconstant2 = sm.add_constant(X_train_sm2)
X_withconstant2
```

    /Users/tawandanigelchitapi/opt/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/tsatools.py:142: FutureWarning: In a future version of pandas all arguments of concat except for the argument 'objs' will be keyword-only
      x = pd.concat(x[::order], 1)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>OverTime</th>
      <th>Married</th>
      <th>Divorced</th>
      <th>Therapist</th>
      <th>Other.1</th>
      <th>Age</th>
      <th>JobSatisfaction</th>
      <th>DailyRate</th>
      <th>JobLevel</th>
      <th>EnvironmentSatisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>32</td>
      <td>2</td>
      <td>334</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>2</td>
      <td>790</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>33</td>
      <td>4</td>
      <td>832</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>54</td>
      <td>3</td>
      <td>1082</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>3</td>
      <td>581</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2063</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>21</td>
      <td>1</td>
      <td>1094</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2064</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>4</td>
      <td>844</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2065</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>4</td>
      <td>478</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2066</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46</td>
      <td>1</td>
      <td>1165</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2067</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>2</td>
      <td>308</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>2068 rows × 11 columns</p>
</div>




```python
# 1. Instantiate model
mylogreg2 = sm.Logit(y_train_sm, X_withconstant2)

#2. Fit the model (this returns a separate object with the parameters)
mylogreg2_results = mylogreg2.fit()

#3. Display the results
mylogreg2_results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.303802
             Iterations 9





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Attrition</td>    <th>  No. Observations:  </th>  <td>  2068</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  2057</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>    10</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 27 Oct 2022</td> <th>  Pseudo R-squ.:     </th>  <td>0.5617</td> 
</tr>
<tr>
  <th>Time:</th>                <td>09:06:57</td>     <th>  Log-Likelihood:    </th> <td> -628.26</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -1433.4</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> 
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                   <td>    8.6994</td> <td>    0.500</td> <td>   17.413</td> <td> 0.000</td> <td>    7.720</td> <td>    9.679</td>
</tr>
<tr>
  <th>OverTime</th>                <td>    2.5249</td> <td>    0.175</td> <td>   14.458</td> <td> 0.000</td> <td>    2.183</td> <td>    2.867</td>
</tr>
<tr>
  <th>Married</th>                 <td>   -2.2306</td> <td>    0.166</td> <td>  -13.467</td> <td> 0.000</td> <td>   -2.555</td> <td>   -1.906</td>
</tr>
<tr>
  <th>Divorced</th>                <td>   -3.3570</td> <td>    0.279</td> <td>  -12.026</td> <td> 0.000</td> <td>   -3.904</td> <td>   -2.810</td>
</tr>
<tr>
  <th>Therapist</th>               <td>   -5.1153</td> <td>    1.132</td> <td>   -4.521</td> <td> 0.000</td> <td>   -7.333</td> <td>   -2.898</td>
</tr>
<tr>
  <th>Other.1</th>                 <td>   -0.9074</td> <td>    0.159</td> <td>   -5.695</td> <td> 0.000</td> <td>   -1.220</td> <td>   -0.595</td>
</tr>
<tr>
  <th>Age</th>                     <td>   -0.0745</td> <td>    0.010</td> <td>   -7.521</td> <td> 0.000</td> <td>   -0.094</td> <td>   -0.055</td>
</tr>
<tr>
  <th>JobSatisfaction</th>         <td>   -0.7311</td> <td>    0.072</td> <td>  -10.216</td> <td> 0.000</td> <td>   -0.871</td> <td>   -0.591</td>
</tr>
<tr>
  <th>DailyRate</th>               <td>   -0.0006</td> <td>    0.000</td> <td>   -2.938</td> <td> 0.003</td> <td>   -0.001</td> <td>   -0.000</td>
</tr>
<tr>
  <th>JobLevel</th>                <td>   -0.9659</td> <td>    0.112</td> <td>   -8.614</td> <td> 0.000</td> <td>   -1.186</td> <td>   -0.746</td>
</tr>
<tr>
  <th>EnvironmentSatisfaction</th> <td>   -0.7732</td> <td>    0.076</td> <td>  -10.202</td> <td> 0.000</td> <td>   -0.922</td> <td>   -0.625</td>
</tr>
</table>



Now all the remaining features have p-values less than 0.05 and this means that these features are relevant at predicting Attrition

## We can calculate an *odds ratio* by taking the exponential function of the calculated parameter values (coefficients).



```python
odds_ratio = np.exp(mylogreg2_results.params)

coeffs_df = pd.DataFrame({"coef": mylogreg2_results.params, "odds_ratio": odds_ratio})
coeffs_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coef</th>
      <th>odds_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>const</th>
      <td>8.699406</td>
      <td>5999.345395</td>
    </tr>
    <tr>
      <th>OverTime</th>
      <td>2.524882</td>
      <td>12.489425</td>
    </tr>
    <tr>
      <th>Married</th>
      <td>-2.230568</td>
      <td>0.107467</td>
    </tr>
    <tr>
      <th>Divorced</th>
      <td>-3.356998</td>
      <td>0.034840</td>
    </tr>
    <tr>
      <th>Therapist</th>
      <td>-5.115258</td>
      <td>0.006004</td>
    </tr>
    <tr>
      <th>Other.1</th>
      <td>-0.907421</td>
      <td>0.403564</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.074477</td>
      <td>0.928229</td>
    </tr>
    <tr>
      <th>JobSatisfaction</th>
      <td>-0.731060</td>
      <td>0.481398</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>-0.000552</td>
      <td>0.999448</td>
    </tr>
    <tr>
      <th>JobLevel</th>
      <td>-0.965948</td>
      <td>0.380622</td>
    </tr>
    <tr>
      <th>EnvironmentSatisfaction</th>
      <td>-0.773159</td>
      <td>0.461553</td>
    </tr>
  </tbody>
</table>
</div>



## Now we will calculate the accruacy of our classification model, by comparing the amount of correct predictions against the total number of the points in the target feature 'Attrition'

We will set a threshold of 50% and so if the probability is greater than 0.5 the model will make a hard prediction of **'1' (Attrition)**, if the probability is less than 0.5 the model will make a hard prediction of **'0' (Stay on the job)**


```python
# remember we fit our model on X_withconstant
model_predictions_prob = mylogreg2_results.predict(X_withconstant2)
# getting the binary predictions
model_predictions_binary = np.where(model_predictions_prob>0.5,1,0)
```


```python
# comparing true and predicted 
(model_predictions_binary == y_train_sm).sum()
```




    1823



The model correctly predicted Attrition or class "1" **1823 times** 


```python
# How many total data points to we have in the Attrition column?
len(y_train_sm)
```




    2068



There are a total of 2068 class data points: **(class 1 and class 0)**


```python
print("The classification train accuracy is:", (1823/2068)*100,"%",  "this is very high and regarded as good" )
```

    The classification train accuracy is: 88.15280464216634 % this is very high and regarded as good


since we removed some features from the training dataset, we need to remove the same features from the testing dataset so that we have a **consistent number and kind of columns in the X_train set and the X_test set**


```python
# Drop the Travel_Frequently, Administrative, HourlyRate columns 
X_test = X_test.drop(['Travel_Frequently', 'Administrative', 'HourlyRate'], axis=1)
X_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OverTime</th>
      <th>Married</th>
      <th>Divorced</th>
      <th>Therapist</th>
      <th>Other.1</th>
      <th>Age</th>
      <th>JobSatisfaction</th>
      <th>DailyRate</th>
      <th>JobLevel</th>
      <th>EnvironmentSatisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>750</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>1</td>
      <td>945</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1044</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>1</td>
      <td>430</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>787</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>4</td>
      <td>654</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>836</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>1</td>
      <td>647</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1514</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>1</td>
      <td>469</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>585</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>52</td>
      <td>3</td>
      <td>1325</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>498</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>32</td>
      <td>4</td>
      <td>646</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>597</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>36</td>
      <td>2</td>
      <td>1041</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>171</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>19</td>
      <td>1</td>
      <td>602</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1225</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>41</td>
      <td>2</td>
      <td>552</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>503 rows × 10 columns</p>
</div>



## We will now compare our training accuracy with the test accuracy and and evaluate our model performance

##### Before we fit our data into any model it is very important that we scale our data since our features do not contain the same degree of values in them, we must scale them so that they are centred at mean = 0 and variance = 1 at the same level.

- we will use a standard scalar to scale our data 


```python
# scale the sampled train data and the unsampled test data
ss_sm = StandardScaler().fit(X_train_sm2)
X_train_sm_ss = ss_sm.transform(X_train_sm2)
X_test_ss = ss_sm.transform(X_test)
```


```python
# Instantiate
employee_logit = LogisticRegression(random_state=1)

# Fit
employee_logit.fit(X_train_sm_ss, y_train_sm)

# Score
print(employee_logit.score(X_train_sm_ss, y_train_sm))
print(employee_logit.score(X_test_ss, y_test))
```

    0.8815280464216635
    0.8449304174950298


The model performed at 88% accuracy on the training data and 84% accuracy on the test data. The model's training accuracy is 4 percentage points higher than the test accuracy. These scores are great, the delta between the two scores is not too wide, however, at this point we can not deductively state that this is a great model.

We need to further granulate the performance matrics of the model and assess how accurate it is at predicting true positive and true negatives.

### Evaluation on Test Data


```python
# class distribution of the y_test column
display(y_test.value_counts())
```


    0    443
    1     60
    Name: Attrition, dtype: int64



```python
# predict classification
y_test_pred = employee_logit.predict(X_test_ss)
```


```python
#Generate the (raw) confusion matrix:

cf_test = confusion_matrix(y_test, y_test_pred)
cf_test
```




    array([[383,  60],
           [ 18,  42]])




```python
# confusion matrix as a dataframe
conmat = pd.DataFrame(
    data = cf_test,
    index = ['true 0', 'true 1'],
    columns = ['predicted 0', 'predicted 1']
)
display(conmat)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted 0</th>
      <th>predicted 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>true 0</th>
      <td>383</td>
      <td>60</td>
    </tr>
    <tr>
      <th>true 1</th>
      <td>18</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>



```python
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

# the function expects the estimator, inputs and target as parameters
plot_confusion_matrix(employee_logit, X_test_ss, y_test, cmap='Reds');
```


    
![png](output_83_0.png)
    


The model predicted:

**383 True Negatives** meaning it correctly predicted **class 0, 382 times**

**42 True Positives** meaning it correctly predicted **class 1, 42 times**

**60 False Positives** meaning it **falsely predicted** class 1, 60 times

**18 False Negatives** meaning it **falsely predicted** class 0, 18 times

Again the raw numbers alone do not tell us a comprehensive story behind the performance of our model and so we will generate a classification report that will provide more information about, **model precision, recall, f1-score and accuracy.**


```python
# Classification report
from sklearn.metrics import classification_report

cf_test_report = classification_report(y_test, y_test_pred)
print(cf_test_report)
```

                  precision    recall  f1-score   support
    
               0       0.96      0.86      0.91       443
               1       0.41      0.70      0.52        60
    
        accuracy                           0.84       503
       macro avg       0.68      0.78      0.71       503
    weighted avg       0.89      0.84      0.86       503
    


## Precision, Recall & F1-score are perfomance metrics 

- Precision (also called 'positive predictive value') measures what proportion of a model assigned to positive are actually members of the positive class. It is a measure of how focused our model is. With a high precision we can be confident points our model identifies as belonging to a class do actually belong to a class. 

- Recall (also known as 'sensitivity') measures how many members of the positive class the model correctly identified out of the total positives. With a high recall we can be confident that our model is not missing many class members.

- f1-Score is the harmonic mean of precision and recall. We use the F1 score in order to try and maximize the precision and recall scores simultaneously. Models with a higher F1 score are usually better at predicting the positive class.







## Summarizing the report above, we will focus on 'class 1' which is Attrition:

- The model has a 41% precision rate of predicting attrition. This is a low precision rate, it means our model is not great at correctly predicting attrition.

- The model has a 70% recall rate. This is a relatively high recall rate and it means that our model is great at detecting attrition, however, there will likely be a high rate of false positives

Given our use case, a high recall rate is most preferred because it detects possible attrition, even though the employee may not neccessarily attrition. A high recall rate enables management to be a sensitive as possible and highly engaged with their employees which in turn increases their chances of avoiding attrition before it occurs.

-  The model has an f1-score 52% and accuracy rate of 84% which is relatively good. 

#### In efforts to further enchance our model performance, we will attempt to optimize our model hyperparameters and evaluate the performance results. In this case, we will be optimizing the Logistic Regression C-value.

- C-value is a hyperparameter

- A high value of C tells the model to give more weight to the training data. A lower value of C will indicate the model to give complexity more weight at the cost of fitting the data. Thus, a high C-value indicates that training data is more important and reflects the real world data



### Hyperparameter Optimization

We'll take our train data and split it to create a validation dataset from it. 
* Train: 70%
* Test Set: 30%

Our training data set is the upsampled dataset, that we applied SMOTE on, which is **"X_train_sm2"**


```python
# Splitting the remainder in two chunks
X_train_new, X_validation, y_train_new, y_validation = \
    train_test_split(X_train_sm2, y_train_sm, test_size = 0.3,
                     random_state=1)
```


```python
print(f'Shape of test set: {X_test.shape}')
print(f'Shape of validation set: {X_validation.shape}')
print(f'Shape of train set: {X_train_new.shape}')
```

    Shape of test set: (503, 10)
    Shape of validation set: (621, 10)
    Shape of train set: (1447, 10)


- The test set has 503 rows and 10 columns 

- The validation set has 621 rows and 10 columns

- The train set has 1447 rows and 10 columns


```python
# scale the train data and transform it and the validation data
ss_sm2 = StandardScaler().fit(X_train_new)
X_train_new_ss = ss_sm2.transform(X_train_new)
X_validation_ss = ss_sm2.transform(X_validation)

```


```python
validation_scores = []
train_scores = []

C_range = np.array([.00000001,.0000001,.000001,.00001,.0001,.001,0.1,\
                1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000])

for c in C_range:
    my_logreg = LogisticRegression(C = c,random_state=1)
    my_logreg.fit(X_train_new_ss,y_train_new)
    
    # train on traning set
    train_scores.append(my_logreg.score(X_train_new_ss,y_train_new))
    # score on validation set
    validation_scores.append(my_logreg.score(X_validation_ss,y_validation))
    
```


```python
plt.figure()
plt.plot(C_range, train_scores,label="Train Score",marker='.')
plt.plot(C_range, validation_scores,label="Validation Scores",marker='.')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend()
plt.show();
```


    
![png](output_96_0.png)
    


From the plot above we can observe that the preferred  optimized c-value is [10^(-1)] or 0.1. At this point our model is not overfitting. Both the train score and validation score sit at about **88% accuracy.** These are significantly high accuracy scores, they  good enough for us to accept and trust our model performance.

However, we will now re-model with the preferred c-value and compare the test accuracy scores with the test score of the non-optimized c-value. There after we will re-evaluate the model performance and assess results.


```python
# now we will run our model with the newly identified c-value
my_optimized_employee_model = LogisticRegression(C=0.1,random_state=1)

# Remember that X_train_sm and y_train_sm held the data I split into train_new and validation
# I can use that data to re-train my model
my_optimized_employee_model.fit(X_train_sm_ss,y_train_sm)
print (my_optimized_employee_model.score(X_validation_ss,y_validation))
print (my_optimized_employee_model.score(X_test_ss,y_test))
```

    0.8824476650563607
    0.8409542743538767


Validation score is **88% accuracy**  and test score is **84% accuracy** after optimizing our c-value, the test accuracy has not changed from the original, this c-value optimization did not improve our model in any way.The accuracy score actually dropped slightly from **0.8449304174950298 to 0.8409542743538767**

#### We will attempt one more method to optimize hyperparameters, we will employ a pipeline grid search and which will yeaild the best hyperparameter values to use in our model. 

### Now we wil employ a pipeline GridSearch to determine the best estimators and prevailing parameters. 


```python
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
```


```python
# estimators
# note that all the planned steps must be included, but the second elements in each tuple are more like placeholders
estimators = [
    ('scaling', StandardScaler()),
    ('reduce_dim', PCA()),
    ('model', LogisticRegression())
]

# instantiate pipeline with the specified steps
pipe = Pipeline(estimators)

# define parameter grid
param_grid = [
    
    # logistic regression with L1 regularization
    {
        'scaling': [MinMaxScaler(), StandardScaler()],
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'model': [LogisticRegression()],
        'model__penalty': ['l1'],
        'model__solver': ['liblinear'],
        'model__C': np.logspace(-5, 5, 11),
        'model__random_state': [1]
    },
    
    # logistic regression with L2 penalty
    {
        'scaling': [MinMaxScaler(), StandardScaler()],
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': [1, 2, 3, 4, 5,6, 7, 8, 9, 10],
        'model': [LogisticRegression()],
        'model__penalty': ['l2'],
        'model__solver': ['lbfgs'],
        'model__C': np.logspace(-5, 5, 11),
        'model__random_state': [1]
    }
]

# instantiate cross-validated grid search object with the steps and parameter grid
grid = GridSearchCV(
    estimator = pipe,
    param_grid = param_grid,
    cv = 5,
    verbose = 5,
    n_jobs = -1
)

# fit the grid to the training data
grid.fit(X_train_sm2, y_train_sm);
```

    Fitting 5 folds for each of 440 candidates, totalling 2200 fits
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.838 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.841 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.855 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.882 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.864 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.824 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.864 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.847 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.826 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.841 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.835 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.862 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.843 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.831 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.874 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.800 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.674 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.712 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.725 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.652 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.686 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.656 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.841 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.795 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.794 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.693 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.657 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.652 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.705 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.693 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.671 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.795 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.686 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.656 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.754 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.766 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.777 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.794 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.713 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.664 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.686 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.663 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.651 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.766 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.684 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.754 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.766 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.777 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.725 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.652 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.838 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.841 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.864 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.831 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.882 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.898 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.814 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.816 total time=   0.1s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scal[CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_stat[CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_ste=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=Minate=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.801 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.841 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.709 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.778 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.821 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.821 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_coMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.725 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.652 total ting=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.725 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.652 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.686 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.656 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.841 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.780 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.650 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.688 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 3/5] END model=Logistic[CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_s[CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_stime=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.686 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.656 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.841 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model[CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model_mponents=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.709 total time=   0.0s
    [CV 5/5] END model=LogisticRegressioRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, redtate=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__uce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.1s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.903 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.886 total time=   0.1s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, redun(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.690 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.778 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.674 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.712 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=[CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxS1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    caler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.847 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.826 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.862 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.847 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.862 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.862 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__ate=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticReg__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, modpenalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.864 total time=   0.0s
    el__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860_C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [Cression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=libliC=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.838 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.893 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.896 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_componence_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s
    near, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.841 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.847 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.838 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.879 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    ts=9, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.882 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__pen total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    V 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.500 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.501 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.499 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.500 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.841 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.838 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=[CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.674 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.712 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__alty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim_[CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__ran_n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.674 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.712 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.88[CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.1s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.903 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.886 total time=   0.1s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=Min   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__pe9 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.903 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.886 total time=   0.1s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dinalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.901 total time=   0.1s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.889 total time=   0.1s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_com__n_components=8, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.792 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.690 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.857 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.862 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.860 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.792 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.862 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.865 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.697 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.780 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.821 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.818 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.893 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.800 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.841 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.903 total time=   0.1s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.886 total time=   0.1s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.547 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.754 total dom_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.833 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=Logisticmponents=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.898 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.1s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.898 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScMaxScaler();, score=0.725 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.652 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.686 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.608 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.680 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.867 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.848 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.855 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.879 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.790 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.792 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.867 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.864 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.792 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.794 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.705 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.640 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.693 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.676 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.656 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.766 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.855 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.882 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.872 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.802 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.800 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.882 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.766 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.777 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.794 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.705 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.635 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.690 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.680 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.855 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.790 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.816 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.874 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.790 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.872 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.847 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.845 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.826 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.867 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.855 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.882 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.872 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.898 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.693 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    Regression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.818 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.661 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.809 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.771 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.789 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.780 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.679 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.628 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.736 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.688 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.678 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.654 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.766 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.650 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.686 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.656 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.766 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.841 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.780 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.862 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.845 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.838 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.848 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.800 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.882 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.872 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.800 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    aler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.792 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.690 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.678 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.816 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.821 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.860 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.862 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.857 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.792 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.792 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [C[CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.674 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.712 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.1s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.855 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.838 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.795 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.847 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.852 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.807 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.835 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.800 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.829 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.867 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.795 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.795 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.829 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.635 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.676 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.709 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.896 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.901 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.898 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.898 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.826 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.831 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.680 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.828 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.816 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.867 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.862 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.833 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.657 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.633 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.679 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.709 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.778 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.829 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.862 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.1s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    V 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.841 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.671 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.626 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.667 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.697 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.899 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.864 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s


After fitting the grid to the training data, use the `best_estimator_` attribute on your grid search object to obtain the most optimal model and its parameters. 


```python
# check the most optimal model
grid.best_estimator_
```




    Pipeline(steps=[('scaling', StandardScaler()),
                    ('reduce_dim', PCA(n_components=8)),
                    ('model',
                     LogisticRegression(C=1000.0, penalty='l1', random_state=1,
                                        solver='liblinear'))])



The best estimator recommended to us requires us to the MinMaxScaler, perform dimension reduction 10 components, using logistric regression with a c-value of 0.1 and a random state of 1.

We will now implement these recommendations and evaluate our model.

### Now using our best estimator we will re-run our model with the specified parameters and evaluate the model


```python
# The best estimator suggests we use the standard scaler 

# make a scaler & fit
ss_scaler = StandardScaler().fit(X_train_sm2)


# transform the data
X_train_sm_sc = ss_scaler.transform(X_train_sm2)
X_test_sc = ss_scaler.transform(X_test)

```


```python
# The best estimator suggest we reduce dimensionality using PCA to 8 components

# Instantiate and fit
my_PCA = PCA(n_components = 8)
my_PCA.fit(X_train_sm_sc)

# Transform train and test
X_train_PCA = my_PCA.transform(X_train_sm_sc)
X_test_PCA = my_PCA.transform(X_test_sc)
```


```python
# Now fit the logistic regression with the best estimator suggested hyper-parameters

# Instantiate the model 
my_final_employee_model = LogisticRegression(C=1000.0,random_state=1,penalty='l1', solver='liblinear' )

# Fit the model 
my_final_employee_model.fit(X_train_PCA,y_train_sm)

# Score the model 
print (my_final_employee_model.score(X_train_PCA,y_train_sm))

print (my_final_employee_model.score(X_test_PCA,y_test))
```

    0.8829787234042553
    0.8409542743538767


After optimizing the hyper-parameters the accuracy did not change significantly, the train accuracy remained at 88% and the test accuracy remained at 84%. This does not neccessarily mean that optimizing hyper paramenters does not woro or help improve our model. Our data is not significantly large itself and so making these adjustments on a small data set may not have a significant impact.

We will go one and evaluate the other performance metric of the model other than the accuracy alone and check for any improvements

### Evaluation on Test Data - After fitting the best estimators and fitting the parameters in the Logistic Regression Model 


```python
# class distribution
display(y_test.value_counts())
```


    0    443
    1     60
    Name: Attrition, dtype: int64



```python
# predict classification
y_test_pred = my_final_employee_model.predict(X_test_PCA)
```

Generate the (raw) confusion matrix:


```python
cf_test = confusion_matrix(y_test, y_test_pred)
cf_test
```




    array([[379,  64],
           [ 16,  44]])




```python
# confusion matrix
conmat = pd.DataFrame(
    data = cf_test,
    index = ['true 0', 'true 1'],
    columns = ['predicted 0', 'predicted 1']
)
display(conmat)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted 0</th>
      <th>predicted 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>true 0</th>
      <td>379</td>
      <td>64</td>
    </tr>
    <tr>
      <th>true 1</th>
      <td>16</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>



```python
# the function expects the estimator, inputs and target as parameters
plot_confusion_matrix(my_final_employee_model, X_test_PCA, y_test, cmap='Reds');
```


    
![png](output_118_0.png)
    


The model predicted:

**379 True Negatives** meaning it correctly predicted **class 0, 379 times**

**44 True Positives** meaning it correctly predicted **class 1, 44 times**

**64 False Positives** meaning it **falsely predicted** class 1, 64 times

**16 False Negatives** meaning it **falsely predicted** class 0, 16 times


Compared to the original:

**383 True Negatives** meaning it correctly predicted **class 0, 382 times**

**42 True Positives** meaning it correctly predicted **class 1, 42 times**

**60 False Positives** meaning it **falsely predicted** class 1, 60 times

**18 False Negatives** meaning it **falsely predicted** class 0, 18 times



Although not significant, optimizing our hyperparameters resulted in slight changes in the predictions. Given that the number of True positives and False positives increase, I would assume that the recall rate increase. We will verify this assumption from the classification report.


```python
# Optimized model Classification report

cf_test_report = classification_report(y_test, y_test_pred)
print(cf_test_report)
```

                  precision    recall  f1-score   support
    
               0       0.96      0.86      0.90       443
               1       0.41      0.73      0.52        60
    
        accuracy                           0.84       503
       macro avg       0.68      0.79      0.71       503
    weighted avg       0.89      0.84      0.86       503
    
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.663 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.831 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.840 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.792 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.671 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.801 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.676 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.628 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.657 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.683 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.702 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.874 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.864 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.864 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.847 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.802 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.674 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.712 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.903 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.864 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.688 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.608 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.542 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.763 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.809 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.771 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.794 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.725 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.652 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.686 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.676 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.680 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.754 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.766 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.870 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.725 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.652 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.686 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.656 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.674 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.712 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.898 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.812 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.841 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.795 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.809 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.882 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.778 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.1s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.896 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.864 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.679 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.628 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.663 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.649 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.848 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.777 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.705 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.635 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.688 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.678 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.663 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.766 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.758 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.777 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.705 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.635 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.674 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.712 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.898 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.695 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.685 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.860 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.823 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.835 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.882 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.853 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.833 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.833 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.802 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.899 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.674 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.712 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.898 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.898 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.903 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.819 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l1, model__random_state=1, model__solver=liblinear, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.795 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.690 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.683 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1e-05, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.857 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.860 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.847 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.795 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.0001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.852 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.869 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.795 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.862 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.001, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.855 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.816 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.864 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.864 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.800 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.874 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.862 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.797 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.814 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.881 total time=   0.1s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.857 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.903 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.785 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.797 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.684 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.652 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.862 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.872 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.831 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.1s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.879 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.835 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.823 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.812 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.881 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.638 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.674 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.712 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.775 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=3, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.894 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.903 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.816 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.01, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.872 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.650 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.688 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.847 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.874 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.852 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=0.1, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.877 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.727 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.647 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.691 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=MinMaxScaler();, score=0.692 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.773 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.843 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.790 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.824 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.886 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.1s
    [CV 1/5] END model=LogisticRegression(), model__C=1.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.1s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.886 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.836 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=4, scaling=StandardScaler();, score=0.862 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.865 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.855 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.860 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.850 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.821 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.845 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.782 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.792 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.870 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.804 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.896 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.877 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=1000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=StandardScaler();, score=0.869 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.787 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=1, scaling=StandardScaler();, score=0.799 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.681 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.655 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.696 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.668 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=MinMaxScaler();, score=0.659 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=2, scaling=StandardScaler();, score=0.768 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.874 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.833 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=StandardScaler();, score=0.884 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.807 total time=   0.0s
    [CV 2/5] END model=LogisticRegression(), model__C=10000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.891 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=MinMaxScaler();, score=0.802 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=5, scaling=StandardScaler();, score=0.853 total time=   0.0s
    [CV 4/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=6, scaling=MinMaxScaler();, score=0.889 total time=   0.0s
    [CV 3/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=7, scaling=MinMaxScaler();, score=0.867 total time=   0.0s
    [CV 1/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=MinMaxScaler();, score=0.807 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=8, scaling=StandardScaler();, score=0.891 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=9, scaling=StandardScaler();, score=0.889 total time=   0.0s
    [CV 5/5] END model=LogisticRegression(), model__C=100000.0, model__penalty=l2, model__random_state=1, model__solver=lbfgs, reduce_dim=PCA(), reduce_dim__n_components=10, scaling=StandardScaler();, score=0.891 total time=   0.0s


After optimizing the hyperparameters using a pipeline GridSearch, the class 1 precision rate remained the same at 41%, the recall rate increased from 70% to 73%. The f1-score remained the same at 52% and the test accuracy also remained the same at 84%.

As I assumed, the recall rate slightly increased. Again, given our use case, where it is preferred to detect as many attrition possibilities, optimizing the hyperparameters actually worked in our favor and improved our model performance





## This is the end of our Logistic Regression Model.

The next step will be deploying the model into production and create a tool that management will be able to use to retain attrition probabilities of their employees.


```python

```
