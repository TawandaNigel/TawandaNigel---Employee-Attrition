### Project: Understanding Employee Attrition with Machine Learning 
### Activity: Data Cleaning and Exploratory Data Analysis 
### Author: Tawanda Nigel Chitapi
### Email: nigel.chitapi@gmail.com    
### Date: September 05, 2022
### Institution: BrainStation

#### The purpose of this project seeks to understand employee attrition within an organization. The dataset used in this analysis is synthetic data created by IBM Data Scientists for the purposes of HR Analytics. The data was tailored to represent helthcare workers.  

#### This notebook will primarily focus on data cleaning and exploratory data analysis to better understand the various factors that potentially influence employee attrition. The target  variable is "Attrition".

#### After cleaning the data and conducting exhaustive exploratoty analysis, Machine Learning models will be explored in separate notebooks. 


```python
# first we will load in the primary python library packages we use most often 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy import stats
from scipy.stats import norm 
```


```python
# load in the data 

data_df = pd.read_csv('data/watson_healthcare_modified.csv')
```


```python
# sanity check to see if the data has loaded successfully

data_df.head(10)
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
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>Shift</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1313919</td>
      <td>41</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Cardiology</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200302</td>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Maternity</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1060315</td>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Maternity</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1272912</td>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Maternity</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1414939</td>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Maternity</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1633361</td>
      <td>32</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1005</td>
      <td>Maternity</td>
      <td>2</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1329390</td>
      <td>59</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1324</td>
      <td>Maternity</td>
      <td>3</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>3</td>
      <td>12</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1699288</td>
      <td>30</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1358</td>
      <td>Maternity</td>
      <td>24</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1469740</td>
      <td>38</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>216</td>
      <td>Maternity</td>
      <td>23</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>3</td>
      <td>9</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1101291</td>
      <td>36</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1299</td>
      <td>Maternity</td>
      <td>27</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>2</td>
      <td>17</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 35 columns</p>
</div>




```python
# check the shape of the data
data_df.shape
```




    (1676, 35)



The data consists of 1676 rows and 35 columns. The dataset is not significantly large in the world data, this is to say, this dataset is not regarded as Big Data. It is fair to assume that the data set would be representative of hospital employees at a relatively large hospital, and so we will make the assumption that we working on case of hospital employees.


```python
#check the structure of the data 
data_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1676 entries, 0 to 1675
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   EmployeeID                1676 non-null   int64 
     1   Age                       1676 non-null   int64 
     2   Attrition                 1676 non-null   object
     3   BusinessTravel            1676 non-null   object
     4   DailyRate                 1676 non-null   int64 
     5   Department                1676 non-null   object
     6   DistanceFromHome          1676 non-null   int64 
     7   Education                 1676 non-null   int64 
     8   EducationField            1676 non-null   object
     9   EmployeeCount             1676 non-null   int64 
     10  EnvironmentSatisfaction   1676 non-null   int64 
     11  Gender                    1676 non-null   object
     12  HourlyRate                1676 non-null   int64 
     13  JobInvolvement            1676 non-null   int64 
     14  JobLevel                  1676 non-null   int64 
     15  JobRole                   1676 non-null   object
     16  JobSatisfaction           1676 non-null   int64 
     17  MaritalStatus             1676 non-null   object
     18  MonthlyIncome             1676 non-null   int64 
     19  MonthlyRate               1676 non-null   int64 
     20  NumCompaniesWorked        1676 non-null   int64 
     21  Over18                    1676 non-null   object
     22  OverTime                  1676 non-null   object
     23  PercentSalaryHike         1676 non-null   int64 
     24  PerformanceRating         1676 non-null   int64 
     25  RelationshipSatisfaction  1676 non-null   int64 
     26  StandardHours             1676 non-null   int64 
     27  Shift                     1676 non-null   int64 
     28  TotalWorkingYears         1676 non-null   int64 
     29  TrainingTimesLastYear     1676 non-null   int64 
     30  WorkLifeBalance           1676 non-null   int64 
     31  YearsAtCompany            1676 non-null   int64 
     32  YearsInCurrentRole        1676 non-null   int64 
     33  YearsSinceLastPromotion   1676 non-null   int64 
     34  YearsWithCurrManager      1676 non-null   int64 
    dtypes: int64(26), object(9)
    memory usage: 458.4+ KB


The data types included are of **'object' and 'integer'** type. Object type refers to 'strings' or 'words' and the integer type referes to 'whole numbers'. 


```python
# check for any null values in the dataset

data_df.isnull().sum()
```




    EmployeeID                  0
    Age                         0
    Attrition                   0
    BusinessTravel              0
    DailyRate                   0
    Department                  0
    DistanceFromHome            0
    Education                   0
    EducationField              0
    EmployeeCount               0
    EnvironmentSatisfaction     0
    Gender                      0
    HourlyRate                  0
    JobInvolvement              0
    JobLevel                    0
    JobRole                     0
    JobSatisfaction             0
    MaritalStatus               0
    MonthlyIncome               0
    MonthlyRate                 0
    NumCompaniesWorked          0
    Over18                      0
    OverTime                    0
    PercentSalaryHike           0
    PerformanceRating           0
    RelationshipSatisfaction    0
    StandardHours               0
    Shift                       0
    TotalWorkingYears           0
    TrainingTimesLastYear       0
    WorkLifeBalance             0
    YearsAtCompany              0
    YearsInCurrentRole          0
    YearsSinceLastPromotion     0
    YearsWithCurrManager        0
    dtype: int64



There are zero null values, this means that all data records (rows) contain data points from all the respective columns in the dataset


```python
# check for any duplicated rows

duplicate_rows = data_df.duplicated().sum()

print(f'There are {duplicate_rows} duplicate rows')
```

    There are 0 duplicate rows



```python
# check number of duplicate columns 

duplicate_columns = data_df.T.duplicated().sum()

print(f'There are {duplicate_columns} duplicate columns')
```

    There are 0 duplicate rows


No rows or columns have been duplicated. Each row and column is unique and does not cointain duplicate data from another row or column nor do the columns have the same name as that of another column.

Our data appears to be considerably clean at least on a high level after conducting the fundamental data cleaning checks. We will now move on to conduct exploratory data analysis on the dataset. 

During this process we may encounter more in depth information about the data that may induce us to conduct further data cleaning in addition to what we have already done. 


This may include some degree of feature engineering, but first we will separate our data into numeric data **'number'** and categorical data **'object'** groups, in order to conduct exploratory data analysis as they require different methods of analysis 


```python
#separate the numeric and categorical columns 

numeric_col_list = list(data_df.select_dtypes("number").columns)

categorical_col_list = list(data_df.select_dtypes("object").columns)


```


```python
# sanity check to see if the data has been successfully separated 

print(numeric_col_list)
```

    ['EmployeeID', 'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'Shift', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']



```python
# sanity check to see if the data has been successfully separated 

print(categorical_col_list)
```

    ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']


Now that our data has been cleaned up, we will now conduct some exploratory data analysis to better understand the how the various features influence employee attrition. 

### EDA - Exploratory Data Analysis

Defined, Exploratory Data Analysis refers to the critical process of performing initial investigations on data to discover patterns, anomalies, to test hypothesis and check assumptions with the help of summary statistics with graphical representations.


It is a good practice to understand the data first and try to gather as many insights from it. EDA is all about making sense of data in hand,before getting dirty with it.

First we will begin with Univariate analysis of the numerical data. Univariate means we will be analyzing the numeric data columns individually to try and investigate what the data in each column inferes about employees and make assumptions of the inferences it might have our target feature (variable.)


#### Univariate analysis for numeric variables (numerical features)


```python
df_numeric = data_df[numeric_col_list].copy()
df_numeric.head()
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
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>Shift</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
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
      <td>94</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
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
      <td>61</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
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
      <td>92</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>56</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
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
      <td>40</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



Pre-processing the numeric columns, we will check the distributions if they are skewed or normal and visually inspect the data for outliers.

We will use a seaborn pairplot to achieve this.

The seaborn pairplot will give us a give look at the data and potential problems with correlations. The diagonal gives the histograms for the columns and the other tiles show scatter plots between the two variables.


```python
sns.pairplot(df_numeric)
plt.show()
```


    
![png](output_24_0.png)
    


- Given the large number of features, the pairplot does not give us a very clear and easily interpretable presentation of the feature relationships, we will use a correlation heatmap, futher down in the notebook to get a clear understanding of the feature relationships.

### Dictionary of numerical value representations of some of the numerical features


Education
1 'Below College'
2 'College'
3 'Bachelor'
4 'Master'
5 'Doctor'

EnvironmentSatisfaction
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

JobInvolvement 
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

JobSatisfaction 
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

PerformanceRating 
1 'Low'
2 'Good'
3 'Excellent'
4 'Outstanding'

RelationshipSatisfaction 
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

WorkLifeBalance 
1 'Bad'
2 'Good'
3 'Better'
4 'Best'


```python
# plot histograms for each of the numeric data columns 
for column in df_numeric.columns:
    sns.histplot(x=column, data=df_numeric)
    plt.title(column)
    plt.show()
```


    
![png](output_27_0.png)
    



    
![png](output_27_1.png)
    



    
![png](output_27_2.png)
    



    
![png](output_27_3.png)
    



    
![png](output_27_4.png)
    



    
![png](output_27_5.png)
    



    
![png](output_27_6.png)
    



    
![png](output_27_7.png)
    



    
![png](output_27_8.png)
    



    
![png](output_27_9.png)
    



    
![png](output_27_10.png)
    



    
![png](output_27_11.png)
    



    
![png](output_27_12.png)
    



    
![png](output_27_13.png)
    



    
![png](output_27_14.png)
    



    
![png](output_27_15.png)
    



    
![png](output_27_16.png)
    



    
![png](output_27_17.png)
    



    
![png](output_27_18.png)
    



    
![png](output_27_19.png)
    



    
![png](output_27_20.png)
    



    
![png](output_27_21.png)
    



    
![png](output_27_22.png)
    



    
![png](output_27_23.png)
    



    
![png](output_27_24.png)
    



    
![png](output_27_25.png)
    


- The EmployeeID does not necessarily impact our target variable as it only helps identify employees and a not tell us more about the employee or the nature of their job

- The ages of employees are normally distributed, with most of the employees aged between 25 years and 50 years

- The daily rate for employees ranges from (200 - 1400), we can assume that employees with more experience, more years on the job, higher education are paid more, this can be further verified when we test for correlations

- Distance from home has a right skewed distribution, most employees do not live too far from their place of work, (the hospital), most employees stay within 10km of the hospital

- Most employees hold a bachelor’s degree **(3)** followed by those with a master’s degree **(4)**, less than 200 employees do not have college level **(2)** education and less than 100 employees hold Doctorate degrees **(5)** 

- About 50% of the employees are satisfied with their job environment as they have rated it as 'high' and 'very high'

- The hourly rate ranges from **(30/hr - 100/hr)**, we can assume that employees with more experience, more years on the job, higher education are paid more, this can be further verified when we test for correlations

- Many employees are highly involved in their job, with this, we can assume that their input and participation is highly valued and recognized, resulting in employees being encouraged to be more involved, this may vary with department and job role.

- The job level chart is right skewed, and this suggests that most of the employees are enrolled in lower level jobs compared to higher level jobs 

- Monthly Income ranges from **(1,250 - 20,000)** the graph is also right skewed, which suggests that most employees earn a monthly income on the lower end and few employees earn monthly income on the higher end. This can be related to job level as we have assumed above

- Most employees have worked for fewer companies in their career and very few have worked for more than 5 companies. Those that have worked for less companies may be part of the group that has been with the hospital for fewer years, we can assume that they have not had the opportunity to change jobs, or they may be part of the group that has worked for the hospital for many years and have not switched companies.

- The Percent Salary Hike graph ranges from 12% to 24% and it is also right skewed, this suggests that most employees receive a lower percentage pay increase and very few receive a high percentage pay increase. we can assume that the higher percentage pay increase hikes are associated with higher level jobs as they possibly undertake more intricate jobs that involve high risk and provide high rewards in return

- Employees are overall satisfied with their work relationships, most of the employees have been working for at least 10 years of their career and this mostly translates to most of them having worked at the hospital for between 0 and 10 years, we can assume that most employees enjoy and love their work and place of work thus they do not move away in large numbers


### Now we will move on to conducting a univariate analysis on the categorical features

#### Univariate analysis on categorical variables (categorical features)


```python
# sanity check to see if the categorical columns 
print(categorical_col_list)
```

    ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']



```python
# create a dataframe for the categorical data
df_categorical = data_df[categorical_col_list].copy()
df_categorical.head()
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
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>Department</th>
      <th>EducationField</th>
      <th>Gender</th>
      <th>JobRole</th>
      <th>MaritalStatus</th>
      <th>Over18</th>
      <th>OverTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Cardiology</td>
      <td>Life Sciences</td>
      <td>Female</td>
      <td>Nurse</td>
      <td>Single</td>
      <td>Y</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>Maternity</td>
      <td>Life Sciences</td>
      <td>Male</td>
      <td>Other</td>
      <td>Married</td>
      <td>Y</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>Maternity</td>
      <td>Other</td>
      <td>Male</td>
      <td>Nurse</td>
      <td>Single</td>
      <td>Y</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>Maternity</td>
      <td>Life Sciences</td>
      <td>Female</td>
      <td>Other</td>
      <td>Married</td>
      <td>Y</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Maternity</td>
      <td>Medical</td>
      <td>Male</td>
      <td>Nurse</td>
      <td>Married</td>
      <td>Y</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check Attrition class balance
data_df['Attrition'].value_counts()
```




    No     1477
    Yes     199
    Name: Attrition, dtype: int64



88% of employees stayed on the job and 12% quit. The classes are extremely imbalanced, the data will have to be upsampled in order to improve model training and model performance 


```python
#Pull out the category (i.e. object information only)
categorical_df = data_df.select_dtypes('object')

#Iterate through all my object datatype columns
for column in categorical_df:

#Plot my results
    categorical_df[column].value_counts().sort_values().plot(kind='barh')
    plt.title(column)
    plt.show()
```


    
![png](output_35_0.png)
    



    
![png](output_35_1.png)
    



    
![png](output_35_2.png)
    



    
![png](output_35_3.png)
    



    
![png](output_35_4.png)
    



    
![png](output_35_5.png)
    



    
![png](output_35_6.png)
    



    
![png](output_35_7.png)
    



    
![png](output_35_8.png)
    


Analyzing the above, we can learn:

- About 12% of employees attritioned 
- 	Most employees rarely travel for business, this is about 72% of the employees
- 	The department most represented at the hospital is the Maternity department, followed by the cardiology department, and lastly the Neurology respectfully
- 	Most employees come from the Life Sciences education field, followed by the medical education field, this is quite representative of a hospital environment. 
- 	Employees at the hospital are predominantly male, making up about a 1000 of the 1676 employees
- 	The most populated Job Role is that of a Nurse, followed by Other, then Therapist, followed by Administrative and Admin. 
- In this instance we do not have access to a data dictionary and so I will assume that Administrative and Admin are the same
- The "Other" job roles have a significantly large number of employees and so it will benefit our analysis more if we knew and understood more about the roles encapsuled within "Other".
- 	Most employees are Married about 800 of them followed by Single ones and lastly a fair share of those who are divorced, just under 400 of them.
- 	All employees at the hospital are over the age of 18
- 	A significant number of employees **do not work over-time** at the hospital. About 500 employees worked overtime. This may vary with different departments and staff demand in the respective departments. Some departments may not necessarily have to work over-time. I would assume that the administrative department does not work as much over time as the nurses would. This can be further explored to get a true reflection of which employees mostly work over-time and which ones do not


#### Bivariate analysis on categorical variables (categorical features) against the target variable "Attrition"

The Bivariate analysis investigates the relation between each categorial variable and the target variable to help us understand the two classes of the Attrition variable, we want to better understand the behavior traits of those who quit and left their jobs vs those that have continued to remain on the job

**Employees that left their job are marked by "yes" and those that remained are marked by "yes"**


```python
#Check the relationship of the BusinessTravel column with our target. 

total_count = df_categorical.groupby(['BusinessTravel', 'Attrition'])['BusinessTravel'].count()
pct_Business_Travel = total_count/df_categorical.groupby('BusinessTravel')['BusinessTravel'].count()

pct_Business_Travel = pct_Business_Travel.unstack()
#we will restack these to plot the chart
pct_Business_Travel.columns = ['stayed', 'attritioned']
pct_Business_Travel.index = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
```


```python
#Plot the above
pct_Business_Travel.plot(kind='bar', stacked=False)
sns.despine()
plt.title("Business_Travel vs. Attrition")
plt.show()
```


    
![png](output_40_0.png)
    


Most employees that left their job traveled frequently, about 18% of those that travelled frequently. Business travel may be assumed to have adversely affected the employees. 


```python
#Check the relationship of the Department column with our target. 


total_count = df_categorical.groupby(['Department', 'Attrition'])['Department'].count()
pct_Department = total_count/df_categorical.groupby('Department')['Department'].count()

pct_Department = pct_Department.unstack()
#we will restack these to plot the chart
pct_Department.columns = ['stayed', 'attritioned']
pct_Department.index = ['Maternity', 'Cardiology', 'Neurology']
```


```python
#Plot the above
pct_Department.plot(kind='bar', stacked=False)
sns.despine()
plt.title("Department vs. Attrition")
plt.show()
```


    
![png](output_43_0.png)
    


Most employees that left their job worked in the Maternity department, followed by Cardiology and Neurology respectively. We can make the assumption that the Maternity department tends to be very busy and physically demanding. It would be worth looking into the Materninty department compansation and how many years employees tend to to work in the department to better understand reason for employee attrition.  


```python
#Check the relationship of the EducationField column with our target
total_count = df_categorical.groupby(['EducationField', 'Attrition'])['EducationField'].count()
pct_EducationField = total_count/df_categorical.groupby('EducationField')['EducationField'].count()

pct_EducationField = pct_EducationField.unstack()
#we will restack these to plot the chart
pct_EducationField.columns = ['stayed', 'attritioned']
pct_EducationField.index = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree',
                             'Other','Human Resources']
```


```python
#Plot the above
pct_EducationField.plot(kind='bar', stacked=False)
sns.despine()
plt.title("EducationField vs. Attrition")
plt.show()
```


    
![png](output_46_0.png)
    


Most employees that quit their jobs had a Life Sciences educational background. About 21% of them attritioned. We could make the assumption that a Life Sciences degree is foundational education for health care workers and so a possible reason reason fro employee attrition could be that they leave to pursue more specialized health care education 


```python
#Check the relationship of the Gender column with our target
total_count = df_categorical.groupby(['Gender', 'Attrition'])['Gender'].count()
pct_Gender= total_count/df_categorical.groupby('Gender')['Gender'].count()

pct_Gender = pct_Gender.unstack()
#we will restack these to plot the chart
pct_Gender.columns = ['stayed', 'attritioned']
pct_Gender.index = ['Male', 'Female']
```


```python
#Plot the above
pct_Gender.plot(kind='bar', stacked=False)
sns.despine()
plt.title("Gender vs. Attrition")
plt.show()
```


    
![png](output_49_0.png)
    


A similar amount of male and female employees quit their job, about 16% male and 15% female, with this we can deductively say that gender does not significantly impact attrition rates and or causes. 


```python
#Check the relationship of the JobRole column with our target

total_count = df_categorical.groupby(['JobRole', 'Attrition'])['JobRole'].count()
pct_JobRole = total_count/df_categorical.groupby('JobRole')['JobRole'].count()

pct_JobRole = pct_JobRole.unstack()
#we will restack these to plot the chart
pct_JobRole.columns = ['stayed', 'attritioned']
pct_JobRole.index = ['Nurse', 'Other', 'Therapist', 'Administrative', 'Admin']
```


```python
#Plot the above
pct_JobRole.plot(kind='bar', stacked=False)
sns.despine()
plt.title("JobRole vs. Attrition")
plt.show()
```


    
![png](output_52_0.png)
    


A significant propotion of employees that quit thier jobs were employeed in Adminstrative roles and Therapists

- The Adminstative and Admin roles seem to be the same, since Admin is short for Adminstrative, this is an issue with the data itself, with the absence of a data dictionary, we cannot quite differentiate the two from each other

- Adminstrative roles encapsule many other specific roles within them and so we are not able to pin point the specific roles
- Less than 5% of employees from the Admin roles attritioned and so for the purposes of this analysis, our lack of understanding of what the difference between the "Admin" roles and the "Adminstrative" roles will not hinder our analysis

- Therapist is a more distinct role, and so, having a significantly high attrition rate is a cause for concern and requires further investigation to understand why the Therapists are leaving the hospital 


```python
#Check the relationship of the MaritalStatus column with our target

total_count = df_categorical.groupby(['MaritalStatus', 'Attrition'])['MaritalStatus'].count()
pct_MaritalStatus = total_count/df_categorical.groupby('MaritalStatus')['MaritalStatus'].count()

pct_MaritalStatus = pct_MaritalStatus.unstack()
#we will restack these to plot the chart
pct_MaritalStatus.columns = ['stayed', 'attritioned']
pct_MaritalStatus.index = ['Married', 'Single', 'Divorced']
```


```python
#Plot the above
pct_MaritalStatus.plot(kind='bar', stacked=False)
sns.despine()
plt.title("MaritalStatus vs. Attrition")
plt.show()
```


    
![png](output_55_0.png)
    


About 21% of Divorced employees left their job. This rate was significanlty high compared to the married and single employees. This calls for further investigation of reasons why Divorced employees are quitting thier jobs. 


```python
#Check the relationship of the OverTime column with our target

total_count = df_categorical.groupby(['OverTime', 'Attrition'])['OverTime'].count()
pct_Business_Travel = total_count/df_categorical.groupby('OverTime')['OverTime'].count()

pct_Business_Travel = pct_Business_Travel.unstack()
#we will restack these to plot the chart
pct_Business_Travel.columns = ['stayed', 'attritioned']
pct_Business_Travel.index = ['OverTime - Yes', 'OverTime - No']
```


```python
#Plot the above
pct_Business_Travel.plot(kind='bar', stacked=False)
sns.despine()
plt.title("OverTime vs. Attrition")
plt.show()
```


    
![png](output_58_0.png)
    


Surprisingly, the majority of employees that quit their jobs did not neccessarily work over-time. An easy assumption may have been that, employees quit their jobs after working long hours and burning out, however, this is not the case.


```python
#Check the relationship of the Over18 column with our target

total_count = df_categorical.groupby(['Over18', 'Attrition'])['Over18'].count()
pct_Over18 = total_count/df_categorical.groupby('Over18')['Over18'].count()

pct_Over18 = pct_Over18.unstack()
#we will restack these to plot the chart
pct_Over18.columns = ['stayed', 'attritioned']
pct_Over18.index = ['Yes']
```


```python
#Plot the above
pct_Over18.plot(kind='bar', stacked=False)
sns.despine()
plt.title("Over18 vs. Attrition")
plt.show()
```


    
![png](output_61_0.png)
    


All employees that work at the hospital are above the age of 18 and so  we only have the "YES" column.

- Given that all employees were above the age of 18, this graph simply represents all hospital employees as well 

- From the graph above we can safely say that about 12% of the total hospital employees left their jobs 

### Correlation of numerical features

It is important to assess for correlation of numerical features in order to understand how the variables relate with each other and most importantly if any variables are strongly correlated, it is important to take note of them and try to avaoid to use them both to predict the target variable. The numerical variables are part of the indepedent variables and so they should be indepedent of each other in order to best predict the target variables.


```python
# calculate correlation of numerical variables
data_df.corr()
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
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>Shift</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EmployeeID</th>
      <td>1.000000</td>
      <td>-0.039033</td>
      <td>-0.002558</td>
      <td>-0.031648</td>
      <td>-0.000781</td>
      <td>NaN</td>
      <td>-0.005930</td>
      <td>0.000007</td>
      <td>-0.062473</td>
      <td>-0.030085</td>
      <td>...</td>
      <td>-0.000650</td>
      <td>NaN</td>
      <td>-0.003884</td>
      <td>-0.031294</td>
      <td>0.014934</td>
      <td>0.033284</td>
      <td>-0.018060</td>
      <td>-0.014384</td>
      <td>0.031340</td>
      <td>-0.013707</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.039033</td>
      <td>1.000000</td>
      <td>0.001441</td>
      <td>-0.010079</td>
      <td>0.204655</td>
      <td>NaN</td>
      <td>0.008945</td>
      <td>0.034671</td>
      <td>0.034193</td>
      <td>0.518333</td>
      <td>...</td>
      <td>0.058528</td>
      <td>NaN</td>
      <td>0.037117</td>
      <td>0.692512</td>
      <td>-0.015408</td>
      <td>-0.004878</td>
      <td>0.319012</td>
      <td>0.222655</td>
      <td>0.217212</td>
      <td>0.215909</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>-0.002558</td>
      <td>0.001441</td>
      <td>1.000000</td>
      <td>-0.009227</td>
      <td>-0.015881</td>
      <td>NaN</td>
      <td>0.010620</td>
      <td>0.027128</td>
      <td>0.058864</td>
      <td>0.009005</td>
      <td>...</td>
      <td>0.014539</td>
      <td>NaN</td>
      <td>0.054407</td>
      <td>0.009378</td>
      <td>0.001901</td>
      <td>-0.028549</td>
      <td>-0.026892</td>
      <td>0.019651</td>
      <td>-0.034571</td>
      <td>-0.025272</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>-0.031648</td>
      <td>-0.010079</td>
      <td>-0.009227</td>
      <td>1.000000</td>
      <td>0.015937</td>
      <td>NaN</td>
      <td>-0.019730</td>
      <td>0.026947</td>
      <td>0.010281</td>
      <td>-0.023455</td>
      <td>...</td>
      <td>0.005482</td>
      <td>NaN</td>
      <td>0.029180</td>
      <td>-0.017663</td>
      <td>-0.055471</td>
      <td>-0.037821</td>
      <td>-0.007420</td>
      <td>0.011448</td>
      <td>-0.000126</td>
      <td>0.000403</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>-0.000781</td>
      <td>0.204655</td>
      <td>-0.015881</td>
      <td>0.015937</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.031925</td>
      <td>0.017996</td>
      <td>0.041046</td>
      <td>0.093227</td>
      <td>...</td>
      <td>-0.005750</td>
      <td>NaN</td>
      <td>0.024451</td>
      <td>0.143324</td>
      <td>-0.014070</td>
      <td>0.003933</td>
      <td>0.057461</td>
      <td>0.051029</td>
      <td>0.045785</td>
      <td>0.055096</td>
    </tr>
    <tr>
      <th>EmployeeCount</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>EnvironmentSatisfaction</th>
      <td>-0.005930</td>
      <td>0.008945</td>
      <td>0.010620</td>
      <td>-0.019730</td>
      <td>-0.031925</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.057505</td>
      <td>-0.007220</td>
      <td>0.008277</td>
      <td>...</td>
      <td>0.003221</td>
      <td>NaN</td>
      <td>0.005283</td>
      <td>0.000208</td>
      <td>-0.017722</td>
      <td>0.027262</td>
      <td>-0.000448</td>
      <td>0.012241</td>
      <td>0.005866</td>
      <td>-0.012417</td>
    </tr>
    <tr>
      <th>HourlyRate</th>
      <td>0.000007</td>
      <td>0.034671</td>
      <td>0.027128</td>
      <td>0.026947</td>
      <td>0.017996</td>
      <td>NaN</td>
      <td>-0.057505</td>
      <td>1.000000</td>
      <td>0.034741</td>
      <td>-0.018830</td>
      <td>...</td>
      <td>0.000601</td>
      <td>NaN</td>
      <td>0.051862</td>
      <td>0.005988</td>
      <td>-0.007194</td>
      <td>-0.009956</td>
      <td>-0.014742</td>
      <td>-0.016776</td>
      <td>-0.028642</td>
      <td>-0.021436</td>
    </tr>
    <tr>
      <th>JobInvolvement</th>
      <td>-0.062473</td>
      <td>0.034193</td>
      <td>0.058864</td>
      <td>0.010281</td>
      <td>0.041046</td>
      <td>NaN</td>
      <td>-0.007220</td>
      <td>0.034741</td>
      <td>1.000000</td>
      <td>-0.013660</td>
      <td>...</td>
      <td>0.045107</td>
      <td>NaN</td>
      <td>0.025999</td>
      <td>-0.001576</td>
      <td>-0.031580</td>
      <td>-0.006931</td>
      <td>-0.013652</td>
      <td>0.020541</td>
      <td>-0.022153</td>
      <td>0.031574</td>
    </tr>
    <tr>
      <th>JobLevel</th>
      <td>-0.030085</td>
      <td>0.518333</td>
      <td>0.009005</td>
      <td>-0.023455</td>
      <td>0.093227</td>
      <td>NaN</td>
      <td>0.008277</td>
      <td>-0.018830</td>
      <td>-0.013660</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.030606</td>
      <td>NaN</td>
      <td>0.010478</td>
      <td>0.780929</td>
      <td>-0.004251</td>
      <td>0.047481</td>
      <td>0.532529</td>
      <td>0.387624</td>
      <td>0.343102</td>
      <td>0.379717</td>
    </tr>
    <tr>
      <th>JobSatisfaction</th>
      <td>-0.007392</td>
      <td>-0.015848</td>
      <td>0.032115</td>
      <td>-0.004758</td>
      <td>-0.003957</td>
      <td>NaN</td>
      <td>0.001518</td>
      <td>-0.073942</td>
      <td>-0.039738</td>
      <td>-0.012497</td>
      <td>...</td>
      <td>-0.018232</td>
      <td>NaN</td>
      <td>0.017740</td>
      <td>-0.021435</td>
      <td>-0.003440</td>
      <td>-0.017280</td>
      <td>0.005376</td>
      <td>-0.001337</td>
      <td>-0.013595</td>
      <td>-0.023042</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>-0.027595</td>
      <td>0.511378</td>
      <td>0.011030</td>
      <td>-0.041201</td>
      <td>0.085116</td>
      <td>NaN</td>
      <td>0.003372</td>
      <td>-0.008443</td>
      <td>-0.019240</td>
      <td>0.951572</td>
      <td>...</td>
      <td>0.033035</td>
      <td>NaN</td>
      <td>0.005847</td>
      <td>0.772938</td>
      <td>-0.009690</td>
      <td>0.039910</td>
      <td>0.513977</td>
      <td>0.364152</td>
      <td>0.337241</td>
      <td>0.350122</td>
    </tr>
    <tr>
      <th>MonthlyRate</th>
      <td>-0.009835</td>
      <td>0.025837</td>
      <td>-0.032211</td>
      <td>0.031672</td>
      <td>-0.019198</td>
      <td>NaN</td>
      <td>0.046771</td>
      <td>-0.025597</td>
      <td>-0.018490</td>
      <td>0.036901</td>
      <td>...</td>
      <td>-0.015690</td>
      <td>NaN</td>
      <td>-0.040083</td>
      <td>0.023876</td>
      <td>0.002747</td>
      <td>0.002448</td>
      <td>-0.032950</td>
      <td>-0.022279</td>
      <td>-0.013589</td>
      <td>-0.053408</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>-0.005114</td>
      <td>0.296045</td>
      <td>0.034296</td>
      <td>-0.024969</td>
      <td>0.126758</td>
      <td>NaN</td>
      <td>0.012640</td>
      <td>0.029132</td>
      <td>0.016303</td>
      <td>0.153179</td>
      <td>...</td>
      <td>0.063381</td>
      <td>NaN</td>
      <td>0.023164</td>
      <td>0.250514</td>
      <td>-0.056122</td>
      <td>-0.005620</td>
      <td>-0.108807</td>
      <td>-0.080578</td>
      <td>-0.025033</td>
      <td>-0.093030</td>
    </tr>
    <tr>
      <th>PercentSalaryHike</th>
      <td>0.006775</td>
      <td>0.007570</td>
      <td>0.019325</td>
      <td>0.034172</td>
      <td>-0.006461</td>
      <td>NaN</td>
      <td>-0.021612</td>
      <td>-0.013240</td>
      <td>-0.019999</td>
      <td>-0.024711</td>
      <td>...</td>
      <td>-0.034936</td>
      <td>NaN</td>
      <td>0.017987</td>
      <td>-0.010131</td>
      <td>-0.029329</td>
      <td>-0.010680</td>
      <td>-0.022281</td>
      <td>0.010046</td>
      <td>-0.000870</td>
      <td>-0.004769</td>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <td>-0.000996</td>
      <td>0.005246</td>
      <td>0.003353</td>
      <td>0.020482</td>
      <td>-0.020664</td>
      <td>NaN</td>
      <td>-0.029104</td>
      <td>-0.000370</td>
      <td>-0.017970</td>
      <td>-0.008759</td>
      <td>...</td>
      <td>-0.027926</td>
      <td>NaN</td>
      <td>0.007546</td>
      <td>0.014407</td>
      <td>-0.036267</td>
      <td>-0.004897</td>
      <td>0.016586</td>
      <td>0.048446</td>
      <td>0.037992</td>
      <td>0.031837</td>
    </tr>
    <tr>
      <th>RelationshipSatisfaction</th>
      <td>-0.000650</td>
      <td>0.058528</td>
      <td>0.014539</td>
      <td>0.005482</td>
      <td>-0.005750</td>
      <td>NaN</td>
      <td>0.003221</td>
      <td>0.000601</td>
      <td>0.045107</td>
      <td>0.030606</td>
      <td>...</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.042412</td>
      <td>0.029257</td>
      <td>0.003090</td>
      <td>0.012302</td>
      <td>0.022223</td>
      <td>-0.014648</td>
      <td>0.040914</td>
      <td>0.003256</td>
    </tr>
    <tr>
      <th>StandardHours</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Shift</th>
      <td>-0.003884</td>
      <td>0.037117</td>
      <td>0.054407</td>
      <td>0.029180</td>
      <td>0.024451</td>
      <td>NaN</td>
      <td>0.005283</td>
      <td>0.051862</td>
      <td>0.025999</td>
      <td>0.010478</td>
      <td>...</td>
      <td>-0.042412</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.009632</td>
      <td>0.025881</td>
      <td>-0.001008</td>
      <td>0.013028</td>
      <td>0.039263</td>
      <td>0.011702</td>
      <td>0.014314</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>-0.031294</td>
      <td>0.692512</td>
      <td>0.009378</td>
      <td>-0.017663</td>
      <td>0.143324</td>
      <td>NaN</td>
      <td>0.000208</td>
      <td>0.005988</td>
      <td>-0.001576</td>
      <td>0.780929</td>
      <td>...</td>
      <td>0.029257</td>
      <td>NaN</td>
      <td>0.009632</td>
      <td>1.000000</td>
      <td>-0.026171</td>
      <td>0.012677</td>
      <td>0.622495</td>
      <td>0.457439</td>
      <td>0.394941</td>
      <td>0.461688</td>
    </tr>
    <tr>
      <th>TrainingTimesLastYear</th>
      <td>0.014934</td>
      <td>-0.015408</td>
      <td>0.001901</td>
      <td>-0.055471</td>
      <td>-0.014070</td>
      <td>NaN</td>
      <td>-0.017722</td>
      <td>-0.007194</td>
      <td>-0.031580</td>
      <td>-0.004251</td>
      <td>...</td>
      <td>0.003090</td>
      <td>NaN</td>
      <td>0.025881</td>
      <td>-0.026171</td>
      <td>1.000000</td>
      <td>0.033482</td>
      <td>0.001739</td>
      <td>-0.007362</td>
      <td>-0.008844</td>
      <td>-0.005057</td>
    </tr>
    <tr>
      <th>WorkLifeBalance</th>
      <td>0.033284</td>
      <td>-0.004878</td>
      <td>-0.028549</td>
      <td>-0.037821</td>
      <td>0.003933</td>
      <td>NaN</td>
      <td>0.027262</td>
      <td>-0.009956</td>
      <td>-0.006931</td>
      <td>0.047481</td>
      <td>...</td>
      <td>0.012302</td>
      <td>NaN</td>
      <td>-0.001008</td>
      <td>0.012677</td>
      <td>0.033482</td>
      <td>1.000000</td>
      <td>0.011721</td>
      <td>0.046360</td>
      <td>0.013051</td>
      <td>0.005276</td>
    </tr>
    <tr>
      <th>YearsAtCompany</th>
      <td>-0.018060</td>
      <td>0.319012</td>
      <td>-0.026892</td>
      <td>-0.007420</td>
      <td>0.057461</td>
      <td>NaN</td>
      <td>-0.000448</td>
      <td>-0.014742</td>
      <td>-0.013652</td>
      <td>0.532529</td>
      <td>...</td>
      <td>0.022223</td>
      <td>NaN</td>
      <td>0.013028</td>
      <td>0.622495</td>
      <td>0.001739</td>
      <td>0.011721</td>
      <td>1.000000</td>
      <td>0.759421</td>
      <td>0.616915</td>
      <td>0.771224</td>
    </tr>
    <tr>
      <th>YearsInCurrentRole</th>
      <td>-0.014384</td>
      <td>0.222655</td>
      <td>0.019651</td>
      <td>0.011448</td>
      <td>0.051029</td>
      <td>NaN</td>
      <td>0.012241</td>
      <td>-0.016776</td>
      <td>0.020541</td>
      <td>0.387624</td>
      <td>...</td>
      <td>-0.014648</td>
      <td>NaN</td>
      <td>0.039263</td>
      <td>0.457439</td>
      <td>-0.007362</td>
      <td>0.046360</td>
      <td>0.759421</td>
      <td>1.000000</td>
      <td>0.548235</td>
      <td>0.721543</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>0.031340</td>
      <td>0.217212</td>
      <td>-0.034571</td>
      <td>-0.000126</td>
      <td>0.045785</td>
      <td>NaN</td>
      <td>0.005866</td>
      <td>-0.028642</td>
      <td>-0.022153</td>
      <td>0.343102</td>
      <td>...</td>
      <td>0.040914</td>
      <td>NaN</td>
      <td>0.011702</td>
      <td>0.394941</td>
      <td>-0.008844</td>
      <td>0.013051</td>
      <td>0.616915</td>
      <td>0.548235</td>
      <td>1.000000</td>
      <td>0.518664</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>-0.013707</td>
      <td>0.215909</td>
      <td>-0.025272</td>
      <td>0.000403</td>
      <td>0.055096</td>
      <td>NaN</td>
      <td>-0.012417</td>
      <td>-0.021436</td>
      <td>0.031574</td>
      <td>0.379717</td>
      <td>...</td>
      <td>0.003256</td>
      <td>NaN</td>
      <td>0.014314</td>
      <td>0.461688</td>
      <td>-0.005057</td>
      <td>0.005276</td>
      <td>0.771224</td>
      <td>0.721543</td>
      <td>0.518664</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>26 rows × 26 columns</p>
</div>




```python
sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(data_df.corr().round(2), vmin=-1, vmax=1, cmap='coolwarm', annot=True, linewidths=1)
plt.show()
```


    
![png](output_66_0.png)
    


From the Correlation heatmap above we observe the following:

- MonthlyIncome is strongly correlated with correlated with JobLevel
- JobLevel is strongly correlated with TotalWorkingYears
- MonthlyIncome is strongly correlated with TotalWorkingYears
- PercentSalaryHike is strongly correlated with PerfomanceRating 
- Age is strongly correlated with TotalWorkingYears
- YearsWithCurrManager is strongly correlated with YeasAtCompany 
- YearsInCurrentRole is strongly correlated with YearsAtCompany

## End of Exploratory Data Analysis



### In order to solve our classification problem we need to convert all categorical columns 'object columns' into numerical columns. There are a few options we could explore to achive this. For columns that have multiple classes that are not in any ordinal order we will use the 'One Hot Encoder' method and for columns that have multiple classes that are ordinal we will use the 'Ordinal Encoder' method. For columns that have only two classes we can simply use the mapping method to binarize the values




### First we will convert the two class features, we will simply binarize them using the mapping method.


```python
# first we will check the classes contained in our target dependent variable which is Attrition

data_df['Attrition'].unique()



```




    array(['No', 'Yes'], dtype=object)




```python
# Binarize Attrition classes

data_df['Attrition'] = np.where(data_df['Attrition'] == 'Yes', 1, 0)

#check if the classes have successfully binarized
data_df.head()
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
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>Shift</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1313919</td>
      <td>41</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Cardiology</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200302</td>
      <td>49</td>
      <td>0</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Maternity</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1060315</td>
      <td>37</td>
      <td>1</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Maternity</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1272912</td>
      <td>33</td>
      <td>0</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Maternity</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1414939</td>
      <td>27</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Maternity</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



## Now we will move on the Gender Column


```python
data_df['Gender'].unique()
```




    array(['Female', 'Male'], dtype=object)




```python
# Binarize the Gendar Column
data_df['Gender'] = np.where(data_df['Gender'] == 'Female', 1, 0)

# sanity check to see if the Gender column has been successfully binarized
data_df['Gender'].head(15)
```




    0     1
    1     0
    2     0
    3     1
    4     0
    5     0
    6     1
    7     0
    8     0
    9     0
    10    0
    11    1
    12    0
    13    0
    14    0
    Name: Gender, dtype: int64




```python
# extra sanity check to see if the Gender column has been successfully been binarized within th edataframe
data_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1676 entries, 0 to 1675
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   EmployeeID                1676 non-null   int64 
     1   Age                       1676 non-null   int64 
     2   Attrition                 1676 non-null   int64 
     3   BusinessTravel            1676 non-null   object
     4   DailyRate                 1676 non-null   int64 
     5   Department                1676 non-null   object
     6   DistanceFromHome          1676 non-null   int64 
     7   Education                 1676 non-null   int64 
     8   EducationField            1676 non-null   object
     9   EmployeeCount             1676 non-null   int64 
     10  EnvironmentSatisfaction   1676 non-null   int64 
     11  Gender                    1676 non-null   int64 
     12  HourlyRate                1676 non-null   int64 
     13  JobInvolvement            1676 non-null   int64 
     14  JobLevel                  1676 non-null   int64 
     15  JobRole                   1676 non-null   object
     16  JobSatisfaction           1676 non-null   int64 
     17  MaritalStatus             1676 non-null   object
     18  MonthlyIncome             1676 non-null   int64 
     19  MonthlyRate               1676 non-null   int64 
     20  NumCompaniesWorked        1676 non-null   int64 
     21  Over18                    1676 non-null   object
     22  OverTime                  1676 non-null   object
     23  PercentSalaryHike         1676 non-null   int64 
     24  PerformanceRating         1676 non-null   int64 
     25  RelationshipSatisfaction  1676 non-null   int64 
     26  StandardHours             1676 non-null   int64 
     27  Shift                     1676 non-null   int64 
     28  TotalWorkingYears         1676 non-null   int64 
     29  TrainingTimesLastYear     1676 non-null   int64 
     30  WorkLifeBalance           1676 non-null   int64 
     31  YearsAtCompany            1676 non-null   int64 
     32  YearsInCurrentRole        1676 non-null   int64 
     33  YearsSinceLastPromotion   1676 non-null   int64 
     34  YearsWithCurrManager      1676 non-null   int64 
    dtypes: int64(28), object(7)
    memory usage: 458.4+ KB


## Now we will move on the Over18 Column


```python
# check the number of classes contained in the Over18 column
data_df['Over18'].unique()
```




    array(['Y'], dtype=object)




```python
# Binarize the Over18 column
data_df['Over18'] = np.where(data_df['Over18'] == 'Y', 1, 0)

# sanity check to see if the Over18 column has been successfully binarized 
data_df['Over18'].head(10)
```




    0    1
    1    1
    2    1
    3    1
    4    1
    5    1
    6    1
    7    1
    8    1
    9    1
    Name: Over18, dtype: int64




```python
# extra sanity check to see if the Over18 column has been successfully been binarized within th edataframe
data_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1676 entries, 0 to 1675
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   EmployeeID                1676 non-null   int64 
     1   Age                       1676 non-null   int64 
     2   Attrition                 1676 non-null   int64 
     3   BusinessTravel            1676 non-null   object
     4   DailyRate                 1676 non-null   int64 
     5   Department                1676 non-null   object
     6   DistanceFromHome          1676 non-null   int64 
     7   Education                 1676 non-null   int64 
     8   EducationField            1676 non-null   object
     9   EmployeeCount             1676 non-null   int64 
     10  EnvironmentSatisfaction   1676 non-null   int64 
     11  Gender                    1676 non-null   int64 
     12  HourlyRate                1676 non-null   int64 
     13  JobInvolvement            1676 non-null   int64 
     14  JobLevel                  1676 non-null   int64 
     15  JobRole                   1676 non-null   object
     16  JobSatisfaction           1676 non-null   int64 
     17  MaritalStatus             1676 non-null   object
     18  MonthlyIncome             1676 non-null   int64 
     19  MonthlyRate               1676 non-null   int64 
     20  NumCompaniesWorked        1676 non-null   int64 
     21  Over18                    1676 non-null   int64 
     22  OverTime                  1676 non-null   object
     23  PercentSalaryHike         1676 non-null   int64 
     24  PerformanceRating         1676 non-null   int64 
     25  RelationshipSatisfaction  1676 non-null   int64 
     26  StandardHours             1676 non-null   int64 
     27  Shift                     1676 non-null   int64 
     28  TotalWorkingYears         1676 non-null   int64 
     29  TrainingTimesLastYear     1676 non-null   int64 
     30  WorkLifeBalance           1676 non-null   int64 
     31  YearsAtCompany            1676 non-null   int64 
     32  YearsInCurrentRole        1676 non-null   int64 
     33  YearsSinceLastPromotion   1676 non-null   int64 
     34  YearsWithCurrManager      1676 non-null   int64 
    dtypes: int64(29), object(6)
    memory usage: 458.4+ KB


## Now we will move on the OverTime Column


```python
data_df['OverTime'].unique()
```




    array(['Yes', 'No'], dtype=object)




```python
# Binarize the Over18 column
data_df['OverTime'] = np.where(data_df['OverTime'] == 'Yes', 1, 0)

# sanity check to see if the Over18 column has been successfully binarized 
data_df['OverTime'].head(10)
```




    0    1
    1    0
    2    1
    3    1
    4    0
    5    0
    6    1
    7    0
    8    0
    9    0
    Name: OverTime, dtype: int64




```python
# extra sanity check to see if the OverTime column has been successfully been binarized within th edataframe
data_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1676 entries, 0 to 1675
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   EmployeeID                1676 non-null   int64 
     1   Age                       1676 non-null   int64 
     2   Attrition                 1676 non-null   int64 
     3   BusinessTravel            1676 non-null   object
     4   DailyRate                 1676 non-null   int64 
     5   Department                1676 non-null   object
     6   DistanceFromHome          1676 non-null   int64 
     7   Education                 1676 non-null   int64 
     8   EducationField            1676 non-null   object
     9   EmployeeCount             1676 non-null   int64 
     10  EnvironmentSatisfaction   1676 non-null   int64 
     11  Gender                    1676 non-null   int64 
     12  HourlyRate                1676 non-null   int64 
     13  JobInvolvement            1676 non-null   int64 
     14  JobLevel                  1676 non-null   int64 
     15  JobRole                   1676 non-null   object
     16  JobSatisfaction           1676 non-null   int64 
     17  MaritalStatus             1676 non-null   object
     18  MonthlyIncome             1676 non-null   int64 
     19  MonthlyRate               1676 non-null   int64 
     20  NumCompaniesWorked        1676 non-null   int64 
     21  Over18                    1676 non-null   int64 
     22  OverTime                  1676 non-null   int64 
     23  PercentSalaryHike         1676 non-null   int64 
     24  PerformanceRating         1676 non-null   int64 
     25  RelationshipSatisfaction  1676 non-null   int64 
     26  StandardHours             1676 non-null   int64 
     27  Shift                     1676 non-null   int64 
     28  TotalWorkingYears         1676 non-null   int64 
     29  TrainingTimesLastYear     1676 non-null   int64 
     30  WorkLifeBalance           1676 non-null   int64 
     31  YearsAtCompany            1676 non-null   int64 
     32  YearsInCurrentRole        1676 non-null   int64 
     33  YearsSinceLastPromotion   1676 non-null   int64 
     34  YearsWithCurrManager      1676 non-null   int64 
    dtypes: int64(30), object(5)
    memory usage: 458.4+ KB


### It is important that we avoid any possibilities of data leakage and so normaly, One Hot Encoding is performed after the train test split to avoid data leakage.

### However, In this case, we are using synthetic data and we are absolutely sure that no new categories will be added to the data and so we will go ahead and perform one hot encoding before the train test split. 

## We will start with the BusinessTravel Column


```python
# first we will import the one hot encoder package from scikit learn

from sklearn.preprocessing import OneHotEncoder
```


```python
data_df['BusinessTravel'].unique()


```




    array(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], dtype=object)




```python
# Instantiate the OneHotEncoder
ohe = OneHotEncoder()
# It expects a 2D array, so we first convert the column into a DataFrame
Business_Travel = pd.DataFrame(data_df['BusinessTravel'])

# Fit the OneHotEncoder to the subcategory column and transform
encoded_bt = ohe.fit_transform(Business_Travel)
encoded_bt
```




    <1676x3 sparse matrix of type '<class 'numpy.float64'>'
    	with 1676 stored elements in Compressed Sparse Row format>



**"_bt"** refers to BusinessTravel


```python
# Convert from sparse matrix to dense array
dense_array_bt = encoded_bt.toarray()
dense_array_bt
```




    array([[0., 0., 1.],
           [0., 1., 0.],
           [0., 0., 1.],
           ...,
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 0., 1.]])




```python
# Check the name of the categories
ohe.categories_
```




    [array(['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'], dtype=object)]




```python
# Put dense array into a dataframe to get column names
encoded_bt_df = pd.DataFrame(dense_array_bt, columns=ohe.categories_[0] , dtype=int)

# Show
encoded_bt_df.head(10)



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
      <th>Non-Travel</th>
      <th>Travel_Frequently</th>
      <th>Travel_Rarely</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check if the number of rows is the same as the original 
encoded_bt_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1676 entries, 0 to 1675
    Data columns (total 3 columns):
     #   Column             Non-Null Count  Dtype
    ---  ------             --------------  -----
     0   Non-Travel         1676 non-null   int64
     1   Travel_Frequently  1676 non-null   int64
     2   Travel_Rarely      1676 non-null   int64
    dtypes: int64(3)
    memory usage: 39.4 KB



```python
# now we will add the encoded columns into the main "Train" data frame
data_df2 = pd.concat([data_df, encoded_bt_df], axis=1)

# sanity check to see if columns have been added successfully
data_df2
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
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>...</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>Non-Travel</th>
      <th>Travel_Frequently</th>
      <th>Travel_Rarely</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1313919</td>
      <td>41</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Cardiology</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200302</td>
      <td>49</td>
      <td>0</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Maternity</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1060315</td>
      <td>37</td>
      <td>1</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Maternity</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>...</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
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
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Maternity</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
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
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Maternity</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>...</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>Travel_Rarely</td>
      <td>471</td>
      <td>Neurology</td>
      <td>24</td>
      <td>3</td>
      <td>Technical Degree</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
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
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>1125</td>
      <td>Cardiology</td>
      <td>10</td>
      <td>3</td>
      <td>Marketing</td>
      <td>1</td>
      <td>...</td>
      <td>15</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>1812428</td>
      <td>20</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>959</td>
      <td>Maternity</td>
      <td>1</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1674</th>
      <td>1812429</td>
      <td>39</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>466</td>
      <td>Neurology</td>
      <td>1</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>...</td>
      <td>21</td>
      <td>3</td>
      <td>3</td>
      <td>21</td>
      <td>6</td>
      <td>11</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1675</th>
      <td>1152329</td>
      <td>27</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>511</td>
      <td>Cardiology</td>
      <td>2</td>
      <td>2</td>
      <td>Medical</td>
      <td>1</td>
      <td>...</td>
      <td>9</td>
      <td>5</td>
      <td>2</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1676 rows × 38 columns</p>
</div>




```python
# drop the original BusinessTravel column
data_df2 = data_df2.drop(['BusinessTravel'], axis = 1)
```


```python
# sanity check to see if the BusinessTravel column has been dropped successfully
data_df2
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
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EnvironmentSatisfaction</th>
      <th>...</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>Non-Travel</th>
      <th>Travel_Frequently</th>
      <th>Travel_Rarely</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1313919</td>
      <td>41</td>
      <td>0</td>
      <td>1102</td>
      <td>Cardiology</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
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
      <td>Maternity</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
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
      <td>Maternity</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
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
      <td>Maternity</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
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
      <td>Maternity</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>471</td>
      <td>Neurology</td>
      <td>24</td>
      <td>3</td>
      <td>Technical Degree</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
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
      <td>0</td>
      <td>1125</td>
      <td>Cardiology</td>
      <td>10</td>
      <td>3</td>
      <td>Marketing</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>15</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>1812428</td>
      <td>20</td>
      <td>0</td>
      <td>959</td>
      <td>Maternity</td>
      <td>1</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1674</th>
      <td>1812429</td>
      <td>39</td>
      <td>0</td>
      <td>466</td>
      <td>Neurology</td>
      <td>1</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>21</td>
      <td>3</td>
      <td>3</td>
      <td>21</td>
      <td>6</td>
      <td>11</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1675</th>
      <td>1152329</td>
      <td>27</td>
      <td>0</td>
      <td>511</td>
      <td>Cardiology</td>
      <td>2</td>
      <td>2</td>
      <td>Medical</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>9</td>
      <td>5</td>
      <td>2</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1676 rows × 37 columns</p>
</div>



**BusinessTravel column has been successfully dropped**

## Now we will move on to the Department Column


```python
data_df2['Department'].unique()
```




    array(['Cardiology', 'Maternity', 'Neurology'], dtype=object)




```python
# Instantiate the OneHotEncoder
ohe = OneHotEncoder()
# It expects a 2D array, so we first convert the column into a DataFrame
Department = pd.DataFrame(data_df2['Department'])

# Fit the OneHotEncoder to the subcategory column and transform
encoded_d = ohe.fit_transform(Department)
encoded_d
```




    <1676x3 sparse matrix of type '<class 'numpy.float64'>'
    	with 1676 stored elements in Compressed Sparse Row format>



**"_d"** refers to Department


```python
# Convert from sparse matrix to dense array
dense_array_d = encoded_d.toarray()
dense_array_d
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           ...,
           [0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.]])




```python
# check categories names
ohe.categories_
```




    [array(['Cardiology', 'Maternity', 'Neurology'], dtype=object)]




```python
# Put dense array into a dataframe to get column names
encoded_d_df = pd.DataFrame(dense_array_d, columns=ohe.categories_[0] , dtype=int)

# Show
encoded_d_df.head()


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
      <th>Cardiology</th>
      <th>Maternity</th>
      <th>Neurology</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# no we will add the encoded columns into the main data frame
data_df3 = pd.concat((data_df2, encoded_d_df), axis=1)

# sanity check to see if columns have been added successfully
data_df3
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
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EnvironmentSatisfaction</th>
      <th>...</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>Non-Travel</th>
      <th>Travel_Frequently</th>
      <th>Travel_Rarely</th>
      <th>Cardiology</th>
      <th>Maternity</th>
      <th>Neurology</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1313919</td>
      <td>41</td>
      <td>0</td>
      <td>1102</td>
      <td>Cardiology</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200302</td>
      <td>49</td>
      <td>0</td>
      <td>279</td>
      <td>Maternity</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
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
      <td>Maternity</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
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
      <th>3</th>
      <td>1272912</td>
      <td>33</td>
      <td>0</td>
      <td>1392</td>
      <td>Maternity</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
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
      <td>Maternity</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>471</td>
      <td>Neurology</td>
      <td>24</td>
      <td>3</td>
      <td>Technical Degree</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1672</th>
      <td>1152327</td>
      <td>46</td>
      <td>0</td>
      <td>1125</td>
      <td>Cardiology</td>
      <td>10</td>
      <td>3</td>
      <td>Marketing</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>1812428</td>
      <td>20</td>
      <td>0</td>
      <td>959</td>
      <td>Maternity</td>
      <td>1</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
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
      <th>1674</th>
      <td>1812429</td>
      <td>39</td>
      <td>0</td>
      <td>466</td>
      <td>Neurology</td>
      <td>1</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>21</td>
      <td>6</td>
      <td>11</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1675</th>
      <td>1152329</td>
      <td>27</td>
      <td>0</td>
      <td>511</td>
      <td>Cardiology</td>
      <td>2</td>
      <td>2</td>
      <td>Medical</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1676 rows × 40 columns</p>
</div>




```python
# now drop the original Department column
data_df3 = data_df3.drop(['Department'], axis = 1)
```


```python
# sanity check to see if the Department column has been dropped successfully
data_df3
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
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>...</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>Non-Travel</th>
      <th>Travel_Frequently</th>
      <th>Travel_Rarely</th>
      <th>Cardiology</th>
      <th>Maternity</th>
      <th>Neurology</th>
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
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200302</td>
      <td>49</td>
      <td>0</td>
      <td>279</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
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
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
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
      <th>3</th>
      <td>1272912</td>
      <td>33</td>
      <td>0</td>
      <td>1392</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
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
      <td>Medical</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>471</td>
      <td>24</td>
      <td>3</td>
      <td>Technical Degree</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1672</th>
      <td>1152327</td>
      <td>46</td>
      <td>0</td>
      <td>1125</td>
      <td>10</td>
      <td>3</td>
      <td>Marketing</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>1812428</td>
      <td>20</td>
      <td>0</td>
      <td>959</td>
      <td>1</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
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
      <th>1674</th>
      <td>1812429</td>
      <td>39</td>
      <td>0</td>
      <td>466</td>
      <td>1</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>21</td>
      <td>6</td>
      <td>11</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1675</th>
      <td>1152329</td>
      <td>27</td>
      <td>0</td>
      <td>511</td>
      <td>2</td>
      <td>2</td>
      <td>Medical</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1676 rows × 39 columns</p>
</div>



**Department column has been successfully dropped**

## Now we will move on to the EducationField Column


```python
# check the classes in the EducationField
data_df3['EducationField'].unique()
```




    array(['Life Sciences', 'Other', 'Medical', 'Marketing',
           'Technical Degree', 'Human Resources'], dtype=object)




```python
# Instantiate the OneHotEncoder
ohe = OneHotEncoder()
# It expects a 2D array, so we first convert the column into a DataFrame
EducationField = pd.DataFrame(data_df3['EducationField'])

# Fit the OneHotEncoder to the subcategory column and transform
encoded_ed = ohe.fit_transform(EducationField)
encoded_ed
```




    <1676x6 sparse matrix of type '<class 'numpy.float64'>'
    	with 1676 stored elements in Compressed Sparse Row format>



**"_ed"** refers to EducationField


```python
# Convert from sparse matrix to dense
dense_array_ed = encoded_ed.toarray()
dense_array_ed
```




    array([[0., 1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0.],
           ...,
           [0., 1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.]])




```python
# check the categories
ohe.categories_
```




    [array(['Human Resources', 'Life Sciences', 'Marketing', 'Medical',
            'Other', 'Technical Degree'], dtype=object)]




```python
# Put dense array into a dataframe to get column names
encoded_ed_df = pd.DataFrame(dense_array_ed, columns=ohe.categories_[0] , dtype=int)

# Show
encoded_ed_df.head()


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
      <th>Human Resources</th>
      <th>Life Sciences</th>
      <th>Marketing</th>
      <th>Medical</th>
      <th>Other</th>
      <th>Technical Degree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# no we will add the encoded columns into the main data frame
data_df4 = pd.concat((data_df3, encoded_ed_df), axis=1)

# sanity check to see if columns have been added successfully
data_df4
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
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>...</th>
      <th>Travel_Rarely</th>
      <th>Cardiology</th>
      <th>Maternity</th>
      <th>Neurology</th>
      <th>Human Resources</th>
      <th>Life Sciences</th>
      <th>Marketing</th>
      <th>Medical</th>
      <th>Other</th>
      <th>Technical Degree</th>
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
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200302</td>
      <td>49</td>
      <td>0</td>
      <td>279</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1272912</td>
      <td>33</td>
      <td>0</td>
      <td>1392</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>Medical</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
      <td>1</td>
      <td>471</td>
      <td>24</td>
      <td>3</td>
      <td>Technical Degree</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
      <td>0</td>
      <td>1125</td>
      <td>10</td>
      <td>3</td>
      <td>Marketing</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>1812428</td>
      <td>20</td>
      <td>0</td>
      <td>959</td>
      <td>1</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1674</th>
      <td>1812429</td>
      <td>39</td>
      <td>0</td>
      <td>466</td>
      <td>1</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1675</th>
      <td>1152329</td>
      <td>27</td>
      <td>0</td>
      <td>511</td>
      <td>2</td>
      <td>2</td>
      <td>Medical</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1676 rows × 45 columns</p>
</div>




```python
# now drop the original Department column
data_df4 = data_df4.drop(['EducationField'], axis = 1)
```


```python
# sanity check to see if the EducationField columns has been dropped successfully
data_df4
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
      <th>Travel_Rarely</th>
      <th>Cardiology</th>
      <th>Maternity</th>
      <th>Neurology</th>
      <th>Human Resources</th>
      <th>Life Sciences</th>
      <th>Marketing</th>
      <th>Medical</th>
      <th>Other</th>
      <th>Technical Degree</th>
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
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
      <td>1</td>
      <td>471</td>
      <td>24</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>66</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
      <td>0</td>
      <td>1125</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>94</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>1812428</td>
      <td>20</td>
      <td>0</td>
      <td>959</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>83</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1674</th>
      <td>1812429</td>
      <td>39</td>
      <td>0</td>
      <td>466</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>65</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1675</th>
      <td>1152329</td>
      <td>27</td>
      <td>0</td>
      <td>511</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>89</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1676 rows × 44 columns</p>
</div>



## Now we will move on to the JobRole Column


```python
# check the classes in JobRole column 
data_df4['JobRole'].unique()
```




    array(['Nurse', 'Other', 'Therapist', 'Administrative', 'Admin'],
          dtype=object)




```python
# Instantiate the OneHotEncoder
ohe = OneHotEncoder()
# It expects a 2D array, so we first convert the column into a DataFrame
JobRole = pd.DataFrame(data_df4['JobRole'])

# Fit the OneHotEncoder to the subcategory column and transform
encoded_j = ohe.fit_transform(JobRole)
encoded_j
```




    <1676x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 1676 stored elements in Compressed Sparse Row format>



"**_j"** refers to JobRole


```python
# Convert from sparse matrix to dense
dense_array_j = encoded_j.toarray()
dense_array_j
```




    array([[0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 1., 0., 0.],
           ...,
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.],
           [0., 0., 1., 0., 0.]])




```python
#check categories
ohe.categories_
```




    [array(['Admin', 'Administrative', 'Nurse', 'Other', 'Therapist'],
           dtype=object)]




```python
# Put into a dataframe to get column names
encoded_j_df = pd.DataFrame(dense_array_j, columns=ohe.categories_[0] , dtype=int)

# Show
encoded_j_df.head()
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
      <th>Admin</th>
      <th>Administrative</th>
      <th>Nurse</th>
      <th>Other</th>
      <th>Therapist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# now we will add the encoded columns into the main data frame
data_df5 = pd.concat((data_df4, encoded_j_df), axis=1)

# sanity check to see if columns have been added successfully
data_df5
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
      <th>Life Sciences</th>
      <th>Marketing</th>
      <th>Medical</th>
      <th>Other</th>
      <th>Technical Degree</th>
      <th>Admin</th>
      <th>Administrative</th>
      <th>Nurse</th>
      <th>Other</th>
      <th>Therapist</th>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
      <td>1</td>
      <td>471</td>
      <td>24</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>66</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1672</th>
      <td>1152327</td>
      <td>46</td>
      <td>0</td>
      <td>1125</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>94</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>1812428</td>
      <td>20</td>
      <td>0</td>
      <td>959</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>83</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1674</th>
      <td>1812429</td>
      <td>39</td>
      <td>0</td>
      <td>466</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>65</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1675</th>
      <td>1152329</td>
      <td>27</td>
      <td>0</td>
      <td>511</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>89</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1676 rows × 49 columns</p>
</div>




```python
# drop the original Department column
data_df5 = data_df5.drop(['JobRole'], axis = 1)
```


```python
# sanity check to see if the JobRole column has been dropped successfully
data_df5.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1676 entries, 0 to 1675
    Data columns (total 48 columns):
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
     13  MaritalStatus             1676 non-null   object
     14  MonthlyIncome             1676 non-null   int64 
     15  MonthlyRate               1676 non-null   int64 
     16  NumCompaniesWorked        1676 non-null   int64 
     17  Over18                    1676 non-null   int64 
     18  OverTime                  1676 non-null   int64 
     19  PercentSalaryHike         1676 non-null   int64 
     20  PerformanceRating         1676 non-null   int64 
     21  RelationshipSatisfaction  1676 non-null   int64 
     22  StandardHours             1676 non-null   int64 
     23  Shift                     1676 non-null   int64 
     24  TotalWorkingYears         1676 non-null   int64 
     25  TrainingTimesLastYear     1676 non-null   int64 
     26  WorkLifeBalance           1676 non-null   int64 
     27  YearsAtCompany            1676 non-null   int64 
     28  YearsInCurrentRole        1676 non-null   int64 
     29  YearsSinceLastPromotion   1676 non-null   int64 
     30  YearsWithCurrManager      1676 non-null   int64 
     31  Non-Travel                1676 non-null   int64 
     32  Travel_Frequently         1676 non-null   int64 
     33  Travel_Rarely             1676 non-null   int64 
     34  Cardiology                1676 non-null   int64 
     35  Maternity                 1676 non-null   int64 
     36  Neurology                 1676 non-null   int64 
     37  Human Resources           1676 non-null   int64 
     38  Life Sciences             1676 non-null   int64 
     39  Marketing                 1676 non-null   int64 
     40  Medical                   1676 non-null   int64 
     41  Other                     1676 non-null   int64 
     42  Technical Degree          1676 non-null   int64 
     43  Admin                     1676 non-null   int64 
     44  Administrative            1676 non-null   int64 
     45  Nurse                     1676 non-null   int64 
     46  Other                     1676 non-null   int64 
     47  Therapist                 1676 non-null   int64 
    dtypes: int64(47), object(1)
    memory usage: 628.6+ KB


**JobRole has been successfully dropped** 

## Now we will move on to the MaritalStatus Column


```python
# check the classes in the MaritalStatus column 
data_df5['MaritalStatus'].unique()
```




    array(['Single', 'Married', 'Divorced'], dtype=object)




```python
# Instantiate the OneHotEncoder
ohe = OneHotEncoder()
# It expects a 2D array, so we first convert the column into a DataFrame
MaritalStatus = pd.DataFrame(data_df5['MaritalStatus'])

# Fit the OneHotEncoder to the subcategory column and transform
encoded_ms = ohe.fit_transform(MaritalStatus)
encoded_ms
```




    <1676x3 sparse matrix of type '<class 'numpy.float64'>'
    	with 1676 stored elements in Compressed Sparse Row format>



**"_ms"** refers to MaritaStatus


```python
# Convert from sparse matrix to dense
dense_array_ms = encoded_ms.toarray()
dense_array_ms
```




    array([[0., 0., 1.],
           [0., 1., 0.],
           [0., 0., 1.],
           ...,
           [0., 0., 1.],
           [0., 1., 0.],
           [0., 0., 1.]])




```python
#Check categories
ohe.categories_
```




    [array(['Divorced', 'Married', 'Single'], dtype=object)]




```python
# Put dense array into a dataframe to get column names
encoded_ms_df = pd.DataFrame(dense_array_ms, columns=ohe.categories_[0] , dtype=int)

# Show
encoded_ms_df.head()
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
      <th>Divorced</th>
      <th>Married</th>
      <th>Single</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# now we will add the encoded columns into the main data frame
data_df6 = pd.concat((data_df5, encoded_ms_df), axis=1)

# sanity check to see if columns have been added successfully
data_df6
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
      <th>Other</th>
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
      <td>1</td>
      <td>471</td>
      <td>24</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>66</td>
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
      <td>0</td>
      <td>1125</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
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
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>1812428</td>
      <td>20</td>
      <td>0</td>
      <td>959</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>83</td>
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
      <td>0</td>
      <td>466</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>65</td>
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
      <td>0</td>
      <td>511</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>89</td>
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
<p>1676 rows × 51 columns</p>
</div>




```python
# drop the original Department column
data_df6 = data_df6.drop(['MaritalStatus'], axis = 1)
```


```python
# sanity check see if the MaritalStatus column has been successfully dropped
data_df6.info()
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
     45  Other                     1676 non-null   int64
     46  Therapist                 1676 non-null   int64
     47  Divorced                  1676 non-null   int64
     48  Married                   1676 non-null   int64
     49  Single                    1676 non-null   int64
    dtypes: int64(50)
    memory usage: 654.8 KB


**MarirtalStatus column has been successfully dropped** 

### Now our data processing and EDA is complete. All columns have been converted to numerical columns. There are a total of 1676 rows and 50 columns. The data is now ready for train test splits and subsequently be fit into our desired models after feature selection 


```python
# rename clean dataframe to employee_final_df
employee_final_df = data_df6

# sanity check to see if the dataframe has been successfully renamed
employee_final_df
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
      <th>Other</th>
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
      <td>1</td>
      <td>471</td>
      <td>24</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>66</td>
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
      <td>0</td>
      <td>1125</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
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
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>1812428</td>
      <td>20</td>
      <td>0</td>
      <td>959</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>83</td>
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
      <td>0</td>
      <td>466</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>65</td>
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
      <td>0</td>
      <td>511</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>89</td>
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
<p>1676 rows × 50 columns</p>
</div>




```python
employee_final_df.info()
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
     45  Other                     1676 non-null   int64
     46  Therapist                 1676 non-null   int64
     47  Divorced                  1676 non-null   int64
     48  Married                   1676 non-null   int64
     49  Single                    1676 non-null   int64
    dtypes: int64(50)
    memory usage: 654.8 KB



```python
# save the dataframe to csv for later access in another notebook

employee_final_df.to_csv('employee_attrition.csv', index = False)
```


```python

```
