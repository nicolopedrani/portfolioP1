import streamlit as st

from utils import *

# For data visualization
import plotly.express as px

# Title and Text Block
st.title("Employee Turnover Analysis at Salifort Motors")

st.markdown("""

## **Pace: Plan**                       
### Understand the business scenario and problem
            
This project analyzes a dataset of employee information from **Salifort Motors** to identify factors that contribute to employee turnover. 
            The goal is to use this information to develop strategies to improve employee retention.  If we can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.  
              
**Objective**: The goals of this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company

This page will explore key findings from the data analysis, including:

* Data Exploration and Cleaning
* Feature Analysis
* Employee Characteristics by Turnover Status
* Correlations Between Features

""")

df0 = st.session_state['df0']

st.subheader("Data Dictionary")
"""
Variable  |Description |
-----|-----|
satisfaction_level|Employee-reported job satisfaction level [0&ndash;1]|
last_evaluation|Score of employee's last performance review [0&ndash;1]|
number_project|Number of projects employee contributes to|
average_monthly_hours|Average number of hours employee worked per month|
time_spend_company|How long the employee has been with the company (years)
Work_accident|Whether or not the employee experienced an accident while at work
left|Whether or not the employee left the company
promotion_last_5years|Whether or not the employee was promoted in the last 5 years
Department|The employee's department
salary|The employee's salary (U.S. dollars)
"""

"""
### Reflect on these questions as you complete the plan stage.

- Who are the stakeholders for this project?
- What are we trying to solve or accomplish?
- What are our initial observations when we explore the data?
- Do we have any ethical considerations in this stage?
"""

# Data Exploration Section
st.header("Explorative Data Analysis (EDA)")
st.markdown('''
- Understand variables
- Clean our dataset (missing data, redundant data, outliers)
            
As a data cleaning step, rename the columns as needed. Standardize the column names so that they are all in _snake\_case_, correct any column names that are misspelled, and make column names more concise as needed.
            
            ''')
st.code(
    """
    df = df.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})
    """
)

st.code('''df.head()''')
st.write(df0.head())

st.subheader("Descriptive Statistics")

# Display sample data (if data loaded locally, use df.describe())
st.code('''df.info()''')
# Redirect output of df.info() to a StringIO object
info_str = """
RangeIndex: 14999 entries, 0 to 14998
Data columns (total 10 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   satisfaction_level     14999 non-null  float64
 1   last_evaluation        14999 non-null  float64
 2   number_project         14999 non-null  int64  
 3   average_monthly_hours  14999 non-null  int64  
 4   tenure                 14999 non-null  int64  
 5   work_accident          14999 non-null  int64  
 6   left                   14999 non-null  int64  
 7   promotion_last_5years  14999 non-null  int64  
 8   department             14999 non-null  object 
 9   salary                 14999 non-null  object 
dtypes: float64(2), int64(6), object(2)
"""
st.code(info_str)
st.code('''df.describe(include='all')''')
st.dataframe(df0.describe(include='all'))
'''**check for missing values**'''
st.code('''df0.isna().sum()''')
st.write(df0.isna().sum())
'''**Check for any duplicate entries in the data**'''
st.code('''df0.duplicated().sum()''')
st.write(df0.duplicated().sum())
'''3,008 rows contain duplicates. That is 20% of the data.  
How likely is it that these are legitimate entries? In other words, how plausible is it that two employees self-reported the exact same response for every column?
With several continuous variables across 10 columns, it seems very unlikely that these observations are legitimate. You can proceed by dropping them.
'''
df1 = df0.drop_duplicates(keep='first')
'''**Check outliers**  
we will focus on the `tenure` variable'''

fig = px.box(df1, y='tenure')

# Update layout for better readability (optional)
fig.update_layout(
    xaxis_title="Tenure",
    yaxis_title="Value",
    xaxis_tickfont_size=12,
    yaxis_tickfont_size=12,
    title="Boxplot to detect outliers for tenure",
)

# Display the plot using Streamlit
st.plotly_chart(fig)
'''
The boxplot above shows that there are outliers in the tenure variable.  
It would be helpful to investigate how many rows in the data contain outliers in the tenure column.  

We calculate the spread, interquartile range IQR, by finding the difference between the 25th and 75th percentile. 
Then, we set upper and lower limits based on 1.5 times the IQR around the quartiles. 
Finally, we count data points in tenure that fall outside these limits and labels them outliers.  
'''

# Determine the number of rows containing outliers
# Compute the 25th percentile value in `tenure`
percentile25 = df1['tenure'].quantile(0.25)
# Compute the 75th percentile value in `tenure`
percentile75 = df1['tenure'].quantile(0.75)
# Compute the interquartile range in `tenure`
iqr = percentile75 - percentile25
# Define the upper limit and lower limit for non-outlier values in `tenure`
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

if 'upper_limit' not in st.session_state:
        st.session_state['upper_limit'] = upper_limit
if 'lower_limit' not in st.session_state:
        st.session_state['lower_limit'] = lower_limit

st.write(f"Lower limit: {lower_limit} | Upper limit: {upper_limit}")
# Identify subset of data containing outliers in `tenure`
outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]
# Count how many rows in the data contain outliers in `tenure`
st.write(f"Number of rows in the data containing outliers in `tenure`: {len(outliers)}")
'''
Certain types of models are more sensitive to outliers than others. When we get to the stage of building our model, we must consider whether to remove these outliers based on the type of model we decide to use.
'''


