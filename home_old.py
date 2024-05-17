import streamlit as st

# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# import custom functions
from utils import *

load_dataset()

# Title and Text Block
st.title("Employee Turnover Analysis at Salifort Motors")

st.page_link("home🏠.py", label="Home", icon="🏠")
st.page_link("pages/1_Plan💭.py", label="Plan", icon="💭")
st.page_link("pages/2_Analyze📊.py", label="Analyze", icon="📊")
st.page_link("pages/3_Construct📈.py", label="Construct", icon="📈")
st.page_link("pages/4_Execute🗒️.py", label="Execute", icon="🗒️")
st.page_link("pages/5_ExecutiveSummary📝.py", label="ExecutiveSummary", icon="📝")

# st.markdown("""

# ## **Pace: Plan**                       
# ### Understand the business scenario and problem
            
# This project analyzes a dataset of employee information from Salifort Motors to identify factors that contribute to employee turnover. 
#             The goal is to use this information to develop strategies to improve employee retention.  If you can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.  
              
# **Objective**: Your goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company

# This page will explore key findings from the data analysis, including:

# * Data Exploration and Cleaning
# * Feature Analysis
# # * Employee Characteristics by Turnover Status
# # * Correlations Between Features

# # """)

# load_dataset()
# df0 = st.session_state['df0'].copy()

# # # Data Loading (if data is stored locally)
# # df0 = pd.read_csv('HR_capstone_dataset.csv')
# # # Rename columns as needed
# # df0 = df0.rename(columns={'Work_accident': 'work_accident',
# #                           'average_montly_hours': 'average_monthly_hours',
# #                           'time_spend_company': 'tenure',
# #                           'Department': 'department'})

# st.subheader("Data Dictionary")
# """
# Variable  |Description |
# -----|-----|
# satisfaction_level|Employee-reported job satisfaction level [0&ndash;1]|
# last_evaluation|Score of employee's last performance review [0&ndash;1]|
# number_project|Number of projects employee contributes to|
# average_monthly_hours|Average number of hours employee worked per month|
# time_spend_company|How long the employee has been with the company (years)
# Work_accident|Whether or not the employee experienced an accident while at work
# left|Whether or not the employee left the company
# promotion_last_5years|Whether or not the employee was promoted in the last 5 years
# Department|The employee's department
# salary|The employee's salary (U.S. dollars)
# """

# """
# ### Reflect on these questions as you complete the plan stage.

# *  Who are your stakeholders for this project?
# - What are you trying to solve or accomplish?
# - What are your initial observations when you explore the data?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# """

# # Data Exploration Section
# st.header("Explorative Data Analysis (EDA)")
# st.markdown('''
# - Understand your variables
# - Clean your dataset (missing data, redundant data, outliers)
            
# As a data cleaning step, rename the columns as needed. Standardize the column names so that they are all in _snake\_case_, correct any column names that are misspelled, and make column names more concise as needed.
            
#             ''')
# st.code(
#     """
#     df = df.rename(columns={'Work_accident': 'work_accident',
#                           'average_montly_hours': 'average_monthly_hours',
#                           'time_spend_company': 'tenure',
#                           'Department': 'department'})
#     """
# )

# st.code('''df.head()''')
# st.write(df0.head())

# st.subheader("Descriptive Statistics")

# # Display sample data (if data loaded locally, use df.describe())
# st.code('''df.info()''')
# # Redirect output of df.info() to a StringIO object
# info_str = """
# RangeIndex: 14999 entries, 0 to 14998
# Data columns (total 10 columns):
#  #   Column                 Non-Null Count  Dtype  
# ---  ------                 --------------  -----  
#  0   satisfaction_level     14999 non-null  float64
#  1   last_evaluation        14999 non-null  float64
#  2   number_project         14999 non-null  int64  
#  3   average_monthly_hours  14999 non-null  int64  
#  4   tenure                 14999 non-null  int64  
#  5   work_accident          14999 non-null  int64  
#  6   left                   14999 non-null  int64  
#  7   promotion_last_5years  14999 non-null  int64  
#  8   department             14999 non-null  object 
#  9   salary                 14999 non-null  object 
# dtypes: float64(2), int64(6), object(2)
# """
# st.code(info_str)
# st.code('''df.describe(include='all')''')
# st.dataframe(df0.describe(include='all'))
# '''**check for missing values**'''
# st.code('''df0.isna().sum()''')
# st.write(df0.isna().sum())
# '''**Check for any duplicate entries in the data**'''
# st.code('''df0.duplicated().sum()''')
# st.write(df0.duplicated().sum())
# '''3,008 rows contain duplicates. That is 20% of the data.  
# How likely is it that these are legitimate entries? In other words, how plausible is it that two employees self-reported the exact same response for every column?  
# With several continuous variables across 10 columns, it seems very unlikely that these observations are legitimate. You can proceed by dropping them.
# '''
# df1 = df0.drop_duplicates(keep='first')
# '''**Check outliers**  
# we will focus on the `tenure` variable'''

# fig = px.box(df1, y='tenure')

# # Update layout for better readability (optional)
# fig.update_layout(
#     xaxis_title="Tenure",
#     yaxis_title="Value",
#     xaxis_tickfont_size=12,
#     yaxis_tickfont_size=12,
#     title="Boxplot to detect outliers for tenure",
# )

# # Display the plot using Streamlit
# st.plotly_chart(fig)
# '''
# The boxplot above shows that there are outliers in the tenure variable.  
# It would be helpful to investigate how many rows in the data contain outliers in the tenure column.  

# We calculate the spread, interquartile range IQR, by finding the difference between the 25th and 75th percentile. 
# Then, we set upper and lower limits based on 1.5 times the IQR around the quartiles. 
# Finally, we count data points in tenure that fall outside these limits and labels them outliers.  
# '''

# # Determine the number of rows containing outliers
# # Compute the 25th percentile value in `tenure`
# percentile25 = df1['tenure'].quantile(0.25)
# # Compute the 75th percentile value in `tenure`
# percentile75 = df1['tenure'].quantile(0.75)
# # Compute the interquartile range in `tenure`
# iqr = percentile75 - percentile25
# # Define the upper limit and lower limit for non-outlier values in `tenure`
# upper_limit = percentile75 + 1.5 * iqr
# lower_limit = percentile25 - 1.5 * iqr
# st.write(f"Lower limit: {lower_limit} | Upper limit: {upper_limit}")
# # Identify subset of data containing outliers in `tenure`
# outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]
# # Count how many rows in the data contain outliers in `tenure`
# st.write(f"Number of rows in the data containing outliers in `tenure`: {len(outliers)}")
# '''
# Certain types of models are more sensitive to outliers than others. When you get to the stage of building your model, consider whether to remove these outliers based on the type of model you decide to use.
# '''

# """
# # pAce: Analyze Stage
# - Perform EDA (analyze relationships between variables)
# """

# """
# 💭
# ### Reflect on these questions as you complete the analyze stage.

# - What did you observe about the relationships between variables?
# - What do you observe about the distributions in the data?
# - What transformations did you make with your data? Why did you chose to make those decisions?
# - What are some purposes of EDA before constructing a predictive model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# """

# st.subheader('Analyze Relationships between Variables')
# '''Begin by understanding how many employees left and what percentage of all employees this figure represents'''

# color_stay = 'skyblue'
# color_left = 'salmon'

# # Create data frame for the bar chart
# chart_data = df1['left'].value_counts(normalize=True).to_frame().T
# chart_data.columns = ['Stayed', 'Left']

# # Create bar chart
# fig3 = go.Figure()
# fig3.add_trace(go.Bar(
#     x=chart_data.columns,
#     y=chart_data.values.flatten(),
#     marker_color=[color_stay, color_left]
# ))

# # Update layout
# fig3.update_layout(
#     title='Employee Status',
#     xaxis=dict(title='Status'),
#     yaxis=dict(title='Percentage'),
#     legend_title='Status'
# )

# st.plotly_chart(fig3)

# """
# Now, examine variables that you're interested in, and create plots to visualize relationships between variables in the data.

# You could start by creating a stacked boxplot showing `average_monthly_hours` distributions for `number_project`, comparing the distributions of employees who stayed versus those who left.  
# Box plots are very useful in visualizing distributions within data, but they can be deceiving without the context of how big the sample sizes that they represent are. So, you could also plot a stacked histogram to visualize the distribution of `number_project` for those who stayed and those who left.
# """

# # Define color scheme and legend titles
# color_stay = 'skyblue'
# color_left = 'salmon'
# legend_titles = {'Stay': 'Stay', 'Left': 'Left'}

# # Create boxplot showing `average_monthly_hours` distributions for `number_project`, comparing employees who stayed versus those who left
# fig = go.Figure()

# # Adding box for employees who stayed
# fig.add_trace(go.Box(
#     x=df1[df1['left'] == 0]['number_project'],
#     y=df1[df1['left'] == 0]['average_monthly_hours'],
#     name='Stay',
#     marker_color=color_stay,
#     boxmean=True,
#     orientation='v'
# ))

# # Adding box for employees who left
# fig.add_trace(go.Box(
#     x=df1[df1['left'] == 1]['number_project'],
#     y=df1[df1['left'] == 1]['average_monthly_hours'],
#     name='Left',
#     marker_color=color_left,
#     boxmean=True,
#     orientation='v'
# ))

# fig.update_layout(
#     title='Monthly hours by number of projects',
#     xaxis=dict(title='Number of Projects'),
#     yaxis=dict(title='Average Monthly Hours')
# )

# # Create histogram showing distribution of `number_project`, comparing employees who stayed versus those who left
# fig2 = px.histogram(df1, x="number_project", color="left", barmode="group", color_discrete_map={0: color_stay, 1: color_left})

# fig2.update_layout(
#     title='Number of projects histogram',
#     xaxis=dict(title='Number of Projects'),
#     yaxis=dict(title='Count'),
#     legend_title=legend_titles['Left']
# )

# # Display plots in Streamlit
# st.plotly_chart(fig)
# st.plotly_chart(fig2)

# """
# It might be natural that people who work on more projects would also work longer hours. This appears to be the case here, with the mean hours of each group (stayed and left) increasing with number of projects worked. However, a few things stand out from this plot.

# 1. There are two groups of employees who left the company: (A) those who worked considerably less than their peers with the same number of projects, and (B) those who worked much more. Of those in group A, it's possible that they were fired. It's also possible that this group includes employees who had already given their notice and were assigned fewer hours because they were already on their way out the door. For those in group B, it's reasonable to infer that they probably quit. The folks in group B likely contributed a lot to the projects they worked in; they might have been the largest contributors to their projects.

# 2. Everyone with seven projects left the company, and the interquartile ranges of this group and those who left with six projects was ~255&ndash;295 hours/month&mdash;much more than any other group.

# 3. The optimal number of projects for employees to work on seems to be 3&ndash;4. The ratio of left/stayed is very small for these cohorts.

# 4. If you assume a work week of 40 hours and two weeks of vacation per year, then the average number of working hours per month of employees working Monday&ndash;Friday `= 50 weeks * 40 hours per week / 12 months = 166.67 hours per month`. This means that, aside from the employees who worked on two projects, every group&mdash;even those who didn't leave the company&mdash;worked considerably more hours than this. It seems that employees here are overworked.
  
# 5. all employees with 7 projects did leave.
# """

# # Define colors
# color_stay = 'skyblue'
# color_left = 'salmon'

# # Create scatter plot
# fig4 = go.Figure()

# # Add scatter plot traces
# fig4.add_trace(go.Scatter(
#     x=df1[df1['left'] == 0]['average_monthly_hours'],
#     y=df1[df1['left'] == 0]['satisfaction_level'],
#     mode='markers',
#     marker=dict(color=color_stay),
#     name='Stay'
# ))

# fig4.add_trace(go.Scatter(
#     x=df1[df1['left'] == 1]['average_monthly_hours'],
#     y=df1[df1['left'] == 1]['satisfaction_level'],
#     mode='markers',
#     marker=dict(color=color_left),
#     name='Left'
# ))

# # Add vertical line
# fig4.add_shape(
#     type='line',
#     x0=166.67,
#     y0=0,
#     x1=166.67,
#     y1=1,
#     line=dict(color='red', dash='dash'),
#     name='166.67 hrs./mo.'
# )

# # Update layout
# fig4.update_layout(
#     title='Monthly hours by Satisfaction Level',
#     xaxis=dict(title='Average Monthly Hours'),
#     yaxis=dict(title='Satisfaction Level'),
#     legend_title='Status'
# )

# st.plotly_chart(fig4)

# """
# The scatterplot above shows that there was a sizeable group of employees who worked ~240&ndash;315 hours per month. 315 hours per month is over 75 hours per week for a whole year. It's likely this is related to their satisfaction levels being close to zero.

# The plot also shows another group of people who left, those who had more normal working hours. Even so, their satisfaction was only around 0.4. It's difficult to speculate about why they might have left. It's possible they felt pressured to work more, considering so many of their peers worked more. And that pressure could have lowered their satisfaction levels.

# Finally, there is a group who worked ~210&ndash;280 hours per month, and they had satisfaction levels ranging ~0.7&ndash;0.9.

# Note the strange shape of the distributions here. This is indicative of data manipulation or synthetic data.
# """

# # Define color scheme and legend titles
# color_stay = 'skyblue'
# color_left = 'salmon'
# legend_titles = {'Stay': 'Stay', 'Left': 'Left'}

# fig5 = go.Figure()

# # Adding box for employees who stayed
# fig5.add_trace(go.Box(
#     x=df1[df1['left'] == 0]['tenure'],
#     y=df1[df1['left'] == 0]['satisfaction_level'],
#     name='Stay',
#     marker_color=color_stay,
#     boxmean=True,
#     orientation='v'
# ))

# # Adding box for employees who left
# fig5.add_trace(go.Box(
#     x=df1[df1['left'] == 1]['tenure'],
#     y=df1[df1['left'] == 1]['satisfaction_level'],
#     name='Left',
#     marker_color=color_left,
#     boxmean=True,
#     orientation='v'
# ))

# fig5.update_layout(
#     title='Satisfaction Level by Tenure',
#     xaxis=dict(title='Tenure'),
#     yaxis=dict(title='Satisfaction Level')
# )

# # Create histogram showing distribution of `tenure`, comparing employees who stayed versus those who left
# fig6 = px.histogram(df1, x="tenure", color="left", barmode="group", color_discrete_map={0: color_stay, 1: color_left})

# fig6.update_layout(
#     title='Tenure histogram',
#     xaxis=dict(title='Tenure'),
#     yaxis=dict(title='Count'),
#     legend_title=legend_titles['Left']
# )

# # Display plots in Streamlit
# st.plotly_chart(fig5)
# st.plotly_chart(fig6)

# """
# There are many observations you could make from this plot.
# - Employees who left fall into two general categories: dissatisfied employees with shorter tenures and very satisfied employees with medium-length tenures.
# - Four-year employees who left seem to have an unusually low satisfaction level. It's worth investigating changes to company policy that might have affected people specifically at the four-year mark, if possible.
# - The longest-tenured employees didn't leave. Their satisfaction levels aligned with those of newer employees who stayed.
# - The histogram shows that there are relatively few longer-tenured employees. It's possible that they're the higher-ranking, higher-paid employees.
# """
# """
# As the next step in analyzing the data, you could calculate the mean and median satisfaction scores of employees who left and those who didn't.
# """

# # Group by 'left' column and calculate mean and median of 'satisfaction_level'
# grouped_df = df1.groupby(['left'])['satisfaction_level'].agg([np.mean, np.median])

# # Plotting
# fig = go.Figure()

# # Add mean trace for left = 0
# fig.add_trace(go.Bar(
#     x=['Left 0'],
#     y=[grouped_df.loc[0, 'mean']],
#     name='Mean',
#     marker_color='rgba(52, 152, 219, 0.5)'  # Blue color with alpha (transparency)
# ))

# # Add median trace for left = 0
# fig.add_trace(go.Bar(
#     x=['Left 0'],
#     y=[grouped_df.loc[0, 'median']],
#     name='Median',
#     marker_color='rgba(41, 128, 185, 1)'  # Blue color with alpha (transparency)
# ))

# # Add mean trace for left = 1
# fig.add_trace(go.Bar(
#     x=['Left 1'],
#     y=[grouped_df.loc[1, 'mean']],
#     name='Mean',
#     marker_color='rgba(231, 76, 60, 0.5)'  # Red color with alpha (transparency)
# ))

# # Add median trace for left = 1
# fig.add_trace(go.Bar(
#     x=['Left 1'],
#     y=[grouped_df.loc[1, 'median']],
#     name='Median',
#     marker_color='rgba(192, 57, 43, 1)'  # Red color with alpha (transparency)
# ))

# # Update layout
# fig.update_layout(
#     title='Satisfaction Level by Left Status',
#     xaxis=dict(title='Left Status'),
#     yaxis=dict(title='Satisfaction Level'),
#     barmode='group'
# )

# # Show the plot
# st.plotly_chart(fig)

# """
# As expected, the mean and median satisfaction scores of employees who left are lower than those of employees who stayed. Interestingly, among employees who stayed, the mean satisfaction score appears to be slightly below the median score. This indicates that satisfaction levels among those who stayed might be skewed to the left.
# """

# """
# Next, you could examine salary levels for different tenures.
# """

# # Define short-tenured employees
# tenure_short = df1[df1['tenure'] < 7]

# # Define long-tenured employees
# tenure_long = df1[df1['tenure'] > 6]

# # Short-tenured histogram
# fig1 = go.Figure()

# for salary_group in ['low', 'medium', 'high']:
#     fig1.add_trace(go.Histogram(
#         x=tenure_short[tenure_short['salary'] == salary_group]['tenure'],
#         name=salary_group,
#         marker_color={'low': 'blue', 'medium': 'green', 'high': 'red'}[salary_group],
#         opacity=0.7,
#         # histnorm='percent',
#         # nbinsx=20,
#         # showlegend=True if salary_group == 'low' else False
#     ))

# fig1.update_layout(
#     title_text='Salary histogram by tenure: short-tenured people',
#     xaxis_title='Tenure',
#     yaxis_title='Percentage',
#     bargap=0.1
# )

# # Long-tenured histogram
# fig2 = go.Figure()

# for salary_group in ['low', 'medium', 'high']:
#     fig2.add_trace(go.Histogram(
#         x=tenure_long[tenure_long['salary'] == salary_group]['tenure'],
#         name=salary_group,
#         marker_color={'low': 'blue', 'medium': 'green', 'high': 'red'}[salary_group],
#         opacity=0.7,
#         # histnorm='percent',
#         # nbinsx=20,
#         # showlegend=True if salary_group == 'low' else False
#     ))

# fig2.update_layout(
#     title_text='Salary histogram by tenure: long-tenured people',
#     xaxis_title='Tenure',
#     yaxis_title='Percentage',
#     bargap=0.1
# )

# # Show the plots
# st.plotly_chart(fig1)
# st.plotly_chart(fig2)

# """
# The plots above show that long-tenured employees were not disproportionately comprised of higher-paid employees.

# Next, you could explore whether there's a correlation between working long hours and receiving high evaluation scores. You could create a scatterplot of `average_monthly_hours` versus `last_evaluation`.
# """

# # Create scatter plot
# fig4 = go.Figure()

# # Add scatter plot traces
# fig4.add_trace(go.Scatter(
#     x=df1[df1['left'] == 0]['average_monthly_hours'],
#     y=df1[df1['left'] == 0]['last_evaluation'],
#     mode='markers',
#     marker=dict(color=color_stay),
#     name='Stay'
# ))

# fig4.add_trace(go.Scatter(
#     x=df1[df1['left'] == 1]['average_monthly_hours'],
#     y=df1[df1['left'] == 1]['last_evaluation'],
#     mode='markers',
#     marker=dict(color=color_left),
#     name='Left'
# ))

# # Add vertical line
# fig4.add_shape(
#     type='line',
#     x0=166.67,
#     y0=0.35,
#     x1=166.67,
#     y1=1,
#     line=dict(color='red', dash='dash'),
#     name='166.67 hrs./mo.'
# )

# # Update layout
# fig4.update_layout(
#     title='Monthly hours by last evaluation score',
#     xaxis=dict(title='Average Monthly Hours'),
#     yaxis=dict(title='Last Evaluation'),
#     legend_title='Status'
# )

# st.plotly_chart(fig4)

# """
# The following observations can be made from the scatterplot above:
# - The scatterplot indicates two groups of employees who left: overworked employees who performed very well and employees who worked slightly under the nominal monthly average of 166.67 hours with lower evaluation scores.
# - There seems to be a correlation between hours worked and evaluation score.
# - There isn't a high percentage of employees in the upper left quadrant of this plot; but working long hours doesn't guarantee a good evaluation score.
# - Most of the employees in this company work well over 167 hours per month.

# Next, you could examine whether employees who worked very long hours were promoted in the last five years.
# """

# # Create scatter plot
# fig4 = go.Figure()

# fig4.add_trace(go.Scatter(
#     x=df1[df1['left'] == 1]['average_monthly_hours'],
#     y=df1[df1['left'] == 1]['promotion_last_5years'],
#     mode='markers',
#     marker=dict(color=color_left),
#     name='Left'
# ))

# # Add scatter plot traces
# fig4.add_trace(go.Scatter(
#     x=df1[df1['left'] == 0]['average_monthly_hours'],
#     y=df1[df1['left'] == 0]['promotion_last_5years'],
#     mode='markers',
#     marker=dict(color=color_stay),
#     name='Stay'
# ))

# # Add vertical line
# fig4.add_shape(
#     type='line',
#     x0=166.67,
#     y0=0,
#     x1=166.67,
#     y1=1,
#     line=dict(color='red', dash='dash'),
#     name='166.67 hrs./mo.'
# )

# # Update layout
# fig4.update_layout(
#     title='Monthly hours by Promotion Last 5 Years',
#     xaxis=dict(title='Average Monthly Hours'),
#     yaxis=dict(title='Promotion Last 5 Years'),
#     legend_title='Status'
# )

# st.plotly_chart(fig4)

# """
# The plot above shows the following:
# - very few employees who were promoted in the last five years left
# - very few employees who worked the most hours were promoted
# - all of the employees who left were working the longest hours  

# Next, you could inspect how the employees who left are distributed across departments.
# """

# fig = go.Figure()

# for left_value in [0, 1]:
#     df_left = df1[df1['left'] == left_value]
#     fig.add_trace(go.Histogram(
#         x=df_left['department'],
#         name=f'Left: {left_value}',
#         marker_color=color_stay if left_value == 0 else color_left,
#         # opacity=0.7,
#         # histnorm='percent',
#         showlegend=True
#     ))

# fig.update_layout(
#     title='Counts of stayed/left by department',
#     xaxis=dict(title='Department'),
#     yaxis=dict(title='Percentage'),
#     bargap=0.1,
#     xaxis_tickangle=-45
# )

# st.plotly_chart(fig)

# """
# There doesn't seem to be any department that differs significantly in its proportion of employees who left to those who stayed.

# Lastly, you could check for strong correlations between variables in the data.
# """

# # Calculate correlation matrix
# correlation_matrix = df1[["satisfaction_level","last_evaluation","number_project","average_monthly_hours","tenure","work_accident","left","promotion_last_5years"]].corr()

# # Define colorscale
# colorscale = [[0, 'navy'], [0.5, 'lightsteelblue'], [1.0, 'firebrick']]

# # Create heatmap
# fig = go.Figure(data=go.Heatmap(
#         z=correlation_matrix.values,
#         x=correlation_matrix.columns,
#         y=correlation_matrix.index,
#         colorscale=colorscale,
#         colorbar=dict(title='Correlation', tickvals=[-1, 0, 1]),
#         zmin=-1,
#         zmax=1,
#         hoverongaps = False
# ))

# # Update layout
# fig.update_layout(
#     title='Correlation Heatmap',
#     title_x=0.5,
#     xaxis_title='Features',
#     yaxis_title='Features',
#     height=600,  # Increase the height
#     width=800    # Increase the width
# )

# st.plotly_chart(fig)

# """
# The correlation heatmap confirms that the number of projects, monthly hours, and evaluation scores all have some positive correlation with each other, and whether an employee leaves is negatively correlated with their satisfaction level.

# ### Insights

# It appears that employees are leaving the company as a result of poor management. Leaving is tied to longer working hours, many projects, and generally lower satisfaction levels. It can be ungratifying to work long hours and not receive promotions or good evaluation scores. There's a sizeable group of employees at this company who are probably burned out. It also appears that if an employee has spent more than six years at the company, they tend not to leave.

# """

# """
# # paCe: Construct Stage 🔎 💭
# - Determine which models are most appropriate
# - Construct the model
# - Confirm model assumptions
# - Evaluate model results to determine how well your model fits the data
# """

# """

# ## Recall model assumptions

# **Logistic Regression model assumptions**
# - Outcome variable is categorical
# - Observations are independent of each other
# - No severe multicollinearity among X variables
# - No extreme outliers
# - Linear relationship between each X variable and the logit of the outcome variable
# - Sufficiently large sample size

# 💭
# ### Reflect on these questions as you complete the constructing stage.

# - Do you notice anything odd?
# - Which independent variables did you choose for the model and why?
# - Are each of the assumptions met?
# - How well does your model fit the data?
# - Can you improve it? Is there anything you would change about the model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?

# ## Model Building, Results and Evaluation
# - Fit a model that predicts the outcome variable using two or more independent variables
# - Check model assumptions
# - Evaluate the model

# ### Identify the type of prediction task.
# Your goal is to predict whether an employee leaves the company, which is a categorical outcome variable. So this task involves classification. More specifically, this involves binary classification, since the outcome variable `left` can be either 1 (indicating employee left) or 0 (indicating employee didn't leave).

# ### Identify the types of models most appropriate for this task.
# Since the variable you want to predict (whether an employee leaves the company) is categorical, you could either build a Logistic Regression model, or a Tree-based Machine Learning model.

# So you could proceed with one of the two following approaches. Or, if you'd like, you could implement both and determine how they compare.

# ### Modeling Approach A: Logistic Regression Model

# This approach covers implementation of Logistic Regression.

# #### Logistic regression
# Note that binomial logistic regression suits the task because it involves binary classification.

# Before splitting the data, encode the non-numeric variables. There are two: `department` and `salary`.

# `department` is a categorical variable, which means you can dummy it for modeling.

# `salary` is categorical too, but it's ordinal. There's a hierarchy to the categories, so it's better not to dummy this column, but rather to convert the levels to numbers, 0&ndash;2.

# """

# st.code("""
# # Copy the dataframe
# df_enc = df1.copy()

# # Encode the `salary` column as an ordinal numeric category
# df_enc['salary'] = (
#     df_enc['salary'].astype('category')
#     .cat.set_categories(['low', 'medium', 'high'])
#     .cat.codes
# )

# # Dummy encode the `department` column
# df_enc = pd.get_dummies(df_enc, drop_first=False)
#         """)

# # Copy the dataframe
# df_enc = df1.copy()

# # Encode the `salary` column as an ordinal numeric category
# df_enc['salary'] = (
#     df_enc['salary'].astype('category')
#     .cat.set_categories(['low', 'medium', 'high'])
#     .cat.codes
# )

# # Dummy encode the `department` column
# df_enc = pd.get_dummies(df_enc, drop_first=False)

# """
# Create a heatmap to visualize how correlated variables are. Consider which variables you're interested in examining correlations between.
# """

# # Calculate correlation matrix
# correlation_matrix = df1[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure']].corr()

# # Define colorscale
# colorscale = [[0, 'navy'], [0.5, 'lightsteelblue'], [1.0, 'firebrick']]

# # Create heatmap
# fig = go.Figure(data=go.Heatmap(
#         z=correlation_matrix.values,
#         x=correlation_matrix.columns,
#         y=correlation_matrix.index,
#         colorscale=colorscale,
#         colorbar=dict(title='Correlation', tickvals=[-1, 0, 1]),
#         zmin=-1,
#         zmax=1,
#         hoverongaps = False
# ))

# # Update layout
# fig.update_layout(
#     title='Correlation Heatmap',
#     title_x=0.5,
#     xaxis_title='Features',
#     yaxis_title='Features',
#     height=600,  # Increase the height
#     width=800    # Increase the width
# )

# st.plotly_chart(fig)

# """
# Since logistic regression is quite sensitive to outliers, it would be a good idea at this stage to remove the outliers in the `tenure` column that were identified earlier.
# """

# st.code('''
# df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <= upper_limit)]
#         ''')

# df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <= upper_limit)]

# """
# Isolate the outcome variable, which is the variable you want your model to predict.  
# Select the features you want to use in your model. Consider which variables will help you predict the outcome variable, `left`.  
# Split the data into training set and testing set. Don't forget to stratify based on the values in `y`, since the classes are unbalanced.  
# Construct a logistic regression model and fit it to the training dataset.  
# Test the logistic regression model: use the model to make predictions on the test set.  
# Create a confusion matrix to visualize the results of the logistic regression model.
# """

# # Isolate the outcome variable
# y = df_logreg['left']
# # Select the features you want to use in your model
# X = df_logreg.drop('left', axis=1)

# st.code("""
# # Split the data into training set and testing set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
# # Construct a logistic regression model and fit it to the training dataset
# log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
# # Use the logistic regression model to get predictions on the test set
# y_pred = log_clf.predict(X_test)   
#         """)

# # Split the data into training set and testing set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
# # Construct a logistic regression model and fit it to the training dataset
# log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
# # Use the logistic regression model to get predictions on the test set
# y_pred = log_clf.predict(X_test) 

# # Compute values for confusion matrix
# log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# # Create display of confusion matrix
# log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm,
#                                   display_labels=log_clf.classes_)

# # Plot confusion matrix
# fig, ax = plt.subplots(figsize=(8, 6))
# log_disp.plot(ax=ax, values_format='')
# ax.set_title('Confusion Matrix | Logistic Regression')
# ax.set_xlabel('Predicted Label')
# ax.set_ylabel('True Label')

# st.pyplot(fig, use_container_width=False, dpi=100)

# """
# The upper-left quadrant displays the number of true negatives.
# The upper-right quadrant displays the number of false positives.
# The bottom-left quadrant displays the number of false negatives.
# The bottom-right quadrant displays the number of true positives.

# - True negatives: The number of people who did not leave that the model accurately predicted did not leave.
# - False positives: The number of people who did not leave the model inaccurately predicted as leaving.
# - False negatives: The number of people who left that the model inaccurately predicted did not leave
# - True positives: The number of people who left the model accurately predicted as leaving

# A perfect model would yield all true negatives and true positives, and no false negatives or false positives.
# """

# """
# Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the logistic regression model.  
# Check the class balance in the data. In other words, check the value counts in the `left` column. Since this is a binary classification task, the class balance informs the way you interpret accuracy metrics.
# """



# color_stay = 'skyblue'
# color_left = 'salmon'

# # Create data frame for the bar chart
# chart_data = df_logreg['left'].value_counts(normalize=True).to_frame().T
# chart_data.columns = ['Stayed', 'Left']

# # Create bar chart
# fig3 = go.Figure()
# fig3.add_trace(go.Bar(
#     x=chart_data.columns,
#     y=chart_data.values.flatten(),
#     marker_color=[color_stay, color_left]
# ))

# # Update layout
# fig3.update_layout(
#     title='Employee Status',
#     xaxis=dict(title='Status'),
#     yaxis=dict(title='Percentage'),
#     legend_title='Status'
# )

# st.plotly_chart(fig3)

# """
# There is an approximately 83%-17% split. So the data is not perfectly balanced, but it is not too imbalanced. If it was more severely imbalanced, you might want to resample the data to make it more balanced. In this case, you can use this data without modifying the class balance and continue evaluating the model.
# """

# # Create classification report for logistic regression model
# target_names = ['Predicted would not leave', 'Predicted would leave']
# res = classification_report(y_test, y_pred, target_names=target_names)

# st.markdown(
# """
# | Predicted Label           | Precision   | Recall   | F1-score   | Support |
# |--------------------------|:------------|:---------|:-----------|----------:|
# | Predicted would not leave | 0.86        | 0.93     | 0.9        |      2321 |
# | Predicted would leave     | 0.44        | 0.26     | 0.33       |       471 |
# | accuracy                  | -           | -        | 0.82          |      2792 |
# | macro avg                 | 0.65        | 0.6      | 0.61       |      2792 |
# | weighted avg              | 0.79        | 0.82     | 0.8        |      2792 |
# """)

# """
# The classification report above shows that the logistic regression model achieved a precision of 79%, recall of 82%, f1-score of 80% (all weighted averages), and accuracy of 82%. However, if it's most important to predict employees who leave, then the scores are significantly lower.
# """
# """
# ### Modeling Approach B: Tree-based Model
# This approach covers implementation of Decision Tree and Random Forest.
# """

# # Isolate the outcome variable
# y = df_enc['left']
# # Select the features you want to use in your model
# X = df_enc.drop('left', axis=1)
# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# """
# #### Decision tree - Round 1
# Construct a decision tree model and set up cross-validated grid-search to exhuastively search for the best model parameters.
# """
# st.code("""
# # Instantiate model
# tree = DecisionTreeClassifier(random_state=0)

# # Assign a dictionary of hyperparameters to search over
# cv_params = {'max_depth':[4, 6, 8, None],
#              'min_samples_leaf': [2, 5, 1],
#              'min_samples_split': [2, 4, 6]
#              }

# # Assign a dictionary of scoring metrics to capture
# scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# # Instantiate GridSearch
# tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')
#         """)
# """
# Fit the decision tree model to the training data.
# `tree1.fit(X_train, y_train)`
# """

# # Instantiate model
# tree = DecisionTreeClassifier(random_state=0)

# # Assign a dictionary of hyperparameters to search over
# cv_params = {'max_depth':[4, 6, 8, None],
#              'min_samples_leaf': [2, 5, 1],
#              'min_samples_split': [2, 4, 6]
#              }

# # Assign a dictionary of scoring metrics to capture
# scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# # Instantiate GridSearch
# tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# tree1.fit(X_train, y_train)

# st.code("""
# GridSearchCV(cv=4, error_score=nan,
#              estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
#                                               criterion='gini', max_depth=None,
#                                               max_features=None,
#                                               max_leaf_nodes=None,
#                                               min_impurity_decrease=0.0,
#                                               min_impurity_split=None,
#                                               min_samples_leaf=1,
#                                               min_samples_split=2,
#                                               min_weight_fraction_leaf=0.0,
#                                               presort='deprecated',
#                                               random_state=0, splitter='best'),
#              iid='deprecated', n_jobs=None,
#              param_grid={'max_depth': [4, 6, 8, None],
#                          'min_samples_leaf': [2, 5, 1],
#                          'min_samples_split': [2, 4, 6]},
#              pre_dispatch='2*n_jobs', refit='roc_auc', return_train_score=False,
#              scoring={'f1', 'precision', 'accuracy', 'roc_auc', 'recall'},
#              verbose=0)
# """)

# st.code("""
# # Check best parameters
# tree1.best_params_
# # -> {'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 2}
# # Check best AUC score on CV
# tree1.best_score_
# # -> 0.969819392792457
#         """)

# """
# This is a strong AUC score, which shows that this model can predict employees who will leave very well.
# """
# # Get all CV scores
# tree1_cv_results = make_results('decision tree cv', tree1, 'auc')
# st.dataframe(tree1_cv_results)

# """
# All of these scores from the decision tree model are strong indicators of good model performance.

# Recall that decision trees can be vulnerable to overfitting, and random forests avoid overfitting by incorporating multiple trees to make predictions. You could construct a random forest model next.
# """

# """
# #### Random forest - Round 1
# Construct a random forest model and set up cross-validated grid-search to exhuastively search for the best model parameters.
# """

# st.code("""
# # Instantiate model
# rf = RandomForestClassifier(random_state=0)

# # Assign a dictionary of hyperparameters to search over
# cv_params = {'max_depth': [3,5, None],
#              'max_features': [1.0],
#              'max_samples': [0.7, 1.0],
#              'min_samples_leaf': [1,2,3],
#              'min_samples_split': [2,3,4],
#              'n_estimators': [300, 500],
#              }

# # Assign a dictionary of scoring metrics to capture
# scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# # Instantiate GridSearch
# rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')
#         """)
# """
# Fit the random forest model to the training data
# """

# # Instantiate model
# rf = RandomForestClassifier(random_state=0)

# # Assign a dictionary of hyperparameters to search over
# cv_params = {'max_depth': [3,5, None],
#              'max_features': [1.0],
#              'max_samples': [0.7, 1.0],
#              'min_samples_leaf': [1,2,3],
#              'min_samples_split': [2,3,4],
#              'n_estimators': [300, 500],
#              }

# # Assign a dictionary of scoring metrics to capture
# scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# # Instantiate GridSearch
# rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# path = './models/'

# # magari se si vuole trainare il modello metto un buttone. Altrimenti ci mette troppo tempo
# if st.button("Push the button to Fit the Model, Wall time: ~10min", help="the model has been already fitted. Do not push the button if want to continue reading quickly"):

#     rf1.fit(X_train, y_train) # 10 minuti. Magari per la demo posso mettere meno parametri

#     st.code(
#     """
#     GridSearchCV(cv=4, error_score=nan,
#                 estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
#                                                 class_weight=None,
#                                                 criterion='gini', max_depth=None,
#                                                 max_features='auto',
#                                                 max_leaf_nodes=None,
#                                                 max_samples=None,
#                                                 min_impurity_decrease=0.0,
#                                                 min_impurity_split=None,
#                                                 min_samples_leaf=1,
#                                                 min_samples_split=2,
#                                                 min_weight_fraction_leaf=0.0,
#                                                 n_estimators=100, n_jobs=None,...
#                                                 verbose=0, warm_start=False),
#                 iid='deprecated', n_jobs=None,
#                 param_grid={'max_depth': [3, 5, None], 'max_features': [1.0],
#                             'max_samples': [0.7, 1.0],
#                             'min_samples_leaf': [1, 2, 3],
#                             'min_samples_split': [2, 3, 4],
#                             'n_estimators': [300, 500]},
#                 pre_dispatch='2*n_jobs', refit='roc_auc', return_train_score=False,
#                 scoring={'f1', 'precision', 'accuracy', 'roc_auc', 'recall'},
#                 verbose=0)
#     """
#     )

#     """
#     It is possible to save your model, and load it when necessary
#     """
#     # Write pickle
#     write_pickle(path, rf1, 'hr_rf1')

# # Read pickle
# rf1 = read_pickle(path, 'hr_rf1')

# """Identify the best AUC score achieved by the random forest model on the training set. Identify the optimal values for the parameters of the random forest model."""
# st.code("""
# # Check best AUC score on CV
# rf1.best_score_
# # -> 0.9804250949807172
# # Check best params
# rf1.best_params_
# # {'max_depth': 5,'max_features': 1.0,'max_samples': 0.7,'min_samples_leaf': 1,'min_samples_split': 4,'n_estimators': 500}
# """)

# # Get all CV scores
# rf1_cv_results = make_results('random forest cv', rf1, 'auc')
# st.dataframe(pd.concat([tree1_cv_results,rf1_cv_results]))

# """
# The evaluation scores of the random forest model are better than those of the decision tree model, with the exception of recall (the recall score of the random forest model is approximately 0.001 lower, which is a negligible amount). This indicates that the random forest model mostly outperforms the decision tree model.

# Next, you can evaluate the final model on the test set.
# """

# # Get predictions on test data
# rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
# st.dataframe(rf1_test_scores)

# """
# The test scores are very similar to the validation scores, which is good. This appears to be a strong model. Since this test set was only used for this model, you can be more confident that your model's performance on this data is representative of how it will perform on new, unseeen data.
# """

# """
# #### Feature Engineering

# You might be skeptical of the high evaluation scores. There is a chance that there is some data leakage occurring. Data leakage is when you use data to train your model that should not be used during training, either because it appears in the test data or because it's not data that you'd expect to have when the model is actually deployed. Training a model with leaked data can give an unrealistic score that is not replicated in production.

# In this case, it's likely that the company won't have satisfaction levels reported for all of its employees. It's also possible that the `average_monthly_hours` column is a source of some data leakage. If employees have already decided upon quitting, or have already been identified by management as people to be fired, they may be working fewer hours.

# The first round of decision tree and random forest models included all variables as features. This next round will incorporate feature engineering to build improved models.

# You could proceed by dropping `satisfaction_level` and creating a new feature that roughly captures whether an employee is overworked. You could call this new feature `overworked`. It will be a binary variable.
# """

# # Drop `satisfaction_level` and save resulting dataframe in new variable
# df2 = df_enc.drop('satisfaction_level', axis=1)

# # Create `overworked` column. For now, it's identical to average monthly hours.
# df2['overworked'] = df2['average_monthly_hours']

# """
# 166.67 is approximately the average number of monthly hours for someone who works 50 weeks per year, 5 days per week, 8 hours per day.

# You could define being overworked as working more than 175 hours per month on average.

# To make the `overworked` column binary, you could reassign the column using a boolean mask.
# - `df2['overworked'] > 175` creates a series of booleans, consisting of `True` for every value > 175 and `False` for every values ≤ 175
# - `.astype(int)` converts all `True` to `1` and all `False` to `0`
# """

# # Define `overworked` as working > 175 hrs/week
# df2['overworked'] = (df2['overworked'] > 175).astype(int)

# """
# Drop the `average_monthly_hours` column.
# """

# # Drop the `average_monthly_hours` column
# df2 = df2.drop('average_monthly_hours', axis=1)

# """
# Again, isolate the features and target variables
# Split the data into training and testing sets.
# """

# # Isolate the outcome variable
# y = df2['left']

# # Select the features
# X = df2.drop('left', axis=1)

# # Create test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# """
# #### Decision tree - Round 2
# """

# # Instantiate model
# tree = DecisionTreeClassifier(random_state=0)

# # Assign a dictionary of hyperparameters to search over
# cv_params = {'max_depth':[4, 6, 8, None],
#              'min_samples_leaf': [2, 5, 1],
#              'min_samples_split': [2, 4, 6]
#              }

# # Assign a dictionary of scoring metrics to capture
# scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# # Instantiate GridSearch
# tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# tree2.fit(X_train, y_train)

# st.code("""
# # Check best params
# tree2.best_params_
# # -> {'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 6}
# # Check best AUC score on CV
# tree2.best_score_
# # -> 0.9586752505340426
#         """)
# """
# This model performs very well, even without satisfaction levels and detailed hours worked data.

# Next, check the other scores.
# """

# # Get all CV scores
# tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')

# st.dataframe(pd.concat([tree1_cv_results,tree2_cv_results,rf1_cv_results]))

# """
# Some of the other scores fell. That's to be expected given fewer features were taken into account in this round of the model. Still, the scores are very good.
# """

# """
# #### Random forest - Round 2
# """

# # Instantiate model
# rf = RandomForestClassifier(random_state=0)

# # Assign a dictionary of hyperparameters to search over
# cv_params = {'max_depth': [3,5, None],
#              'max_features': [1.0],
#              'max_samples': [0.7, 1.0],
#              'min_samples_leaf': [1,2,3],
#              'min_samples_split': [2,3,4],
#              'n_estimators': [300, 500],
#              }

# # Assign a dictionary of scoring metrics to capture
# scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# # Instantiate GridSearch
# rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# # magari se si vuole trainare il modello metto un buttone. Altrimenti ci mette troppo tempo
# if st.button("Push the model to Fit the Model, Wall time: ~7min", help="the model has been already fitted. Do not push the button if want to continue reading quickly"):
#     rf2.fit(X_train, y_train) # --> Wall time: 7min 5s

#     # Write pickle
#     write_pickle(path, rf2, 'hr_rf2')

# # Read in pickle
# rf2 = read_pickle(path, 'hr_rf2')

# st.code("""
# # Check best params
# rf2.best_params_
# # {'max_depth': 5,'max_features': 1.0,'max_samples': 0.7,'min_samples_leaf': 2,'min_samples_split': 2,'n_estimators': 300}
# # Check best AUC score on CV
# rf2.best_score_
# # -> 0.9648100662833985
# """)

# # Get all CV scores
# rf2_cv_results = make_results('random forest2 cv', rf2, 'auc')
# st.dataframe(pd.concat([tree1_cv_results,tree2_cv_results,rf1_cv_results, rf2_cv_results]))

# """
# Again, the scores dropped slightly, but the random forest performs better than the decision tree if using AUC as the deciding metric.

# Score the champion model on the test set now.
# """

# # Get predictions on test data
# rf2_test_scores = get_scores('random forest2 test', rf2, X_test, y_test)
# st.dataframe(pd.concat([rf1_test_scores,rf2_test_scores]))

# """
# This seems to be a stable, well-performing final model.

# Plot a confusion matrix to visualize how well it predicts on the test set.
# """

# # Generate array of values for confusion matrix
# preds = rf2.best_estimator_.predict(X_test)
# cm = confusion_matrix(y_test, preds, labels=rf2.classes_)

# # Plot confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                              display_labels=rf2.classes_)

# # Plot confusion matrix
# fig, ax = plt.subplots(figsize=(8, 6))
# disp.plot(ax=ax, values_format='')
# ax.set_title('Confusion Matrix | Random Forest (2)')
# ax.set_xlabel('Predicted Label')
# ax.set_ylabel('True Label')

# st.pyplot(fig, use_container_width=False, dpi=100)

# """
# The model predicts more false positives than false negatives, which means that some employees may be identified as at risk of quitting or getting fired, when that's actually not the case. But this is still a strong model.

# For exploratory purpose, you might want to inspect the most important features in the random forest model.
# """

# """
# #### Decision tree feature importance

# You can also get feature importance from decision trees
# """

# #tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=X.columns)
# tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_,
#                                  columns=['gini_importance'],
#                                  index=X.columns
#                                 )
# tree2_importances = tree2_importances.sort_values(by='gini_importance', ascending=False)

# # Only extract the features with importances > 0
# tree2_importances = tree2_importances[tree2_importances['gini_importance'] != 0]

# tree2_importances

# st.bar_chart(tree2_importances)

# """
# The barplot above shows that in this decision tree model, `last_evaluation`, `number_project`, `tenure`, and `overworked` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`.
# """

# """
# #### Random forest feature importance

# Now, plot the feature importances for the random forest model.
# """

# # Get feature importances
# feat_impt = rf2.best_estimator_.feature_importances_

# # Get indices of top 10 features
# ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]

# # Get column labels of top 10 features
# feat = X.columns[ind]

# # Filter `feat_impt` to consist of top 10 feature importances
# feat_impt = feat_impt[ind]

# y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
# y_sort_df = y_df.sort_values("Importance")

# y_sort_df
# st.bar_chart(y_sort_df, x="Feature", y="Importance")

# """
# The plot above shows that in this random forest model, `last_evaluation`, `number_project`, `tenure`, and `overworked` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`, and they are the same as the ones used by the decision tree model.
# """

# """
# # pacE: Execute Stage
# - Interpret model performance and results
# - Share actionable steps with stakeholders

# ✏
# ## Recall evaluation metrics

# - **AUC** is the area under the ROC curve; it's also considered the probability that the model ranks a random positive example more highly than a random negative example.
# - **Precision** measures the proportion of data points predicted as True that are actually True, in other words, the proportion of positive predictions that are true positives.
# - **Recall** measures the proportion of data points that are predicted as True, out of all the data points that are actually True. In other words, it measures the proportion of positives that are correctly classified.
# - **Accuracy** measures the proportion of data points that are correctly classified.
# - **F1-score** is an aggregation of precision and recall.

# 💭
# ### Reflect on these questions as you complete the executing stage.

# - What key insights emerged from your model(s)?
# - What business recommendations do you propose based on the models built?
# - What potential recommendations would you make to your manager/company?
# - Do you think your model could be improved? Why or why not? How?
# - Given what you know about the data and the models you were using, what other questions could you address for the team?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?

# ## Step 4. Results and Evaluation
# - Interpret model
# - Evaluate model performance using metrics
# - Prepare results, visualizations, and actionable steps to share with stakeholders

# ### Summary of model results

# **Logistic Regression**

# The logistic regression model achieved precision of 80%, recall of 83%, f1-score of 80% (all weighted averages), and accuracy of 83%, on the test set.

# **Tree-based Machine Learning**

# After conducting feature engineering, the decision tree model achieved AUC of 93.8%, precision of 87.0%, recall of 90.4%, f1-score of 88.7%, and accuracy of 96.2%, on the test set. The random forest modestly outperformed the decision tree model.
# """

# """
# ### Conclusion, Recommendations, Next Steps

# The models and the feature importances extracted from the models confirm that employees at the company are overworked.

# To retain employees, the following recommendations could be presented to the stakeholders:

# * Cap the number of projects that employees can work on.
# * Consider promoting employees who have been with the company for atleast four years, or conduct further investigation about why four-year tenured employees are so dissatisfied.
# * Either reward employees for working longer hours, or don't require them to do so.
# * If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear.
# * Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts.
# * High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort.

# **Next Steps**

# It may be justified to still have some concern about data leakage. It could be prudent to consider how predictions change when `last_evaluation` is removed from the data. It's possible that evaluations aren't performed very frequently, in which case it would be useful to be able to predict employee retention without this feature. It's also possible that the evaluation score determines whether an employee leaves or stays, in which case it could be useful to pivot and try to predict performance score. The same could be said for satisfaction score.

# For another project, you could try building a K-means model on this data and analyzing the clusters. This may yield valuable insight.
# """

# """
# # EXECUTIVE SUMMARY

# ## Salifort Motors
# Employee Retention Project 

# """

# col1, col2 = st.columns(2)

# with col1:
    
#     st.subheader("ISSUE/PROBLEM")
#     st.markdown("""Salifort Motors seeks to improve employee retention and answer the following question:
#                 *What’s likely to make the employee leave the company?*""")

#     st.subheader("RESPONSE")
#     st.markdown("""Since the variable we are seeking to predict is categorical, the team could build either a logistic regression or a tree-based machine learning model.
#     The random forest model slightly outperforms the decision tree model.""")
#     st.subheader("IMPACT")
#     st.markdown("""This model helps predict whether an employee will leave and identify which factors are most influential. These insights can help HR make decisions to improve employee retention.""")

# with col2: 
#     st.bar_chart(tree2_importances)
#     """Barplot above shows the most relevant variables: `last_evaluation`, `number_project`,  `tenure` and `overworked`."""
#     st.bar_chart(y_sort_df, x="Feature", y="Importance")
#     """In the random forest model above, `last_evaluation`, `tenure`, `number_project`, `overworked`, `salary_low`, and `work_accident` have the highest importance. These variables are most helpful in predicting the outcome variable, `left`."""

# st.subheader("INSIGHT AND NEXT STEPS")
# """
# Cap the number of projects that employees can work on.
# Consider promoting employees who have been with the company for at least four years, or conduct further investigation about why four-year tenured employees are so dissatisfied.
# Either reward employees for working longer hours, or don't require them to do so.
# If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear.
# Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts.
# High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort.
# """

