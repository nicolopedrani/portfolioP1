import streamlit as st

from utils import *

# For data visualization
import plotly.express as px
import plotly.graph_objects as go

import numpy as np

# Title and Text Block
st.title("Employee Turnover Analysis at Salifort Motors")

df1 = st.session_state['df0'].drop_duplicates(keep='first')

"""
# pAce: Analyze Stage
- Perform EDA (analyze relationships between variables)
"""

"""
💭
### Reflect on these questions as you complete the analyze stage.

- What did you observe about the relationships between variables?
- What do you observe about the distributions in the data?
- What transformations did you make with your data? Why did you chose to make those decisions?
- What are some purposes of EDA before constructing a predictive model?
- What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
- Do you have any ethical considerations in this stage?
"""

st.subheader('Analyze Relationships between Variables')
'''Begin by understanding how many employees left and what percentage of all employees this figure represents'''

color_stay = 'skyblue'
color_left = 'salmon'

# Create data frame for the bar chart
chart_data = df1['left'].value_counts(normalize=True).to_frame().T
chart_data.columns = ['Stayed', 'Left']

# Create bar chart
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=chart_data.columns,
    y=chart_data.values.flatten(),
    marker_color=[color_stay, color_left]
))

# Update layout
fig3.update_layout(
    title='Employee Status',
    xaxis=dict(title='Status'),
    yaxis=dict(title='Percentage'),
    legend_title='Status'
)

st.plotly_chart(fig3)

"""
Now, examine variables that you're interested in, and create plots to visualize relationships between variables in the data.

You could start by creating a stacked boxplot showing `average_monthly_hours` distributions for `number_project`, comparing the distributions of employees who stayed versus those who left.  
Box plots are very useful in visualizing distributions within data, but they can be deceiving without the context of how big the sample sizes that they represent are. So, you could also plot a stacked histogram to visualize the distribution of `number_project` for those who stayed and those who left.
"""

# Define color scheme and legend titles
color_stay = 'skyblue'
color_left = 'salmon'
legend_titles = {'Stay': 'Stay', 'Left': 'Left'}

# Create boxplot showing `average_monthly_hours` distributions for `number_project`, comparing employees who stayed versus those who left
fig = go.Figure()

# Adding box for employees who stayed
fig.add_trace(go.Box(
    x=df1[df1['left'] == 0]['number_project'],
    y=df1[df1['left'] == 0]['average_monthly_hours'],
    name='Stay',
    marker_color=color_stay,
    boxmean=True,
    orientation='v'
))

# Adding box for employees who left
fig.add_trace(go.Box(
    x=df1[df1['left'] == 1]['number_project'],
    y=df1[df1['left'] == 1]['average_monthly_hours'],
    name='Left',
    marker_color=color_left,
    boxmean=True,
    orientation='v'
))

fig.update_layout(
    title='Monthly hours by number of projects',
    xaxis=dict(title='Number of Projects'),
    yaxis=dict(title='Average Monthly Hours')
)

# Create histogram showing distribution of `number_project`, comparing employees who stayed versus those who left
fig2 = px.histogram(df1, x="number_project", color="left", barmode="group", color_discrete_map={0: color_stay, 1: color_left})

fig2.update_layout(
    title='Number of projects histogram',
    xaxis=dict(title='Number of Projects'),
    yaxis=dict(title='Count'),
    legend_title=legend_titles['Left']
)

# Display plots in Streamlit
st.plotly_chart(fig)
st.plotly_chart(fig2)

"""
It might be natural that people who work on more projects would also work longer hours. This appears to be the case here, with the mean hours of each group (stayed and left) increasing with number of projects worked. However, a few things stand out from this plot.

1. There are two groups of employees who left the company: (A) those who worked considerably less than their peers with the same number of projects, and (B) those who worked much more. Of those in group A, it's possible that they were fired. It's also possible that this group includes employees who had already given their notice and were assigned fewer hours because they were already on their way out the door. For those in group B, it's reasonable to infer that they probably quit. The folks in group B likely contributed a lot to the projects they worked in; they might have been the largest contributors to their projects.

2. Everyone with seven projects left the company, and the interquartile ranges of this group and those who left with six projects was ~255&ndash;295 hours/month&mdash;much more than any other group.

3. The optimal number of projects for employees to work on seems to be 3&ndash;4. The ratio of left/stayed is very small for these cohorts.

4. If you assume a work week of 40 hours and two weeks of vacation per year, then the average number of working hours per month of employees working Monday&ndash;Friday `= 50 weeks * 40 hours per week / 12 months = 166.67 hours per month`. This means that, aside from the employees who worked on two projects, every group&mdash;even those who didn't leave the company&mdash;worked considerably more hours than this. It seems that employees here are overworked.
  
5. all employees with 7 projects did leave.
"""

# Define colors
color_stay = 'skyblue'
color_left = 'salmon'

# Create scatter plot
fig4 = go.Figure()

# Add scatter plot traces
fig4.add_trace(go.Scatter(
    x=df1[df1['left'] == 0]['average_monthly_hours'],
    y=df1[df1['left'] == 0]['satisfaction_level'],
    mode='markers',
    marker=dict(color=color_stay),
    name='Stay'
))

fig4.add_trace(go.Scatter(
    x=df1[df1['left'] == 1]['average_monthly_hours'],
    y=df1[df1['left'] == 1]['satisfaction_level'],
    mode='markers',
    marker=dict(color=color_left),
    name='Left'
))

# Add vertical line
fig4.add_shape(
    type='line',
    x0=166.67,
    y0=0,
    x1=166.67,
    y1=1,
    line=dict(color='red', dash='dash'),
    name='166.67 hrs./mo.'
)

# Update layout
fig4.update_layout(
    title='Monthly hours by Satisfaction Level',
    xaxis=dict(title='Average Monthly Hours'),
    yaxis=dict(title='Satisfaction Level'),
    legend_title='Status'
)

st.plotly_chart(fig4)

"""
The scatterplot above shows that there was a sizeable group of employees who worked ~240&ndash;315 hours per month. 315 hours per month is over 75 hours per week for a whole year. It's likely this is related to their satisfaction levels being close to zero.

The plot also shows another group of people who left, those who had more normal working hours. Even so, their satisfaction was only around 0.4. It's difficult to speculate about why they might have left. It's possible they felt pressured to work more, considering so many of their peers worked more. And that pressure could have lowered their satisfaction levels.

Finally, there is a group who worked ~210&ndash;280 hours per month, and they had satisfaction levels ranging ~0.7&ndash;0.9.

Note the strange shape of the distributions here. This is indicative of data manipulation or synthetic data.
"""

# Define color scheme and legend titles
color_stay = 'skyblue'
color_left = 'salmon'
legend_titles = {'Stay': 'Stay', 'Left': 'Left'}

fig5 = go.Figure()

# Adding box for employees who stayed
fig5.add_trace(go.Box(
    x=df1[df1['left'] == 0]['tenure'],
    y=df1[df1['left'] == 0]['satisfaction_level'],
    name='Stay',
    marker_color=color_stay,
    boxmean=True,
    orientation='v'
))

# Adding box for employees who left
fig5.add_trace(go.Box(
    x=df1[df1['left'] == 1]['tenure'],
    y=df1[df1['left'] == 1]['satisfaction_level'],
    name='Left',
    marker_color=color_left,
    boxmean=True,
    orientation='v'
))

fig5.update_layout(
    title='Satisfaction Level by Tenure',
    xaxis=dict(title='Tenure'),
    yaxis=dict(title='Satisfaction Level')
)

# Create histogram showing distribution of `tenure`, comparing employees who stayed versus those who left
fig6 = px.histogram(df1, x="tenure", color="left", barmode="group", color_discrete_map={0: color_stay, 1: color_left})

fig6.update_layout(
    title='Tenure histogram',
    xaxis=dict(title='Tenure'),
    yaxis=dict(title='Count'),
    legend_title=legend_titles['Left']
)

# Display plots in Streamlit
st.plotly_chart(fig5)
st.plotly_chart(fig6)

"""
There are many observations you could make from this plot.
- Employees who left fall into two general categories: dissatisfied employees with shorter tenures and very satisfied employees with medium-length tenures.
- Four-year employees who left seem to have an unusually low satisfaction level. It's worth investigating changes to company policy that might have affected people specifically at the four-year mark, if possible.
- The longest-tenured employees didn't leave. Their satisfaction levels aligned with those of newer employees who stayed.
- The histogram shows that there are relatively few longer-tenured employees. It's possible that they're the higher-ranking, higher-paid employees.
"""
"""
As the next step in analyzing the data, you could calculate the mean and median satisfaction scores of employees who left and those who didn't.
"""

# Group by 'left' column and calculate mean and median of 'satisfaction_level'
grouped_df = df1.groupby(['left'])['satisfaction_level'].agg([np.mean, np.median])

# Plotting
fig = go.Figure()

# Add mean trace for left = 0
fig.add_trace(go.Bar(
    x=['Left 0'],
    y=[grouped_df.loc[0, 'mean']],
    name='Mean',
    marker_color='rgba(52, 152, 219, 0.5)'  # Blue color with alpha (transparency)
))

# Add median trace for left = 0
fig.add_trace(go.Bar(
    x=['Left 0'],
    y=[grouped_df.loc[0, 'median']],
    name='Median',
    marker_color='rgba(41, 128, 185, 1)'  # Blue color with alpha (transparency)
))

# Add mean trace for left = 1
fig.add_trace(go.Bar(
    x=['Left 1'],
    y=[grouped_df.loc[1, 'mean']],
    name='Mean',
    marker_color='rgba(231, 76, 60, 0.5)'  # Red color with alpha (transparency)
))

# Add median trace for left = 1
fig.add_trace(go.Bar(
    x=['Left 1'],
    y=[grouped_df.loc[1, 'median']],
    name='Median',
    marker_color='rgba(192, 57, 43, 1)'  # Red color with alpha (transparency)
))

# Update layout
fig.update_layout(
    title='Satisfaction Level by Left Status',
    xaxis=dict(title='Left Status'),
    yaxis=dict(title='Satisfaction Level'),
    barmode='group'
)

# Show the plot
st.plotly_chart(fig)

"""
As expected, the mean and median satisfaction scores of employees who left are lower than those of employees who stayed. Interestingly, among employees who stayed, the mean satisfaction score appears to be slightly below the median score. This indicates that satisfaction levels among those who stayed might be skewed to the left.
"""

"""
Next, you could examine salary levels for different tenures.
"""

# Define short-tenured employees
tenure_short = df1[df1['tenure'] < 7]

# Define long-tenured employees
tenure_long = df1[df1['tenure'] > 6]

# Short-tenured histogram
fig1 = go.Figure()

for salary_group in ['low', 'medium', 'high']:
    fig1.add_trace(go.Histogram(
        x=tenure_short[tenure_short['salary'] == salary_group]['tenure'],
        name=salary_group,
        marker_color={'low': 'blue', 'medium': 'green', 'high': 'red'}[salary_group],
        opacity=0.7,
        # histnorm='percent',
        # nbinsx=20,
        # showlegend=True if salary_group == 'low' else False
    ))

fig1.update_layout(
    title_text='Salary histogram by tenure: short-tenured people',
    xaxis_title='Tenure',
    yaxis_title='Percentage',
    bargap=0.1
)

# Long-tenured histogram
fig2 = go.Figure()

for salary_group in ['low', 'medium', 'high']:
    fig2.add_trace(go.Histogram(
        x=tenure_long[tenure_long['salary'] == salary_group]['tenure'],
        name=salary_group,
        marker_color={'low': 'blue', 'medium': 'green', 'high': 'red'}[salary_group],
        opacity=0.7,
        # histnorm='percent',
        # nbinsx=20,
        # showlegend=True if salary_group == 'low' else False
    ))

fig2.update_layout(
    title_text='Salary histogram by tenure: long-tenured people',
    xaxis_title='Tenure',
    yaxis_title='Percentage',
    bargap=0.1
)

# Show the plots
st.plotly_chart(fig1)
st.plotly_chart(fig2)

"""
The plots above show that long-tenured employees were not disproportionately comprised of higher-paid employees.

Next, you could explore whether there's a correlation between working long hours and receiving high evaluation scores. You could create a scatterplot of `average_monthly_hours` versus `last_evaluation`.
"""

# Create scatter plot
fig4 = go.Figure()

# Add scatter plot traces
fig4.add_trace(go.Scatter(
    x=df1[df1['left'] == 0]['average_monthly_hours'],
    y=df1[df1['left'] == 0]['last_evaluation'],
    mode='markers',
    marker=dict(color=color_stay),
    name='Stay'
))

fig4.add_trace(go.Scatter(
    x=df1[df1['left'] == 1]['average_monthly_hours'],
    y=df1[df1['left'] == 1]['last_evaluation'],
    mode='markers',
    marker=dict(color=color_left),
    name='Left'
))

# Add vertical line
fig4.add_shape(
    type='line',
    x0=166.67,
    y0=0.35,
    x1=166.67,
    y1=1,
    line=dict(color='red', dash='dash'),
    name='166.67 hrs./mo.'
)

# Update layout
fig4.update_layout(
    title='Monthly hours by last evaluation score',
    xaxis=dict(title='Average Monthly Hours'),
    yaxis=dict(title='Last Evaluation'),
    legend_title='Status'
)

st.plotly_chart(fig4)

"""
The following observations can be made from the scatterplot above:
- The scatterplot indicates two groups of employees who left: overworked employees who performed very well and employees who worked slightly under the nominal monthly average of 166.67 hours with lower evaluation scores.
- There seems to be a correlation between hours worked and evaluation score.
- There isn't a high percentage of employees in the upper left quadrant of this plot; but working long hours doesn't guarantee a good evaluation score.
- Most of the employees in this company work well over 167 hours per month.

Next, you could examine whether employees who worked very long hours were promoted in the last five years.
"""

# Create scatter plot
fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=df1[df1['left'] == 1]['average_monthly_hours'],
    y=df1[df1['left'] == 1]['promotion_last_5years'],
    mode='markers',
    marker=dict(color=color_left),
    name='Left'
))

# Add scatter plot traces
fig4.add_trace(go.Scatter(
    x=df1[df1['left'] == 0]['average_monthly_hours'],
    y=df1[df1['left'] == 0]['promotion_last_5years'],
    mode='markers',
    marker=dict(color=color_stay),
    name='Stay'
))

# Add vertical line
fig4.add_shape(
    type='line',
    x0=166.67,
    y0=0,
    x1=166.67,
    y1=1,
    line=dict(color='red', dash='dash'),
    name='166.67 hrs./mo.'
)

# Update layout
fig4.update_layout(
    title='Monthly hours by Promotion Last 5 Years',
    xaxis=dict(title='Average Monthly Hours'),
    yaxis=dict(title='Promotion Last 5 Years'),
    legend_title='Status'
)

st.plotly_chart(fig4)

"""
The plot above shows the following:
- very few employees who were promoted in the last five years left
- very few employees who worked the most hours were promoted
- all of the employees who left were working the longest hours  

Next, you could inspect how the employees who left are distributed across departments.
"""

fig = go.Figure()

for left_value in [0, 1]:
    df_left = df1[df1['left'] == left_value]
    fig.add_trace(go.Histogram(
        x=df_left['department'],
        name=f'Left: {left_value}',
        marker_color=color_stay if left_value == 0 else color_left,
        # opacity=0.7,
        # histnorm='percent',
        showlegend=True
    ))

fig.update_layout(
    title='Counts of stayed/left by department',
    xaxis=dict(title='Department'),
    yaxis=dict(title='Percentage'),
    bargap=0.1,
    xaxis_tickangle=-45
)

st.plotly_chart(fig)

"""
There doesn't seem to be any department that differs significantly in its proportion of employees who left to those who stayed.

Lastly, you could check for strong correlations between variables in the data.
"""

# Calculate correlation matrix
correlation_matrix = df1[["satisfaction_level","last_evaluation","number_project","average_monthly_hours","tenure","work_accident","left","promotion_last_5years"]].corr()

# Define colorscale
colorscale = [[0, 'navy'], [0.5, 'lightsteelblue'], [1.0, 'firebrick']]

# Create heatmap
fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale=colorscale,
        colorbar=dict(title='Correlation', tickvals=[-1, 0, 1]),
        zmin=-1,
        zmax=1,
        hoverongaps = False
))

# Update layout
fig.update_layout(
    title='Correlation Heatmap',
    title_x=0.5,
    xaxis_title='Features',
    yaxis_title='Features',
    height=600,  # Increase the height
    width=800    # Increase the width
)

st.plotly_chart(fig)

"""
The correlation heatmap confirms that the number of projects, monthly hours, and evaluation scores all have some positive correlation with each other, and whether an employee leaves is negatively correlated with their satisfaction level.

### Insights

It appears that employees are leaving the company as a result of poor management. Leaving is tied to longer working hours, many projects, and generally lower satisfaction levels. It can be ungratifying to work long hours and not receive promotions or good evaluation scores. There's a sizeable group of employees at this company who are probably burned out. It also appears that if an employee has spent more than six years at the company, they tend not to leave.

"""
