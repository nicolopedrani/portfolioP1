import streamlit as st

# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
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

# Title and Text Block
st.title("Employee Turnover Analysis at Salifort Motors")

df1 = st.session_state['df0'].drop_duplicates(keep='first')

"""
# paCe: Construct Stage 🔎 🔨
- Determine which models are most appropriate
- Construct the model
- Confirm model assumptions
- Evaluate model results to determine how well our model fits the data
"""

"""

## Recall model assumptions

**Logistic Regression model assumptions**
- Outcome variable is categorical
- Observations are independent of each other
- No severe multicollinearity among X variables
- No extreme outliers
- Linear relationship between each X variable and the logit, logarithm of the odds ln(p/(1- p)), of the outcome variable
- Sufficiently large sample size

### Reflect on these questions

- Which independent variables do we choose for the model and why?
- Are each of the assumptions met?
- How well does our model fit the data?
- Can we improve it? Is there anything we would change about the model?

## Model Building, Results and Evaluation
- Fit a model that predicts the outcome variable using two or more independent variables
- Check model assumptions
- Evaluate the model

### Identify the type of prediction task.
Our goal is to predict whether an employee leaves the company, which is a categorical outcome variable. So this task involves classification. More specifically, this involves binary classification, since the outcome variable `left` can be either 1 (indicating employee left) or 0 (indicating employee didn't leave).

### Identify the types of models most appropriate for this task.
Since the variable we want to predict (whether an employee leaves the company) is categorical, we could either build a Logistic Regression model, or a Tree-based Machine Learning model.

### Modeling Approach A: Logistic Regression Model

The binomial logistic regression suits the task because it involves binary classification.

Before splitting the data, encode the non-numeric variables. There are two: `department` and `salary`.

`department` is a categorical variable, which means we can dummy it for modeling.

`salary` is categorical too, but it's ordinal. There's a hierarchy to the categories, so it's better not to dummy this column, but rather to convert the levels to numbers, 0&ndash;2.

"""

st.code("""
# Copy the dataframe
df_enc = df1.copy()

# Encode the `salary` column as an ordinal numeric category
df_enc['salary'] = (
    df_enc['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
df_enc = pd.get_dummies(df_enc, drop_first=False)
        """)

# Copy the dataframe
df_enc = df1.copy()

# Encode the `salary` column as an ordinal numeric category
df_enc['salary'] = (
    df_enc['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
df_enc = pd.get_dummies(df_enc, drop_first=False)

"""
Create a heatmap to visualize how correlated variables are. Consider which variables we're interested in examining correlations between.
"""

# Calculate correlation matrix
correlation_matrix = df1[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure']].corr()

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
Since logistic regression is quite sensitive to outliers, it would be a good idea at this stage to remove the outliers in the `tenure` column that were identified earlier.
"""

st.code('''
df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <= upper_limit)]
        ''')

df_logreg = df_enc[(df_enc['tenure'] >= st.session_state['lower_limit']) & (df_enc['tenure'] <= st.session_state['upper_limit'])]

"""
- Isolate the outcome variable, which is the variable we want our model to predict.  
- Select the features we want to use in our model. Consider which variables will help we predict the outcome variable, `left`.  
- Split the data into training set and testing set.  
- Construct a logistic regression model and fit it to the training dataset.  
- Test the logistic regression model: use the model to make predictions on the test set.  
- Create a confusion matrix to visualize the results of the logistic regression model.
"""

# Isolate the outcome variable
y = df_logreg['left']
# Select the features you want to use in your model
X = df_logreg.drop('left', axis=1)

st.code("""
# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
# Use the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test)   
        """)

# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
# Use the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test) 

# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm,
                                  display_labels=log_clf.classes_)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
log_disp.plot(ax=ax, values_format='')
ax.set_title('Confusion Matrix | Logistic Regression')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')

st.pyplot(fig, use_container_width=False, dpi=100)

"""
The upper-left quadrant displays the number of true negatives.
The upper-right quadrant displays the number of false positives.
The bottom-left quadrant displays the number of false negatives.
The bottom-right quadrant displays the number of true positives.

- True negatives: The number of people who did not leave that the model accurately predicted did not leave.
- False positives: The number of people who did not leave the model inaccurately predicted as leaving.
- False negatives: The number of people who left that the model inaccurately predicted did not leave
- True positives: The number of people who left the model accurately predicted as leaving

A perfect model would yield all true negatives and true positives, and no false negatives or false positives.
"""

"""
Check the class balance in the data, the class balance informs the way you interpret accuracy metrics, and create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the logistic regression model.  
"""

color_stay = 'skyblue'
color_left = 'salmon'

# Create data frame for the bar chart
chart_data = df_logreg['left'].value_counts(normalize=True).to_frame().T
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
There is an approximately 83%-17% split. So the data is not perfectly balanced, but it is not too imbalanced. If it was more severely imbalanced, we might want to resample the data to make it more balanced.
"""

# Create classification report for logistic regression model
target_names = ['Predicted would not leave', 'Predicted would leave']
res = classification_report(y_test, y_pred, target_names=target_names)

st.markdown(
"""
| Predicted Label           | Precision   | Recall   | F1-score   | Support |
|--------------------------|:------------|:---------|:-----------|----------:|
| Predicted would not leave | 0.86        | 0.93     | 0.9        |      2321 |
| Predicted would leave     | 0.44        | 0.26     | 0.33       |       471 |
| accuracy                  | -           | -        | 0.82          |      2792 |
| macro avg                 | 0.65        | 0.6      | 0.61       |      2792 |
| weighted avg              | 0.79        | 0.82     | 0.8        |      2792 |
""")

"""
The classification report above shows that the logistic regression model achieved a precision of 79%, recall of 82%, f1-score of 80% (all weighted averages), and accuracy of 82%. However, if it's most important to predict employees who leave, then the scores are significantly lower.
"""
"""
### Modeling Approach B: Tree-based Model
"""

# Isolate the outcome variable
y = df_enc['left']
# Select the features you want to use in your model
X = df_enc.drop('left', axis=1)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

"""
#### Decision tree - Round 1
Construct a decision tree model and set up cross-validated grid-search to exhuastively search for the best model parameters.
"""
st.code("""
# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')
        """)
"""
Fit the decision tree model to the training data.
`tree1.fit(X_train, y_train)`
"""

# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Instantiate GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

tree1.fit(X_train, y_train)

st.code("""
GridSearchCV(cv=4, error_score=nan,
             estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features=None,
                                              max_leaf_nodes=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              presort='deprecated',
                                              random_state=0, splitter='best'),
             iid='deprecated', n_jobs=None,
             param_grid={'max_depth': [4, 6, 8, None],
                         'min_samples_leaf': [2, 5, 1],
                         'min_samples_split': [2, 4, 6]},
             pre_dispatch='2*n_jobs', refit='roc_auc', return_train_score=False,
             scoring={'f1', 'precision', 'accuracy', 'roc_auc', 'recall'},
             verbose=0)
""")

st.code("""
# Check best parameters
tree1.best_params_
# -> {'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 2}
# Check best AUC score on CV
tree1.best_score_
# -> 0.969819392792457
        """)

"""
This is a strong AUC score, which shows that this model can predict employees who will leave very well.
"""
# Get all CV scores
tree1_cv_results = make_results('decision tree cv', tree1, 'auc')
st.dataframe(tree1_cv_results)

"""
All of these scores from the decision tree model are strong indicators of good model performance.

Recall that decision trees can be vulnerable to overfitting, and random forests avoid overfitting by incorporating multiple trees to make predictions. We could construct a random forest model next.
"""

"""
#### Random forest - Round 1
Construct a random forest model and set up cross-validated grid-search to exhuastively search for the best model parameters.
"""

st.code("""
# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None],
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')
        """)
"""
Fit the random forest model to the training data
"""

# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None],
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }

# Assign a dictionary of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Instantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

path = './models/'

# magari se si vuole trainare il modello metto un buttone. Altrimenti ci mette troppo tempo
if st.button("Push the button to Fit the Model, Wall time: ~10min", help="the model has been already fitted. Do not push the button if want to continue reading quickly"):

    rf1.fit(X_train, y_train) # 10 minuti. Magari per la demo posso mettere meno parametri

    st.code(
    """
    GridSearchCV(cv=4, error_score=nan,
                estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                                class_weight=None,
                                                criterion='gini', max_depth=None,
                                                max_features='auto',
                                                max_leaf_nodes=None,
                                                max_samples=None,
                                                min_impurity_decrease=0.0,
                                                min_impurity_split=None,
                                                min_samples_leaf=1,
                                                min_samples_split=2,
                                                min_weight_fraction_leaf=0.0,
                                                n_estimators=100, n_jobs=None,...
                                                verbose=0, warm_start=False),
                iid='deprecated', n_jobs=None,
                param_grid={'max_depth': [3, 5, None], 'max_features': [1.0],
                            'max_samples': [0.7, 1.0],
                            'min_samples_leaf': [1, 2, 3],
                            'min_samples_split': [2, 3, 4],
                            'n_estimators': [300, 500]},
                pre_dispatch='2*n_jobs', refit='roc_auc', return_train_score=False,
                scoring={'f1', 'precision', 'accuracy', 'roc_auc', 'recall'},
                verbose=0)
    """
    )

    """
    It is possible to save your model, and load it when necessary
    """
    # Write pickle
    write_pickle(path, rf1, 'hr_rf1')

# Read pickle
rf1 = read_pickle(path, 'hr_rf1')

"""Identify the best AUC score achieved by the random forest model on the training set. Identify the optimal values for the parameters of the random forest model."""
st.code("""
# Check best AUC score on CV
rf1.best_score_
# -> 0.9804250949807172
# Check best params
rf1.best_params_
# {'max_depth': 5,'max_features': 1.0,'max_samples': 0.7,'min_samples_leaf': 1,'min_samples_split': 4,'n_estimators': 500}
""")

# Get all CV scores
rf1_cv_results = make_results('random forest cv', rf1, 'auc')
st.dataframe(pd.concat([tree1_cv_results,rf1_cv_results]))

"""
The evaluation scores of the random forest model are better than those of the decision tree model, with the exception of recall (the recall score of the random forest model is approximately 0.001 lower, which is a negligible amount). This indicates that the random forest model mostly outperforms the decision tree model.

Next, we can evaluate the final model on the test set.
"""

# Get predictions on test data
rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
st.dataframe(rf1_test_scores)

"""
The test scores are very similar to the validation scores, which is good. This appears to be a strong model. Since this test set was only used for this model, we can be more confident that our model's performance on this data is representative of how it will perform on new, unseeen data.
"""

"""
#### Feature Engineering

There is a chance that there is some data leakage occurring. Data leakage is when you use data to train your model that should not be used during training, either because it appears in the test data or because it's not data that you'd expect to have when the model is actually deployed. Training a model with leaked data can give an unrealistic score that is not replicated in production.

In this case, it's likely that the company won't have satisfaction levels reported for all of its employees. It's also possible that the `average_monthly_hours` column is a source of some data leakage. If employees have already decided upon quitting, or have already been identified by management as people to be fired, they may be working fewer hours.

The first round of decision tree and random forest models included all variables as features. This next round will incorporate feature engineering to build improved models.

We could proceed by dropping `satisfaction_level` and creating a new feature that roughly captures whether an employee is overworked. We could call this new feature `overworked`. It will be a binary variable.
"""

# Drop `satisfaction_level` and save resulting dataframe in new variable
df2 = df_enc.drop('satisfaction_level', axis=1)

# Create `overworked` column. For now, it's identical to average monthly hours.
df2['overworked'] = df2['average_monthly_hours']

"""
166.67 is approximately the average number of monthly hours for someone who works 50 weeks per year, 5 days per week, 8 hours per day.

We could define being overworked as working more than 175 hours per month on average.

To make the `overworked` column binary, we could reassign the column using a boolean mask.
- `df2['overworked'] > 175` creates a series of booleans, consisting of `True` for every value > 175 and `False` for every values ≤ 175
- `.astype(int)` converts all `True` to `1` and all `False` to `0`
"""

# Define `overworked` as working > 175 hrs/week
df2['overworked'] = (df2['overworked'] > 175).astype(int)

"""
Drop the `average_monthly_hours` column.
"""

# Drop the `average_monthly_hours` column
df2 = df2.drop('average_monthly_hours', axis=1)

"""
Again, isolate the features and target variables
Split the data into training and testing sets.
"""

# Isolate the outcome variable
y = df2['left']

# Select the features
X = df2.drop('left', axis=1)

# Create test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

"""
#### Decision tree - Round 2
"""

# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Instantiate GridSearch
tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

tree2.fit(X_train, y_train)

st.code("""
# Check best params
tree2.best_params_
# -> {'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 6}
# Check best AUC score on CV
tree2.best_score_
# -> 0.9586752505340426
        """)
"""
This model performs very well, even without satisfaction levels and detailed hours worked data.

Next, check the other scores.
"""

# Get all CV scores
tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')

st.dataframe(pd.concat([tree1_cv_results,tree2_cv_results,rf1_cv_results]))

"""
Some of the other scores fell. That's to be expected given fewer features were taken into account in this round of the model. Still, the scores are very good.
"""

"""
#### Random forest - Round 2
"""

# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None],
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }

# Assign a dictionary of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Instantiate GridSearch
rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# magari se si vuole trainare il modello metto un buttone. Altrimenti ci mette troppo tempo
if st.button("Push the model to Fit the Model, Wall time: ~7min", help="the model has been already fitted. Do not push the button if want to continue reading quickly"):
    rf2.fit(X_train, y_train) # --> Wall time: 7min 5s

    # Write pickle
    write_pickle(path, rf2, 'hr_rf2')

# Read in pickle
rf2 = read_pickle(path, 'hr_rf2')

st.code("""
# Check best params
rf2.best_params_
# {'max_depth': 5,'max_features': 1.0,'max_samples': 0.7,'min_samples_leaf': 2,'min_samples_split': 2,'n_estimators': 300}
# Check best AUC score on CV
rf2.best_score_
# -> 0.9648100662833985
""")

# Get all CV scores
rf2_cv_results = make_results('random forest2 cv', rf2, 'auc')
st.dataframe(pd.concat([tree1_cv_results,tree2_cv_results,rf1_cv_results, rf2_cv_results]))

"""
Again, the scores dropped slightly, but the random forest performs better than the decision tree if using AUC as the deciding metric.

Score the champion model on the test set now.
"""

# Get predictions on test data
rf2_test_scores = get_scores('random forest2 test', rf2, X_test, y_test)
st.dataframe(pd.concat([rf1_test_scores,rf2_test_scores]))

"""
This seems to be a stable, well-performing final model.

Plot a confusion matrix to visualize how well it predicts on the test set.
"""

# Generate array of values for confusion matrix
preds = rf2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf2.classes_)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, values_format='')
ax.set_title('Confusion Matrix | Random Forest (2)')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')

st.pyplot(fig, use_container_width=False, dpi=100)

"""
The model predicts more false positives than false negatives, which means that some employees may be identified as at risk of quitting or getting fired, when that's actually not the case. But this is still a strong model.

For exploratory purpose, you might want to inspect the most important features in the random forest model.
"""

"""
#### Decision tree feature importance

You can also get feature importance from decision trees
"""

#tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=X.columns)
tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_,
                                 columns=['gini_importance'],
                                 index=X.columns
                                )
tree2_importances = tree2_importances.sort_values(by='gini_importance', ascending=False)

# Only extract the features with importances > 0
tree2_importances = tree2_importances[tree2_importances['gini_importance'] != 0]

tree2_importances

st.bar_chart(tree2_importances)


if 'tree2_importances' not in st.session_state:
        st.session_state['tree2_importances'] = tree2_importances

"""
The barplot above shows that in this decision tree model, `last_evaluation`, `number_project`, `tenure`, and `overworked` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`.
"""

"""
#### Random forest feature importance

Now, plot the feature importances for the random forest model.
"""

# Get feature importances
feat_impt = rf2.best_estimator_.feature_importances_

# Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]

# Get column labels of top 10 features
feat = X.columns[ind]

# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]

y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")

if 'y_sort_df' not in st.session_state:
        st.session_state['y_sort_df'] = y_sort_df
    
y_sort_df
st.bar_chart(y_sort_df, x="Feature", y="Importance")

"""
The plot above shows that in this random forest model, `last_evaluation`, `number_project`, `tenure`, and `overworked` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`, and they are the same as the ones used by the decision tree model.
"""
