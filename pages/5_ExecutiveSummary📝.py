import streamlit as st

# Title and Text Block
st.title("Employee Turnover Analysis at Salifort Motors")

"""
# Executive Summary

## Salifort Motors
Employee Retention Project 

"""

col1, col2 = st.columns(2)

with col1:
    
    st.subheader("ISSUE/PROBLEM")
    st.markdown("""Salifort Motors seeks to improve employee retention and answer the following question:
                *What’s likely to make the employee leave the company?*""")

    st.subheader("RESPONSE")
    st.markdown("""Since the variable we are seeking to predict is categorical, the team could build either a logistic regression or a tree-based machine learning model.
    The random forest model slightly outperforms the decision tree model.""")
    st.subheader("IMPACT")
    st.markdown("""This model helps predict whether an employee will leave and identify which factors are most influential. These insights can help HR make decisions to improve employee retention.""")

with col2: 
    st.bar_chart(st.session_state['tree2_importances'])
    """Barplot above shows the most relevant variables: `last_evaluation`, `number_project`,  `tenure` and `overworked`."""
    st.bar_chart(st.session_state['y_sort_df'], x="Feature", y="Importance")
    """In the random forest model above, `last_evaluation`, `tenure`, `number_project`, `overworked`, `salary_low`, and `work_accident` have the highest importance. These variables are most helpful in predicting the outcome variable, `left`."""

st.subheader("INSIGHT AND NEXT STEPS")
"""
Cap the number of projects that employees can work on.
Consider promoting employees who have been with the company for at least four years, or conduct further investigation about why four-year tenured employees are so dissatisfied.
Either reward employees for working longer hours, or don't require them to do so.
If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear.
Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts.
High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort.
"""

