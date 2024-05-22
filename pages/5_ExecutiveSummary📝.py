import streamlit as st

# Title and Text Block
st.title("Employee Turnover Analysis at Salifort Motors")

"""
# Executive Summary
## Salifort Motors | Employee Retention Project  
  
"""

if 'y_sort_df' not in st.session_state:
      cols = st.columns([1,3,1])
      with cols[1]:
        st.write("### Go first to the Construct page before continue")
        st.page_link("pages/3_Construct📈.py", label="Construct", icon="📈")
        st.stop()

col1, col2, col3 = st.columns([1,1,1])

with col1:
    
    st.subheader("Issue/Problem")
    st.markdown("""Salifort Motors seeks to improve employee retention and answer the following question:
                *What’s likely to make the employee leave the company?*""")

    st.subheader("Response")
    st.markdown("""Since the variable we are seeking to predict is categorical, the team could build either a logistic regression or a tree-based machine learning model.
    The random forest model slightly outperforms the decision tree model.""")
    st.subheader("Impact")
    st.markdown("""This model helps predict whether an employee will leave and identify which factors are most influential. These insights can help HR make decisions to improve employee retention.""")

with col3: 
    st.bar_chart(st.session_state['tree2_importances'])
    """Barplot above shows the most relevant variables in tree model: `last_evaluation`, `number_project`,  `tenure` and `overworked`."""
with col2:
    st.bar_chart(st.session_state['y_sort_df'], x="Feature", y="Importance")
    """Barplot above shows the most relevant variables in random forest model: `last_evaluation`, `tenure`, `number_project`, `overworked`, `salary_low`, and `work_accident`."""

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.subheader("Insight and Next Steps")
"""
Cap the number of projects that employees can work on.
Consider promoting employees who have been with the company for at least four years, or conduct further investigation about why four-year tenured employees are so dissatisfied.
Either reward employees for working longer hours, or don't require them to do so.
If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear.
Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts.
High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort.
"""

