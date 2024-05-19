import streamlit as st

# import custom functions
from utils import *

st.set_page_config(
    page_title="Portfolio Project: Salifort Motors🏍️",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/nicolo-pedrani/',
        'Report a bug': "https://github.com/nicolopedrani",
        'About': "# Portfolio Project from Google Advanced Data Analytics Professional Certificate"
    }
)

load_dataset()

# Title and Text Block
st.title("Employee Turnover Analysis at Salifort Motors")

st.markdown("""
The following portfolio project is based on PACE Workflow, a structured approach designed to efficiently manage projects.   
PACE stands for:

1. **Plan**: This stage involves defining project goals, scope, and resources. It's crucial to identify stakeholders, gather requirements, and establish clear deliverables.
2. **Analyze**: Here, we delve deeper into the project. Data is analyzed to understand trends, identify potential challenges, and inform decision-making.
3. **Construct**: This is the building phase. Based on the plan and analysis, we develop the project components, whether it's a data pipeline, a machine learning model, or a web application like this one.
4. **Execute**: Finally, the project is deployed and launched. This stage involves monitoring performance, addressing issues, and iterating based on feedback.  
            
The PACE Workflow is iterative, meaning you can revisit any stage as needed throughout the project lifecycle. This flexibility allows for course correction and ensures the project remains aligned with its goals.

### Benefits of using PACE Workflow:

* Increased Efficiency: By following a structured approach, you avoid wasting time and resources on ad-hoc processes.
* Improved Communication: Clear communication between stakeholders is fostered by defining goals and expectations upfront.
* Reduced Risk: Potential issues are identified early through analysis, enabling proactive mitigation strategies.
* Enhanced Project Success: The PACE Workflow provides a roadmap for project execution, increasing the likelihood of achieving desired outcomes.
This portfolio project leverages the PACE Workflow to guide you through your project.  Let's get started!
            """)

col1, col2, col3 = st.columns(3)

with col2:
    st.header("Menu")
    st.page_link("home🏠.py", label="Home", icon="🏠")
    st.page_link("pages/1_Plan💭.py", label="Plan", icon="💭")
    st.page_link("pages/2_Analyze📊.py", label="Analyze", icon="📊")
    st.page_link("pages/3_Construct📈.py", label="Construct", icon="📈")
    st.page_link("pages/4_Execute🗒️.py", label="Execute", icon="🗒️")
    st.page_link("pages/5_ExecutiveSummary📝.py", label="ExecutiveSummary", icon="📝")

