import base64
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
# Load variables from .env file
load_dotenv()

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

APP_TITLE = "Talk to Tabular Data"
SVG_ICON_PATH = 'resources/static/cohorte_logo_content.svg'

# Read the SVG file and encode it
with open(SVG_ICON_PATH, 'rb') as file:
    data = file.read()
    b64 = base64.b64encode(data).decode("utf-8")
    image_html = f'<img src="data:image/svg+xml;base64,{b64}" style="width:50px; height:auto;"/>'

# Combine the image HTML and the Title in Markdown
combined_html_title = f"""
<div style='display: flex; align-items: center;'>
    {image_html}
    <h1 style='margin-left: 20px;'>{APP_TITLE}</h1>
</div>
"""


def load_css(file_name):
    with open(file_name) as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


def main():
    load_css('style.css')

    # Display the combined SVG and Title
    st.markdown(combined_html_title, unsafe_allow_html=True)

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file")

    df = None
    if uploaded_file is not None:
        # To read file as CSV (if applicable):
        if uploaded_file.type == "text/csv":
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            st.write("DataFrame (upto 1000 rows):")
            st.write(df.head(1000))

    # User input for query text
    query_text = st.text_area("Enter your query:", height=50)

    if st.button("Submit"):
        assert df.shape[0] > 0
        agent_executor = create_pandas_dataframe_agent(llm, df, verbose=True, handle_parsing_errors=True)
        prompt = f"""
        While trying to answer, first try to import the libraries needed and then perform the task. 
        The commands that you will run should be minimal and accurate. It should match with the given CSV file contents. 
        Final Answer should utilize bulleted lists for clarity and organization, whenever possible, 
        instead of consolidating all information into a single paragraph.

        Query:
        {query_text}
        """
        response = agent_executor.invoke(prompt)
        st.write(response["output"])


if __name__ == "__main__":
    main()
