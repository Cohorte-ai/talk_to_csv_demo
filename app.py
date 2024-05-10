import base64
import streamlit as st
import streamlit.components.v1 as components
import openai

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
os.makedirs('data/artifacts/', exist_ok=True)

from dotenv import load_dotenv
# Load variables from .env file
load_dotenv()

llm_gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0)
llm_gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

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

# Inject custom CSS to set the width of the sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 50% !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_css(file_name):
    with open(file_name) as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


def main():
    load_css('style.css')

    # Display the combined SVG and Title
    st.markdown(combined_html_title, unsafe_allow_html=True)
    with st.sidebar:
        # File uploader widget
        uploaded_file = st.file_uploader("Choose a CSV file")

        df = None
        if uploaded_file is not None:
            # To read file as CSV (if applicable):
            if uploaded_file.type == "text/csv":
                import pandas as pd
                df = pd.read_csv(uploaded_file)
                st.write("DataFrame (upto 40 rows):")
                st.write(df.head(40))

    if df is not None and df.shape[0] > 0:
        df_sample = df.head(1)
        # Summarize the DataFrame content into a prompt
        summary_prompt = "Based on the data: " + ", ".join(
            [f"{col} includes {df_sample[col].unique().tolist()}" for col in df_sample.columns])

        # Append a request for generating questions
        suggested_ques_prompt = (f"Given the information that {summary_prompt}, "
                                 f"what are three relevant questions one might ask about this data?"
                                 f"Don't end the answer prematurely. give bulleted list.")

        # Make a call to OpenAI API to generate questions
        res = llm_gpt35.predict(suggested_ques_prompt)
        # breakpoint()
        # Print the generated questions
        st.write(f"possible questions to ask:\n{res}")
        # User input for query text
        query_text = st.text_area("Enter your query:", height=50)

    if df is not None and df.shape[0] > 0 and st.button("Submit"):
        with st.spinner("Invoking Agent:"):
            assert df.shape[0] > 0
            agent_executor = create_pandas_dataframe_agent(llm_gpt4, df, verbose=True, handle_parsing_errors=True)
            prompt_text = f"""
                Begin by immediately importing all necessary libraries, as this is a mandatory first step to ensure that all required functionalities are available. 
                Failure to import necessary libraries first may result in incomplete or erroneous task execution.

                Once the libraries are imported, execute only the essential commands that directly correspond with the contents of the provided CSV file.

                If applicable, generate a Plotly figure object but don't show rather it must be saved at the location 'data/artifacts/visual_plotly_obj_<some_uuid>', where '<some_uuid>' is a unique identifier for each session. 

                Construct the final answer as a dictionary that must strictly adhere to the following structure. 
                I have added datatypes of them as well.
                - "output_text": 
                    - <string datatype value answering the query>
                    - This key should include all textual information/explanation, organized in bulleted lists to ensure clarity and ease of reading, assuming no Plotly figure was created.
                - "fig_html": 
                    - "<string datatype value mentioning the file path>"
                    - This key should contain the absolute file path of the Plotly chart figure HTML object, formatted as 'data/artifacts/visual_plotly_obj_<some_uuid>', or None if no figure was generated.

                It is crucial that the final dictionary is properly formatted to ensure it can be loaded as a JSON object without any errors.
                It needs to be RFC8259 compliant JSON. 
                This format is mandatory and required for successful data processing.
                
                Final Answer should also have key `explanations` containing all thought, action and action input of ReAct framework. 
                
                Query:
                {query_text}
            """

            response = agent_executor.invoke(
                prompt_text,
                # include_run_info=True,
                # return_only_outputs=False
            )
            # breakpoint()
            if isinstance(response["output"], str):
                try:
                    ans = response["output"]
                    from ast import literal_eval
                    response["output"] = literal_eval(ans.strip())
                except Exception as e:
                    print(f"Exception occurred: \n {str(e)}\n\n")
                    response["output"] = ans
            output_result = response["output"]

            if isinstance(output_result, dict):
                print(f"output_result:\n{type(output_result)}\n{output_result}")
                if "output_text" in output_result:
                    st.write(output_result["output_text"])
                if (
                        "fig_html" in output_result
                        and
                        (
                                not output_result["fig_html"]
                                or
                                not str(output_result["fig_html"]).strip().lower() == "none"
                        )
                ):
                        print(f"fig_html path: {output_result['fig_html']}")
                        components.html(
                            open(output_result["fig_html"],  'r', encoding='utf-8', ).read(),
                            height=550,
                            scrolling=True
                        )
                if (
                        "explanations" in output_result
                        and
                        len(str(output_result["explanations"])) > 1
                ):
                    with st.expander("See explanation"):
                        st.write(output_result["explanations"])
            else:
                st.write(output_result)
            print("Done!!!")


if __name__ == "__main__":
    main()
