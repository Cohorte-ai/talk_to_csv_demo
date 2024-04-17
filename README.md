# Talk to Tabular Data Web Application README

## Demo Link: https://drive.google.com/file/d/1YrNTULdYUUB4FHZN4oVVi3lqU0d8gL2U/view?usp=sharing

## Overview

This README document provides guidance on setting up and running the "Talk to Tabular Data" web application. The application utilizes Streamlit for web deployment and leverages OpenAI's GPT-4 to interact with and analyze tabular data uploaded by users.

## Theoretical Concepts

### Streamlit

Streamlit is a Python library that accelerates the development of data applications by converting Python scripts into interactive web apps.

### OpenAI GPT-4

OpenAI's GPT-4 is a cutting-edge language model capable of performing diverse language tasks. In this application, it processes natural language queries to generate insights from data.

### Langchain

Langchain is designed for building applications that integrate language models with structured data sources. This application employs Langchain to create agents facilitating data interactions.

### Pandas

Pandas is a Python library used for data manipulation and analysis, which handles the operations on CSV data uploaded by users.

### Environment Variables with dotenv

Environment variables are managed using the `python-dotenv` library, which reads variables from a `.env` file, enhancing security and configuration management.

## Installation

To install and run this application:

1. **Clone the Repository**:
git clone [repository URL]
cd [repository-directory]
2. **Set Up a Virtual Environment** (recommended):
python -m venv venv
source venv/bin/activate # For Windows use venv\Scripts\activate
3. **Install Dependencies**:
pip install -r requirements.txt
4. **Environment Setup**:
Create a `.env` file in the project's root directory and populate it with necessary variables like the OpenAI API key.
5. **Launch the Application**:
streamlit run app.py

## Usage

- **Upload CSV**: Use the file uploader for your data.
- **Enter Query**: Type your query in the provided text area.
- **Submit**: Click 'Submit' to process your query and view results below the button.

## Key Function: `create_pandas_dataframe_agent`
`create_pandas_dataframe_agent` from Langchain allows dynamic interaction with data through natural language by integrating a dataframe with an LLM, which interprets queries and executes Python code.
### Features
- **LLM Integration**: Uses models like GPT-4 for query interpretation.
- **Dataframe Interaction**: Operates on Pandas dataframes for data manipulation.
- **Agent Customization**: Supports various agent types for different interaction styles and capabilities.
- **Verbose Debugging**: Provides detailed logging options.
- **Configuration Flexibility**: Offers extensive parameters to customize agent behavior.

## Additional Notes

- Ensure no sensitive data is uploaded as it is processed by OpenAI's models.
- Primarily supports basic CSV files; complex data structures may need further customization.

This setup guide facilitates running the application locally, emphasizing the use of `create_pandas_dataframe_agent` for structured data interaction.

