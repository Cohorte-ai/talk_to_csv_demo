import pandas as pd
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

csv_path = "data/raw/customers-10000.csv"
df = pd.read_csv(csv_path)

print(df.shape)

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

agent_executor = create_pandas_dataframe_agent(llm, df, verbose=True)

# Find the number of unique cities
unique_cities_count = len(df["City"].unique())
print(unique_cities_count)
_ = agent_executor.invoke("Find the number of unique cities")
print(f"received: {_}")

# Find the number of unique countries
unique_countries_count = len(df["Country"].unique())
print(unique_countries_count)
_ = agent_executor.invoke("Find the number of unique Countries")
print(f"received: {_}")

# Find the number of unique countries
countries_value_count = df["Country"].value_counts()
print(countries_value_count)
_ = agent_executor.invoke("Find the number of row of each Country")
print(f"received: {_}")
