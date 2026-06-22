import os
import requests
import streamlit as st
import certifi
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub

from langchain_community.tools.tavily_search import TavilySearchResults

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()


GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY= os.getenv("WEATHERSTACK_API_KEY")

#Streamlit Page Config
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain")

#Search Tool

search_tool = TavilySearchResults(max_results=2)

#Weather Tool
@tool
def getWeather_data(city: str) -> str:
    
    """
    Get weather data for a given city.
    """
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response= requests.get(url)

    data=response.json()

    if "current" not in data:
        return "City not found or weather data unavailable."
    
    return (
        f"Weather in {city}: \n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
        )


# result= search_tool.invoke("Give me the latest news on AI")
# result

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# response = llm.invoke("Day today")
# response

#Prompt
prompt= hub.pull("hwchase17/react")

tools = [search_tool,
    getWeather_data]

#Create Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt

)

#Create Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) 

#UI Input

user_query= st.text_input("Enter your query here:",
                          placeholder="Find the capital of Jharkhand and the current weather")


#Execution
# ==========================================
# RUN AGENT
# ==========================================

if st.button("Run Agent"): 

    if user_query:

        with st.spinner("Agent is thinking..."):

            try:
                response = agent_executor.invoke({
                    "input": user_query
                })

                st.success("Response Generated")

                st.markdown("## Final Response")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a query")