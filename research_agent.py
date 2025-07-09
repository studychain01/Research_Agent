# ================================
# 📦 Module Significance Overview
# ================================

# streamlit: Used to build the frontend/UI for the app (buttons, input, layout, etc.)
# os: Enables interaction with the operating system, especially to read environment variables like API keys.
# uuid: Generates unique identifiers (e.g., for session tracking or trace IDs).
# asyncio: Supports asynchronous execution, allowing multiple tasks (like agent calls) to run without blocking.
# datetime: Provides utilities to work with date and time (e.g., saving timestamps for collected facts).
# load_dotenv: Loads environment variables from a .env file into os.environ for secure configuration.
# BaseModel (from pydantic): Defines structured, validated data models (e.g., for ResearchPlan and ResearchReport).
# agents (custom module):
#   - Agent: Represents an LLM-based agent with instructions, tools, and output behavior.
#   - Runner: Handles execution of agents with input prompts and tracks output.
#   - WebSearchTool: A built-in tool for live web searching via the agent.
#   - function_tool: Decorator to turn a Python function into a callable agent tool.
#   - handoff: Transfers the task from one agent to another in a workflow.
#   - trace: Captures logs and traces of the agent's behavior for debugging or visualization.

import os 
import uuid
import asyncio 
import streamlit as st 
from datetime import datetime 
from dotenv import load_dotenv

from agents import(
    Agent, 
    Runner, 
    WebSearchTool, 
    function_tool, 
    handoff,
    trace,
)

from pydantic import BaseModel 

load_dotenv()

#Set up page configuration 
st.set_page_config(
    page_title="OpenAI Researcher Agent",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

#API Key 
if not os.environ.get("OPENAI_API_KEY"):
    st.error("Please set your OPENAI_API_KEY environment variable")
    st.stop()

st.title("📰 OpenAI Researcher Agent")
st.subheader("Powered by OpenAI Agents SDK")
st.markdown("""
This app demonstrates the power of OpenAI's Agents SDK by creating a multi-agent system 
that researches news topics and generates comprehensive research reports.
""")

#Define data models 
class ResearchPlan(BaseModel):
    topic: str
    search_queries: list[str]
    focus_areas: list[str]

class ResearchReport(BaseModel):
    title: str
    outline: list[str]
    report: str
    sources: list[str]
    word_count: int


#Custom tool for saving facts found during research 
@function_tool
def save_important_fact(fact: str, source: str = None) -> str:
    """ Save an important fact discovered during research. 

    Args: 
        fact: The important fact to save. 
        source: Optional source of the fact

    Returns: 
        Confirmation message. 
    """

    if "collected_facts" not in st.session_state:
        st.session_state.collected_facts = []

    st.session_state.collected_facts.append({
        "fact": fact, 
        "source": source,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

    return f"Fact saved: {fact}"


research_agent = Agent(
    name="Research Agent",
    instructions="You are a research assistant. Given a search term, you search the web for that term and"
    "produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300"
    "words. Capture the main points. Write succintly, no need to have complete sentences or good"
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the"
    "essence and ignore any fluff. Do not include any additional commentary other than the summary"
    "itself.",
    model="gpt-4o-mini",
    tools=[
        WebSearchTool(),
        save_important_fact
    ],
)

editor_agent = Agent(
    name="Editor Agent",
    handoff_description="A senior researcher who writes comprehensive research reports.",
    instructions="You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research "
    "assistant.\n"

    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim"
    "for 5-10 pages of content, at least 1000 words.",
    model="gpt-4o-mini",
    output_type=ResearchReport,
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="""You are the coordinator of this research operation. Your job is to:
    1. Understand the user's research topic
    2. Create a research plan with the following elements:
       - topic: A clear statement of the research topic
       - search_queries: A list of 3-5 specific search queries that will help gather information
       - focus_areas: A list of 3-5 key aspects of the topic to investigate
    3. Hand off to the Research Agent to collect information
    4. After research is complete, hand off to the Editor Agent who will write a comprehensive report
    
    Make sure to return your plan in the expected structured format with topic, search_queries, and focus_areas.
    """,
    handofs=[
        handoff(research_agent),
        handoff(editor_agent)
    ],
    model="gpt-4o-mini",
    output_type=ResearchPlan,
)

