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
