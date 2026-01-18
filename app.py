import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

@tool
def destination_search(destination: str) -> str:
    """Returns popular attractions and experiences for a destination."""
    return f"""
    Popular things to do in {destination}:
    - Famous landmarks and sightseeing
    - Local street food and restaurants
    - Cultural and historical attractions
    - Nature spots and leisure activities
    """


@tool
def budget_estimator(days: int, total_budget: int) -> str:
    """Estimates per-day budget."""
    daily_budget = total_budget // days
    return f"Estimated daily budget: ₹{daily_budget}"


tools = [destination_search, budget_estimator]

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

st.set_page_config(
    page_title="AI Travel Planner ✈️",
    page_icon="🧭"
)

st.title("🧭 AI-Powered Travel Planning Agent")
st.write("Describe your trip in natural language and get a complete itinerary.")

user_input = st.text_area(
    "Tell me about your trip:",
    placeholder=(
        "Example:\n"
        "I want a 4-day budget trip to Jaipur under ₹15,000.\n"
        "I enjoy history, forts, and local food."
    ),
    height=150
)

if st.button("Generate Itinerary"):
    if not user_input.strip():
        st.warning("Please enter your trip details.")
    else:
        with st.spinner("✈️ Planning your trip..."):
            response = agent.run(
                f"""
                You are an AI travel planning agent.
                Create a structured travel itinerary in this format:

                Destination:
                Duration:
                Travel Style:
                Estimated Daily Budget:

                Day-wise Itinerary:
                Day 1:
                - Morning:
                - Afternoon:
                - Evening:

                Day 2:
                ...

                User Request:
                {user_input}
                """
            )

        st.success("🎉 Your itinerary is ready!")
        st.markdown("## 🗺️ Travel Itinerary")
        st.write(response)

st.markdown("---")
st.caption("Built using LangChain Agents, Tool Calling, Memory, and Streamlit")
