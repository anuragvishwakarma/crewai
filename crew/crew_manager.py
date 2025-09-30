# crew/crew_manager.py
from crewai import Crew
from langchain_ollama import ChatOllama
from .agent import maintenance_scheduler, field_support_agent, workload_manager
from .tasks import create_scheduling_task, create_support_task, create_workload_task

def route_query(query: str) -> str:
    llm = ChatOllama(model="llama3:8b", temperature=0.0)
    prompt = f"""Classify into one word: scheduling, support, or workload.
    Query: "{query}"
    Answer:"""
    response = llm.invoke(prompt)
    agent = response.content.strip().lower()
    return agent if agent in ["scheduling", "support", "workload"] else "support"

def run_crew(query: str, history: list = None) -> dict:
    # Inject chat history into query for context
    if history:
        context = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history])
        query = f"Previous conversation:\n{context}\n\nCurrent question: {query}"

    agent_type = route_query(query)
    
    if agent_type == "scheduling":
        task = create_scheduling_task(query)
        agent = maintenance_scheduler
    elif agent_type == "workload":
        task = create_workload_task(query)
        agent = workload_manager
    else:
        task = create_support_task(query)
        agent = field_support_agent

    crew = Crew(agents=[agent], tasks=[task], verbose=0)
    result = crew.kickoff()
    
    return {
        "response": result,
        "agent_used": agent_type
    }