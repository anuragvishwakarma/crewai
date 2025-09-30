# crew/tasks.py
from crewai import Task
from .agent import maintenance_scheduler, field_support_agent, workload_manager

def create_scheduling_task(query: str):
    return Task(
        description=f"Analyze this request and recommend a maintenance schedule: {query}",
        expected_output='JSON with equipment_id, recommended_date (YYYY-MM-DD), and reason',
        agent=maintenance_scheduler
    )

def create_support_task(query: str):
    return Task(
        description=f"Answer the fieldworker's question using available knowledge: {query}",
        expected_output="Clear, step-by-step answer. If unsure: 'I don't know — consult the manual or supervisor.'",
        agent=field_support_agent
    )

def create_workload_task(query: str):
    return Task(
        description=f"Assess workload and suggest task reassignments: {query}",
        expected_output='JSON with "reassignments" list containing worker_id, task_id, reason',
        agent=workload_manager
    )