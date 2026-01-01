from crewai import Crew
from crew.tasks import load_tasks
from crew.agents import load_agents

agents = load_agents()
tasks = load_tasks()

experiment_crew = Crew(
    agents=list(agents.values()),
    tasks=tasks,
    verbose=True
)
