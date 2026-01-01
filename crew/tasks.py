import yaml
from crewai import Task
from crew.agents import load_agents

agents = load_agents()

def load_tasks():
    with open("crew/tasks.yaml", "r") as f:
        config = yaml.safe_load(f)

    tasks = []
    for cfg in config.values():
        tasks.append(
            Task(
                description=cfg["description"],
                expected_output=cfg["expected_output"],
                agent=agents[cfg["agent"]]
            )
        )
    return tasks
