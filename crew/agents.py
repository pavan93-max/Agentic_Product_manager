import yaml
from crewai import Agent

def load_agents():
    with open("crew/agents.yaml", "r") as f:
        config = yaml.safe_load(f)

    agents = {}
    for name, cfg in config.items():
        agents[name] = Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            verbose=True
        )
    return agents
