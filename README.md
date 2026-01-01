# 🚀 Autonomous Product Experimentation Agent

An AI-powered autonomous experimentation system that uses multi-agent AI (CrewAI) to generate product ideas, design A/B experiments, run Bayesian statistical analysis, and make data-driven shipping decisions.

## ✨ Features

- **🤖 Multi-Agent AI System**: CrewAI agents collaborate to generate ideas, formulate hypotheses, and design experiments
- **📊 Bayesian A/B Testing**: Probabilistic inference with PyMC (with automatic fallback to analytical approximation)
- **🎯 Autonomous Decision Making**: SHIP/ROLLBACK/ITERATE decisions based on Bayesian evidence
- **📈 Interactive Dashboard**: Real-time visualization of experiment results and trends
- **🔄 End-to-End Pipeline**: From idea generation to decision-making, fully automated

## 🏗️ Architecture

```
┌─────────────────┐
│  CrewAI Agents   │  → Idea Generation, Hypothesis, Design
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Experiment     │  → User Simulation, Data Collection
│  Execution      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bayesian       │  → Probabilistic Inference & Analysis
│  Analysis        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Decision       │  → Autonomous Decision Making
│  Engine         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dashboard      │  → Streamlit Visualization
└─────────────────┘
```

## 🛠️ Tech Stack

- **Multi-Agent AI**: CrewAI, LangChain
- **Statistical Analysis**: PyMC, SciPy (Bayesian inference with automatic fallback)
- **Data Science**: NumPy
- **Visualization**: Streamlit, Matplotlib
- **Type Safety**: Pydantic
- **Testing**: pytest

## 📋 Prerequisites

- Python 3.10 or higher
- OpenAI API key (or compatible LLM provider)
- Virtual environment (recommended)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd project1
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Run an Experiment

```bash
python main.py
```

This will:
- Run the complete experimentation pipeline
- Generate product ideas using CrewAI agents
- Design and execute A/B experiments
- Perform Bayesian analysis
- Make autonomous decisions
- Log results to `memory/experiment_memory.json`

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

**Note:** Run at least one experiment before viewing the dashboard.

## 📁 Project Structure

```
project1/
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
├── pytest.ini             # Pytest configuration
├── .env                   # Environment variables (create this)
│
├── crew/                  # CrewAI multi-agent system
│   ├── agents.yaml        # Agent configurations
│   ├── tasks.yaml         # Task definitions
│   ├── agents.py          # Agent loader
│   ├── tasks.py           # Task loader
│   └── crew.py            # Crew setup
│
├── engine/                # Core experimentation engine
│   ├── bayesian.py        # Bayesian A/B testing
│   ├── simulator.py       # User behavior simulation
│   ├── decision_rule.py   # Decision logic
│   ├── memory.py          # Experiment logging
│   └── schemas.py         # Pydantic models
│
├── dashboard/             # Streamlit dashboard
│   └── app.py            # Main dashboard
│
├── tests/                 # Unit tests
│   ├── test_bayesian.py
│   ├── test_decision_rule.py
│   └── test_simulator.py
│
└── memory/               # Experiment memory storage
    └── experiment_memory.json
```

## 🔬 Key Features Explained

### Bayesian A/B Testing

Uses probabilistic inference to:
- Compute posterior distributions for conversion rates
- Provide credible intervals (not confidence intervals)
- Calculate probability that treatment > control
- No reliance on p-values or null hypothesis testing

**Automatic Fallback**: If PyMC compilation fails (common on Windows), the system automatically uses an analytical approximation with identical statistical properties.

### Autonomous Decision Making

Decision logic based on Bayesian evidence:
- **SHIP**: High confidence (P ≥ 0.95) that treatment is better
- **ROLLBACK**: Low confidence (P ≤ 0.60) that treatment is better  
- **ITERATE**: Medium confidence - needs more data or refinement

### Multi-Agent System

Five specialized AI agents:
1. **Idea Agent**: Product growth strategist
2. **Hypothesis Agent**: Causal inference analyst
3. **Design Agent**: Experiment design specialist
4. **Evaluation Agent**: Bayesian statistician
5. **Decision Agent**: Autonomous decision maker

## 📊 Dashboard Features

- **Latest Experiment Results**: Decision, posterior lift, probability metrics
- **Posterior Lift Distribution**: 95% credible interval visualization
- **Decision History**: Distribution of SHIP/ROLLBACK/ITERATE decisions
- **Experiment Trends**: Cumulative regret and lift over time
- **Experiment Details**: Full JSON view of experiment configuration

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Or run specific tests:

```bash
pytest tests/test_bayesian.py
pytest tests/test_decision_rule.py
pytest tests/test_simulator.py
```

## 🔧 Configuration

### Decision Thresholds

Modify thresholds in `engine/decision_rule.py`:

```python
SHIP_THRESHOLD = 0.95      # Probability threshold for SHIP
ROLLBACK_THRESHOLD = 0.60  # Probability threshold for ROLLBACK
```

### Agent Configuration

Customize agent behavior in:
- `crew/agents.yaml` - Agent roles, goals, and backstories
- `crew/tasks.yaml` - Task descriptions and expected outputs

### Sample Size Defaults

If sample size is not specified in experiment design, defaults to 1000 per variant (configurable in `main.py`).

## 🐛 Troubleshooting

### PyMC Compilation Issues

If you see compilation errors on Windows, don't worry! The system automatically falls back to an analytical approximation that provides identical statistical results. This is handled transparently.

### Module Not Found Errors

Make sure you're running commands from the project root directory. The dashboard automatically adds the project root to Python's path.

### Empty Dashboard

Run at least one experiment (`python main.py`) before viewing the dashboard. The dashboard reads from `memory/experiment_memory.json`.

## 📈 Example Output

```
============================================================
AUTONOMOUS EXPERIMENT RESULT
============================================================

Idea: Change CTA button color from blue to green

Hypothesis: Green CTA will increase conversion rate by 2-3%

Bayesian Analysis:
   - Posterior Lift: 1.85%
   - P(Treatment > Control): 0.8723
   - 95% Credible Interval: [0.0045, 0.0321]

Decision: ITERATE
============================================================
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built as a demonstration of autonomous experimentation systems combining:
- Multi-agent AI (CrewAI)
- Bayesian statistics (PyMC/SciPy)
- Product decision-making
- Data visualization (Streamlit)

---

**Note**: This is a demonstration project. For production use, consider additional factors around data privacy, model validation, and system reliability.
