# LuxAI-agents
Agents and supporting code for the Kaggle Lux AI challenge (Season 1)

## TLDR

```python
from kaggle_environments import make
import sys

sys.path.append('/Users/jaime/Documents/MachineLearning/LuxAI/LuxAI-agents/')
from agent_loader import AgentLoader
from agents.rl_agents import RLAgent

my_agent = AgentLoader(agent_class=RLAgent).game_loop
env = make("lux_ai_2021")
steps = env.run([my_agent, "simple_agent"])
env.render(mode="ipython", width=1200, height=800)
```

See example in `notebooks/Training.ipynb`.

---

The `lux` directory is a copy of [LuxAI's python kit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits/python/simple/lux) @ [fd1de7aad8ee890dd88b4f384039acc347da937f](https://github.com/Lux-AI-Challenge/Lux-Design-2021/commit/fd1de7aad8ee890dd88b4f384039acc347da937f).
