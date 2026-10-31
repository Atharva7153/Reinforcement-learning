# 🚀 Step-by-Step Reinforcement Learning

A hands-on journey building Reinforcement Learning fundamentals from scratch in pure Python — progressing from basic Multi-Armed Bandits to 2D GridWorld Q-Learning, with the ultimate goal of landing rockets using Deep RL.

---

## 🗺️ Learning Roadmap & Progression

| Step / Project | Focus | Core Concept | Key Formula / Logic |
| :--- | :--- | :--- | :--- |
| **[Step 1](step1.ipynb)** | Environment Setup | Multi-Armed Bandit & Random Sampling | `action = random.choice(actions)` |
| **[Step 2](step2.ipynb)** | Agent Memory | Sample-Average Value Estimation | $Q(a) = \frac{\sum R_a}{N(a)}$ |
| **[Step 3](step3.ipynb)** | Policy & Balance | $\epsilon$-Greedy (Exploration vs. Exploitation) | Explore with prob $\epsilon$, else $\arg\max Q(a)$ |
| **[Step 4](step4.ipynb)** | Incremental Learning | Learning Rate ($\alpha$) & Prediction Error | $Q(a) \leftarrow Q(a) + \alpha [R - Q(a)]$ |
| **[Step 5](step5.ipynb)** | Markov Decision Processes | States, Episodes & Discount Factor ($\gamma$) | $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ |
| **[Step 6](step6.ipynb)** | 1D Tabular Q-Learning | Temporal Difference & Bellman Equation | $Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| **[Mini-Project](proj.py)** | 2D GridWorld Navigation | 2D State Space & 4-Direction Control | $(r, c) \in \{0,1,2\}^2$, Actions: `['left', 'right', 'up', 'down']` |

---

## 📌 Breakdown of Steps & Projects

### 1. [Multi-Armed Bandit Basics (`step1.ipynb`)](step1.ipynb)
- Defined a 2-armed slot machine environment (`'left'` with 20% win-rate, `'right'` with 80% win-rate).
- Observed the baseline performance of purely random actions without memory or learning.

### 2. [Giving the Agent Memory (`step2.ipynb`)](step2.ipynb)
- Introduced `reward_count` and `action_count` dictionaries.
- Estimated action values using sample averages: after enough trials, the agent's estimates converged near the true probabilities (`~0.2` and `~0.8`).

### 3. [$\epsilon$-Greedy Strategy (`step3.ipynb`)](step3.ipynb)
- Solved the **Exploration vs. Exploitation** dilemma:
  - **Exploration ($\epsilon = 0.1$)**: 10% chance to pick randomly and discover rewards.
  - **Exploitation ($1 - \epsilon = 0.9$)**: 90% chance to pick the highest estimated action.
- Added **The 3-Box Reward Game Challenge** (testing 3 actions with 10%, 50%, and 80% odds).

### 4. [Incremental Updates & Learning Rate (`step4.ipynb`)](step4.ipynb)
- Eliminated history storage (`reward_count` dictionary removed).
- Updated estimates on the fly using **Surprise / Prediction Error**:
  $$\text{NewEstimate} = \text{OldEstimate} + \alpha \times (\text{Reward} - \text{OldEstimate})$$
- Controlled update magnitude using learning rate $\alpha = 0.1$.

### 5. [States & The Discount Factor (`step5.ipynb`)](step5.ipynb)
- Transitioned from stateless bandits to **State-based worlds**: 1D Grid `[0] -> [1] -> [2] -> [3 (Goal)]`.
- Implemented **Discounted Return ($G_t$)** to balance immediate step costs (`-1`) with long-term goal rewards (`+10`).
- Demonstrated why low $\gamma$ (e.g. `0.2`) makes an agent short-sighted, while high $\gamma$ (e.g. `0.9`) encourages reaching future goals.

### 6. [1D Tabular Q-Learning (`step6.ipynb`)](step6.ipynb)
- Implemented full **Tabular Q-Learning** across multiple states.
- Propagated future values back to earlier states using the Bellman target:
  $$\text{Target} = R + \gamma \max_{a'} Q(s', a')$$
- Tested the trained agent under pure exploitation ($\epsilon = 0$), verifying it takes the optimal shortest path straight to the goal (`0 -> 1 -> 2 -> 3`).

### 7. [Mini-Project: 2D GridWorld Navigation (`proj.py`)](proj.py)
- Scaled Q-Learning to a **3x3 2D Coordinate Grid** with 9 distinct states $(r, c)$ from `(0, 0)` to `(2, 2)`.
- Expanded the action space to 4 directional moves: `['left', 'right', 'up', 'down']`.
- **Environment Dynamics**:
  - Start state: `(0, 0)`.
  - Goal state: `(2, 2)` with a terminal reward of `+100` (`done = True`).
  - Step penalty: `-1` per move to encourage taking the shortest path.
  - Boundaries: Walking out of grid bounds resets position back to `(0, 0)`.
- Learned optimal policies where states near the goal (`(1, 2)` pointing `up` and `(2, 1)` pointing `right`) converge to high Q-values approaching `~100`.

---

## 🛠️ How to Run

Clone the repository and run any notebook using Jupyter:

```bash
jupyter notebook
```

Or run standalone Python experiment scripts:

```bash
# Run 1D Q-learning experiment
python3 test2.py

# Run 2D GridWorld project
python3 proj.py
```

*Zero heavy dependencies required for foundation steps — standard library Python only.*
