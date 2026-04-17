# README: Adapting VAPO-Style Value-Based RL to LLM-Driven HVAC Setpoint Control

## 1. Background and Goal

We are developing an LLM-based HVAC controller for a multi-zone building simulation environment (EnergyPlus-based), where the model directly outputs structured cooling setpoints:

```json
{"setpoints": [8 floats]}
```

The current challenge is not merely whether an LLM can produce valid actions, but whether the training signal is sufficiently dense and well-assigned to enable the policy to systematically improve beyond strong PPO baselines.

Our recent discussions suggest that the main bottleneck is **credit assignment**, rather than model size alone. In previous day-level GRPO-style training, a whole-day rollout often produced only one coarse update signal, which was then implicitly shared across many decisions within the day. This is likely too weak for a control problem with multiple blocks and many setpoint decisions.

This README proposes a new execution plan inspired by **VAPO** (Value-based Augmented PPO), but adapted to the structure of HVAC control rather than long chain-of-thought language reasoning.

---

## 2. Why VAPO Is Relevant, and Why It Cannot Be Copied Directly

### 2.1 What is useful from VAPO

VAPO is valuable because it argues that value-based RL can outperform critic-free group-relative methods when the following issues are handled properly:

- value initialization bias,
- sparse rewards,
- long-horizon credit assignment,
- instability caused by noisy policy updates.

The most useful VAPO ideas for our HVAC setting are:

1. **Value Pretraining**  
   Pretrain the value model before relying on it for policy updates.

2. **Decoupled Critic and Policy Logic**  
   The critic may need a different target construction and update schedule than the policy.

3. **Group Sampling Still Has Value**  
   Relative comparison among multiple sampled trajectories can still be useful, but it should not be the only baseline.

### 2.2 What should not be copied mechanically

Our task is fundamentally different from long-CoT reasoning:

- the state is structured numerical control state, not hidden language state;
- the reward is physically grounded and directly computable;
- the action is a structured setpoint vector, not free-form text reasoning;
- the horizon is long, but temporally regular and externally segmented.

Therefore, we **should not directly copy** the following VAPO mechanisms without adaptation:

- token-length tricks designed for variable-length reasoning traces,
- language-side value modeling over token hidden states,
- sequence-length-adaptive rules meant for CoT generation.

Instead, we should translate the VAPO principle into a **numeric block-value critic** suitable for control.

---

## 3. Core Design Decision

### 3.1 Actor

The actor remains an LLM policy that outputs direct zone setpoints.

Example output:

```json
{"setpoints": [24.3, 24.1, 24.8, 24.6, 23.9, 24.0, 24.5, 24.4]}
```

The LLM is used because it can:

- interpret forecast summaries,
- integrate semantic context across zones and time,
- operate as a structured planner under low-frequency control.

### 3.2 Critic

We introduce a **small numerical critic network**, not a second LLM.

The critic does **not** score tokens.  
It scores **control states**.

The recommended first version is:

\[
V_\phi(s_b)
\]

where \(s_b\) is the state at the start of block \(b\).

This means the critic estimates the expected future **relative return from the current block onward**.

### 3.3 Why choose `V_block` first

We considered three candidates:

1. **`V_day(s_day_start)`**
2. **`V_block(s_block_start)`**
3. **`Q(s, a)`**

We recommend starting with **`V_block`** because:

- it is finer than `V_day`,
- it is more stable and easier than `Q(s, a)`,
- it aligns naturally with our control decomposition,
- it directly targets the current credit assignment problem.

---

## 4. MDP Abstraction for the New Method

We define a day as a sequence of blocks:

\[
b = 0, 1, \dots, B-1
\]

Each block contains one or more setpoint decisions (or a short setpoint plan).

### 4.1 State

At the start of block \(b\), the critic receives a structured numerical state:

- zone dry-bulb temperatures,
- zone PMV values,
- humidity,
- occupancy,
- current PV / net grid,
- outdoor weather,
- short-horizon forecast summaries,
- block index / time-of-day,
- optional previous setpoint summary.

### 4.2 Action

The actor outputs zone-wise setpoints.

### 4.3 Reward

We keep the existing physically grounded reward:

\[
r_t = -0.01 \cdot \Big( \text{net\_grid\_kwh}_t + 50 \cdot \sum_z \text{occupied\_pmv\_violation}_{z,t} \Big)
\]

### 4.4 Baseline-relative return

We continue using a fixed baseline controller (for example, constant 24.0\(^\circ\)C) and compute relative block return:

\[
R_b^{rel} = R_b^{cand} - R_b^{base}
\]

The future return from block \(b\) is:

\[
G_b = \sum_{j=b}^{B-1} R_j^{rel}
\]

The critic learns:

\[
V_\phi(s_b) \approx \mathbb{E}[G_b \mid s_b]
\]

---

## 5. How Actor and Critic Collaborate

This is the central mechanism.

### 5.1 Critic target

For each sampled trajectory, after the day rollout is complete, we compute:

- relative block rewards \(R_b^{rel}\),
- backward cumulative block returns \(G_b\).

These become the regression targets for the critic.

### 5.2 Advantage

For each block in a trajectory:

\[
\tilde{A}_b = G_b - V_\phi(s_b)
\]

This is the critic-based advantage.

### 5.3 Group normalization

If we sample \(K\) trajectories from the same day-start state, we further normalize the block-level advantages within the group:

\[
A_b^{(i)} = \frac{\tilde{A}_b^{(i)} - \mu_b}{\sigma_b + \epsilon}
\]

where:

- \(i\) indexes the sampled trajectory,
- \(\mu_b\) and \(\sigma_b\) are computed across the \(K\) trajectories for the same block.

This means we combine:

- **value-based variance reduction**, and
- **group-relative ranking**.

### 5.4 Credit assignment to the actor

The block-level advantage is assigned to the action representation for that block.

Important design choice:

- We **do not** let the critic score individual tokens.
- We treat one block action (or one knot JSON action) as the RL action unit.
- All tokens corresponding to that structured action share the same environment-level advantage.

This keeps the language interface while preserving physically meaningful credit assignment.

---

## 6. Recommended Training Procedure

## Stage 0: Freeze actor, pretrain critic

Before joint training, collect rollouts from:

- current LLM policy,
- baseline controller,
- optionally PPO checkpoints,
- optionally random or perturbed policies for diversity.

Construct a dataset of:

\[
(s_b, G_b)
\]

Train the critic with Huber loss:

\[
\mathcal{L}_{critic} = \text{Huber}(V_\phi(s_b), G_b)
\]

This is the HVAC analogue of VAPO-style **Value Pretraining**.

### Why this matters

If the critic is randomly initialized and immediately used online, it may inject severe bias into the policy update. Pretraining reduces this initialization risk.

---

## Stage 1: Online grouped rollout

For each sampled day-start state:

1. Sample \(K=3\) trajectories from the actor.
2. Roll each trajectory through the full day.
3. Slice each day into blocks.
4. Compute relative block rewards and cumulative returns.

---

## Stage 2: Critic update

Use the observed \(G_b\) values to update the critic.

Optionally:

- use replay buffer for critic stability,
- use target network later if needed,
- keep critic updates more frequent than actor updates.

But for the first version, a simple supervised regression critic is sufficient.

---

## Stage 3: Actor update

Compute critic-based group-normalized advantages.

Use a PPO-style clipped objective on the tokens that generated each structured action:

\[
\mathcal{L}_{actor} = - \min \Big( r_t A_b, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_b \Big)
\]

where \(A_b\) is the block-level advantage associated with the corresponding action block.

The important point is:

- clipping remains useful,
- the baseline is no longer only group mean,
- the value estimate provides more local and lower-variance feedback.

---

## 7. Critic Input Specification (First Version)

Recommended critic input vector:

### Zone-level features (for each of 8 zones)

- current dry-bulb temperature,
- current PMV,
- current humidity,
- occupancy,
- optional last applied setpoint.

### Global features

- current outdoor temperature,
- outdoor humidity,
- cloud cover,
- precipitation probability,
- current PV,
- current net grid,
- time-of-day encoding,
- block index,
- day index or date embedding if needed.

### Forecast summary features

We recommend summary statistics rather than long raw vectors in critic v1:

- next 2h temperature mean,
- next 2h cloudcover max,
- next 2h precipitation probability max,
- next 2h PV risk indicator,
- next 6h temperature trend.

The actor may still read richer text/structured forecast; the critic should stay compact and numerical.

---

## 8. Why Not Start with Q(s, a)

A Q-function is attractive in principle, but we do not recommend it as the first implementation.

Reasons:

1. the action is 8-dimensional continuous setpoint output,
2. the actor is an LLM, not a conventional continuous policy network,
3. learning a reliable `Q(s, a)` from sparse trajectory samples is harder than learning `V_block(s)`.

After `V_block` is working, we can later consider:

- `Q(s, a)` for block-level actions,
- twin-Q critics,
- a hybrid design where the LLM becomes high-level planner and a low-level continuous actor handles execution.

---

## 9. Minimal Execution Plan

## Phase A: Critic pretraining

- collect a block-level dataset from existing Houston / Miami rollouts,
- build `(state_block_start, return_from_block)` dataset,
- train `V_block` until value loss stabilizes and explained variance becomes meaningfully positive.

## Phase B: Joint training

- keep current setpoint-only LLM actor,
- replace pure day-level GRPO with block-value-based actor update,
- sample 3 trajectories per day-start state,
- update critic and actor jointly.

## Phase C: Evaluation

Compare against:

1. current day-level GRPO baseline,
2. PPO baseline,
3. baseline fixed setpoint controller,
4. ablation without critic,
5. ablation without group normalization.

---

## 10. Ablation Plan

To verify that the critic is genuinely useful, run the following ablations:

1. **Pure Day-GRPO**  
   Current baseline.

2. **Day critic only (`V_day`)**  
   Tests whether a coarse value baseline already helps.

3. **Block critic (`V_block`)**  
   Main proposed method.

4. **Block critic without group normalization**  
   Tests whether value alone is enough.

5. **Block critic with group normalization**  
   Full proposed method.

6. **Macro-hint on/off**  
   Tests whether the method still works without exploration scaffolding.

The most important comparison is:

- **same actor, same environment, with vs without critic**.

---

## 11. Expected Benefits

If this works, we expect:

1. lower variance than pure day-level group-relative updates,
2. better temporal credit assignment,
3. fewer pathological updates where a bad late block is reinforced because the whole day happened to win,
4. improved sample efficiency,
5. a more control-native training loop.

---

## 12. Main Risks

### Risk 1: Critic bias

If the critic is poor, it may inject the wrong baseline and hurt the actor.

**Mitigation:**
- value pretraining,
- conservative critic architecture,
- delayed actor update until critic quality is acceptable.

### Risk 2: Block granularity mismatch

If blocks are still too large, credit assignment may remain coarse.

**Mitigation:**
- start with current block setup,
- later refine into sub-block or knot-level return only after `V_block` is stable.

### Risk 3: Actor and critic optimize on different abstractions

The actor emits text-encoded structured actions, while the critic sees numeric states only.

**Mitigation:**
- define the RL action unit at the JSON/setpoint block level,
- do not let the critic model token-level semantics.

---

## 13. Short Version of the Research Position

Our current position is:

- Pure GRPO-style day-level updates are likely too coarse for multi-block HVAC control.
- HVAC control is more suitable for a learned critic than long-CoT reasoning because the state and reward are structured and physically grounded.
- The most promising VAPO-inspired translation is **not** a language-side value model, but a **small numerical block-value critic**.
- Therefore, the next concrete step is:

> keep the LLM as a structured setpoint actor, add a numerical `V_block` critic, pretrain the critic offline, and switch policy optimization from pure day-level group-relative updates to block-level value-augmented grouped updates.

---

## 14. Immediate Action Checklist

### Week 1

- [ ] define block slicing code for all existing rollout logs
- [ ] export `(state_block_start, G_block)` dataset
- [ ] implement `V_block` MLP critic
- [ ] train critic offline and inspect explained variance

### Week 2

- [ ] integrate critic inference into online training loop
- [ ] compute `A_block = G_block - V_block(s)`
- [ ] add group normalization on top of critic baseline
- [ ] replace pure day-level update with block-level update

### Week 3

- [ ] run Houston smoke experiments
- [ ] compare against current day-level GRPO
- [ ] inspect whether reward variance and stability improve

### Week 4

- [ ] run Miami fair comparison
- [ ] perform critic ablation study
- [ ] decide whether to stay with `V_block` or move to finer knot-level critic

---

## 15. Final Recommendation

The first implementation should be deliberately simple:

- **actor**: current LLM setpoint generator,
- **critic**: small MLP `V_block`,
- **target**: Monte Carlo block-to-go return,
- **advantage**: critic baseline + group normalization,
- **training**: value pretraining first, then joint actor-critic updates.

This is the most practical and technically justified way to bring VAPO-style value-based thinking into the current HVAC control project.
