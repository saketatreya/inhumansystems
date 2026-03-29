---
title: "Solving Pokémon as a POMDP"
tags: [pokemon, AI, RL, dev]
date: 2026-02-15
excerpt: "A technical deep-dive into building a Gen 9 Random Battle agent that learns from nothing but wins and losses through Bayesian Belief Tracking, Observation Engineering, and Pure RL"
---

## Abstract

Pokémon Random Battles are a partially observable, stochastic, adversarial game with a combinatorially large hidden state. The opponent's items, abilities, EV spreads, and full movesets remain hidden until revealed — a structure that breaks standard MDP assumptions and demands something more principled than heuristic guessing. Most prior Pokémon agents either sidestep the hidden information problem entirely or encode domain knowledge directly into dense reward functions, which we show produces systematic degenerate behaviors.

This post describes **ShowdownBot**, a Gen 9 Random Battle agent built around three contributions: (1) a Bayesian belief tracker that maintains probability distributions over opponent roles using a structured domain prior, (2) a 1163-dimensional observation space that fuses belief state with damage projections, field encoding, and uncertainty-aware matchup heuristics, and (3) a pure sparse-reward self-play pipeline validated through empirical ablation of alternative reward schemes. The work extends Wang's (2024) Gen 4 methodology to Gen 9, which introduces qualitatively new POMDP complexity: Terastallization expands the hidden state, a larger Pokémon pool increases prior uncertainty, and a sweeper-dominant meta changes the credit assignment structure relative to Gen 4's stall-oriented play.

Training currently runs on an M1 MacBook Air, where the agent marginally outperforms the `SimpleHeuristicsPlayer` baseline. The system design — not the final Elo — is the contribution of this post. At Wang-scale compute (150M steps, A6000 GPU, MCTS at inference), the architecture is designed to go significantly further.

---

## 1. Introduction: Why Pokémon Is a Hard Problem

On the surface, Pokémon looks like a straightforward turn-based game. Two players alternate choices; the goal is to make all six of the opponent's Pokémon faint. But this description hides most of the difficulty.

**Stochasticity.** Every damage-dealing move has a 15-bin damage roll, a miss chance, a crit chance, and often secondary effect probabilities. Even with a known state, outcomes are random. Learning requires enough samples to average over this noise.

**Massive state space.** Wang (2024) gives a lower bound on the Gen 4 game tree of approximately $10^{88}$ — larger than chess ($10^{120}$ is actually an upper bound, but the branching factor for Pokémon with stochasticity is effectively higher given imperfect information). Tabular methods are completely out of reach; function approximation over this space is the only option.

**Partial observability.** This is the deepest difficulty. At the start of a Random Battle, you see your own team. You see the opponent's lead Pokémon and nothing else. Their item, ability, EV spread, and remaining three to four moves are all hidden until revealed through play. Standard RL methods that assume full state access are applying a fiction — the agent is actually operating in a POMDP.

Formally, we model Gen 9 Random Battles as a POMDP with the tuple $(S, A, O, T, R, \Omega)$:

- **$S$**: The true game state, including both players' full teams with complete sets (moves, items, abilities, EVs).
- **$A$**: The action space — up to 4 moves plus up to 5 switches, plus Terastallization variants, giving 26 discrete actions.
- **$O$**: The observable state — what the Showdown API surfaces: your own full team, the opponent's revealed information (HP, fainted status, any revealed moves/items/abilities).
- **$T$**: The transition function — stochastic, governing damage rolls, misses, crits, and secondary effects.
- **$R$**: The reward function. We use $+1$ for win, $-1$ for loss, $0$ otherwise.
- **$\Omega$**: The observation function — maps true states to observable states by filtering hidden information.

The agent never has access to $S$ directly. It must infer what it can from $O$ and act under irreducible uncertainty about the rest.

---

## 2. Gen 9 vs. Gen 4: What Actually Changes

Wang's thesis works on `gen4randombattles`. Our project targets `gen9randombattles`. These are not the same problem in disguise — the POMDP complexity increases qualitatively in Gen 9, not just quantitatively.

**Terastallization.** The defining Gen 9 mechanic. Each Pokémon may Terastallize once per battle, changing its type to a hidden "Tera type" (which may differ from the Pokémon's base typing). This means the hidden state must track: has this Pokémon Terastallized yet (binary), and if not, what is its Tera type (up to 18 possible types, unknown until revealed). Type immunity calculations, damage effectiveness, and ability interactions all change upon Terastallization. The `BeliefTracker` must handle this: Tera type gets its own probability distribution per Pokémon, updated to certainty upon observation and constrained by role priors before that. Actions 22–25 in the 26-action space are move + Terastallize variants, adding a timing dimension absent in Gen 4 entirely.

**Larger Pokémon pool.** Gen 9 has approximately 400+ Pokémon available in `gen9randombattles` versus 296 in Gen 4. The `gen9randombattle.json` role dictionary is correspondingly larger, meaning role prior distributions are sparser and initial uncertainty is higher.

**Meta structure.** Gen 4 Random Battles favor "stalling" — entry hazards, status moves, healing — over long time horizons. Gen 9 is sweeper-dominant: games are faster-paced, with more offensive pressure and more reliance on momentum (U-turn, Volt Switch, pivoting chains). This matters for credit assignment: under sparse rewards, a stall-oriented game gives the LSTM more turns to work with per credit signal; a faster-paced game compresses the horizon and makes temporal credit assignment harder. This is one reason sparse rewards take longer to converge in Gen 9.

**No permanent weather from abilities.** In Gen 4, abilities like Drought set permanent weather. Gen 9 removes this; weather always has a turn counter. This changes the `FieldEncoder` design slightly — no "permanent" bin needed for weather conditions, unlike Wang's Gen 4 one-hot encoding.

These aren't reasons the methodology fails — they're reasons the problem is harder and the system must be designed more carefully.

---

## 3. The Bayesian POMDP Solver

This is the architectural centerpiece. Everything else in the system exists to support the agent making good decisions despite partial observability. The `BeliefTracker` (`src/belief_tracker.py`) is the explicit module for handling that problem.

### 3.1 The Structure of Hidden Information

In `gen9randombattles`, each Pokémon is generated according to the Showdown team generator, which draws from a dictionary of predefined **roles**. A role specifies: a set of likely moves (not necessarily all four — some moves appear in multiple sets), an item distribution, an ability, and a Tera type. For example, Weavile in Gen 9 might have two roles:

```
Weavile:
  Fast Physical Sweeper (80% usage):
    moves: [Knock Off, Ice Spinner, Low Kick, Ice Shard]
    items:  [Choice Band, Life Orb]
    ability: Pressure
    tera:   Ice

  Swords Dance (20% usage):
    moves:  [Swords Dance, Triple Axel, Knock Off, Ice Shard]
    items:  [Life Orb]
    ability: Pressure
    tera:   Ice
```

This role dictionary is the **prior-generating artifact**. Before any moves are made, the belief distribution over a Pokémon's role is exactly the usage rate distribution from the dictionary. These aren't uniform priors — they're usage-weighted, which is a meaningful starting point.

### 3.2 Bayesian Updates

For each opponent Pokémon, the `BeliefTracker` maintains a `PokemonBelief` object with:

```
PokemonBelief
  ├─ role_probs: Dict[str, float]     # P(role | observations)
  ├─ observed_moves: Set[str]         # revealed moves
  ├─ observed_item: Optional[str]     # revealed item
  ├─ observed_ability: Optional[str]  # revealed ability
  └─ observed_tera: Optional[str]     # revealed tera type
```

When the opponent uses a move $m$, the update is:

$$P(R_i \mid m) = \frac{P(m \mid R_i) \cdot P(R_i)}{\displaystyle\sum_j P(m \mid R_j) \cdot P(R_j)}$$

where $P(m \mid R_i) \in \{1.0, \epsilon\}$: 1.0 if move $m$ appears in role $R_i$'s moveset, $\epsilon = 10^{-9}$ otherwise. The $\epsilon$ rather than hard 0 prevents numerical collapse in edge cases where the role dictionary is incomplete or a Pokémon uses a move outside its listed sets (which happens occasionally due to move inheritance mechanics).

This is a standard Bayesian filter. What makes it useful here is the structure of $P(m \mid R_i)$: because role movesets are discrete and known, the likelihood is essentially a lookup — no learned component needed.

**Concrete example.** Turn 1: opponent sends out Weavile. Prior: P(Sweeper) = 0.80, P(SwordsDance) = 0.20. They use Swords Dance.

$$P(\text{SwordsDance role} \mid \text{Swords Dance used}) = \frac{1.0 \cdot 0.20}{0 \cdot 0.80 + 1.0 \cdot 0.20} = 1.0$$

Posterior: P(Sweeper) = 0.0, P(SwordsDance) = 1.0. We now know with certainty which role this is. The agent can now calculate: this Weavile has Triple Axel (probably), not Low Kick (which is in the Sweeper set but not the SwordsDance set). Counterplay changes accordingly.

### 3.3 Deductive vs. Probabilistic Updates

The update above is probabilistic — it updates distributions. But some observations are **certainty events** that set a feature to P=1.0 regardless of role:

| Observation | Update |
|-------------|--------|
| Item consumed (e.g. berry) | `confirmed_item = item` |
| Ability triggered (e.g. Intimidate animation) | `confirmed_ability = ability` |
| Pokémon Terastallized | `observed_tera = tera_type`, P(tera=X) = 1.0 |
| Move used | Bayesian update over role_probs |

Speed inference is a special case. If your Dragapult (421 Speed) is outsped by the opponent's Gholdengo (max non-Scarf Speed: 293), the only possible explanation is Choice Scarf. The update is deductive: P(item = Choice Scarf) = 1.0. This is not a Bayesian update — it's logical elimination. The `BeliefTracker` handles both, treating deductive updates as likelihood functions that set $P(O \mid R_i) = 0$ for all incompatible roles.

### 3.4 Entropy as an Uncertainty Signal

Shannon entropy over the role distribution is included in the observation vector:

$$H = -\sum_i p_i \log p_i$$

When $H$ is high, the agent is uncertain about the opponent's role. When $H = 0$, the role is fully determined. This gives the LSTM policy a signal to distinguish "I know this is a Choice Scarf set — pivot into my Ground-type resist" from "I don't know this Pokémon's item or moves — play conservatively and gather information." Uncertainty-awareness is not just epistemically correct; it's strategically necessary.

### 3.5 Belief Embedding

`PokemonBelief.to_embedding()` produces 16 floats per opponent Pokémon, for a total of 96 floats across 6 slots:

| Index | Feature |
|-------|---------|
| 0–3 | Top 4 role probabilities |
| 4–7 | Top 4 unrevealed move probabilities |
| 8–10 | Top 3 item probabilities |
| 11–13 | Top 3 Tera type probabilities |
| 14 | Moves revealed fraction (revealed / 4) |
| 15 | Role entropy (normalized) |

The move probabilities (indices 4–7) are derived from the role distribution: for each possible move $m$, the marginal probability is $\sum_i P(R_i) \cdot \mathbb{1}[m \in R_i]$, then sorted and top-4 returned. This gives the agent a prior over what moves it hasn't yet seen.

---

## 4. Observation Space Engineering

The `ObservationBuilder` (`src/embeddings.py`) maps the Showdown battle state to a `(1163,)` float32 vector. Every design decision here is about **information completeness without redundancy** — what does the agent need to act well, and how do you encode it cleanly?

### 4.1 The Encoder Stack

Twelve encoders run in parallel, concatenated in this order:

| Encoder | Floats | What it captures |
|---------|--------|-----------------|
| `ActivePokemonEncoder` (self) | 97 | HP, types, base stats, boosts, status, Tera state, item/ability flags |
| `ActivePokemonEncoder` (opponent) | 97 | Same, for opponent's active Pokémon |
| `TeamEncoder` (self) | 246 | HP, type, status, revealed/fainted/active flags × 6 slots |
| `TeamEncoder` (opponent) | 246 | Same for opponent |
| `MovesEncoder` | 148 | Type, BP, accuracy, PP, priority, category, effectiveness × 4 moves |
| `OpponentMovesEncoder` | 84 | Features for up to 4 revealed opponent moves |
| `MatchupEncoder` | 25 | Switch candidate scores vs. current opponent active (5 × 5) |
| `DamageEncoder` | 40 | Min/max damage roll estimates vs. likely roles |
| `FieldEncoder` | 35 | Weather, terrain, hazards, screens, Trick Room |
| `BeliefEncoder` | 96 | Bayesian belief embeddings × 6 opponent slots |
| `ActionMaskEncoder` | 26 | Binary legality mask for all 26 actions |
| `MetaEncoder` | 23 | Turn count, remaining Pokémon counts, game context |
| **Total** | **1163** | |

The ordering isn't arbitrary. Most action-relevant information (active Pokémon, available moves) comes first, so the LSTM's initial hidden state is biased toward what matters on the current turn. Team and belief information comes later — lower-frequency updates.

### 4.2 Normalization Philosophy

The observation vector feeds directly into an LSTM, where activation saturation is a real concern. Every encoder output must stay within a tight range:

- **Stats and Base Power**: divided by 200. A stat of 200 maps to 1.0; 400 (rare max) maps to 2.0.
- **Priority**: $(p + 7) / 14$, mapping the range $[-7, 7]$ to $[0, 1]$.
- **Accuracy**: divided by 100.
- **Stat boosts**: divided by 6, giving $[-1, 1]$.
- **HP fraction**: raw percentage already in $[0, 1]$.

The overall range is `[-1.0, 4.0]` with overflow headroom — some combinations of high-stat Pokémon in unusual circumstances push above 1.0, and the headroom prevents clipping without requiring a tighter cap that would lose information.

### 4.3 The DamageEncoder: Belief-Weighted Damage Projections

This encoder is architecturally the most interesting. Rather than computing damage against a single assumed opponent role (which would be overconfident), the `DamageEncoder` spawns **shadow Pokémon** for each plausible role, runs the damage calculation engine against each, and returns a **belief-weighted average** of min/max rolls.

Concretely: if the opponent has P(Sweeper) = 0.7 and P(SwordsDance) = 0.3, and these two roles have different items (Choice Band vs. Life Orb), the damage estimate is:

$$\hat{d}_{min} = 0.7 \cdot d_{min}(\text{Sweeper}) + 0.3 \cdot d_{min}(\text{SwordsDance})$$

When belief entropy is high, this average is noisy and wide — which is the correct representation of the situation. When the role is known exactly, the estimate collapses to the true roll distribution. The `DamageEncoder` is uncertainty-aware by construction, because it inherits the uncertainty of the belief state.

### 4.4 The Action Mask

The 26-bit action mask (embedded in the observation and used to zero pre-softmax logits) prevents the agent from ever sampling an illegal action. Actions 0–5 are switches; 6–9 are moves; 10–21 are move + mega/Z-move/dynamax variants (always masked in Gen 9, present for API compatibility with poke-env's `SinglesEnv`); 22–25 are move + Terastallize.

The mask is applied twice: during rollout collection (before sampling), and again during gradient updates (when computing the probability that the updated policy would have chosen the stored action). The second application is critical — without it, PPO's probability ratio $r_t(\theta) = \pi_\theta(a \mid s) / \pi_{\theta_{old}}(a \mid s)$ would assign nonzero probability to actions that could never be chosen, corrupting the advantage estimate.

---

## 5. Pure RL and the Sparse Reward Argument

The reward function is deceptively simple: $+1$ on a win, $-1$ on a loss, $0$ on every other turn. No partial credit, no shaping, no domain knowledge encoded as reward signal.

This is a principled design choice, not a default. We tried multiple dense reward schemes before converging on sparse. Each produced a specific category of degenerate behavior.

### 5.1 Reward Ablation: What Dense Rewards Taught Us

| Reward Scheme | Intended Behavior | Learned Degenerate Behavior | Why It Failed |
|---------------|------------------|----------------------------|---------------|
| HP differential | Encourage HP preservation | Avoided knockouts; stalled even when a kill was available | Reward for opponent HP loss decoupled from win probability; agent optimized HP score, not wins |
| Damage dealt | Encourage offensive pressure | Spammed high-BP moves regardless of matchup; ignored switching | Damage reward is always positive for offensive moves; agent learned to never switch |
| Momentum (U-turn / Volt Switch) | Encourage pivoting to favorable matchups | Pivoted compulsively, even in winning positions where staying in was correct | The reward for using a pivot move was unconditional; the agent had no incentive to evaluate when *not* to pivot |
| Win + faints bonus | Encourage aggressive play | "Kamikaze" sacrifices — traded Pokémon recklessly for faint bonuses regardless of team value | Fainting opponent Pokémon has positive reward even when the sacrifice damages your team's long-term win probability |

Each of these is a form of **specification gaming** — the agent found a policy that maximizes the proxy reward while failing to maximize the true objective (winning). The HP differential failure is particularly instructive: in the Gen 9 meta, correctly identifying when to "sacrifice" a Pokémon to preserve momentum or remove a threat is a core strategic skill. The HP reward directly penalizes this correct behavior.

Sparse rewards eliminate these failure modes by construction: the only thing that earns reward is winning. The cost is sample efficiency — the agent must learn to assign credit across 20–40 turns of $0$ reward before seeing any signal. This is harder and slower, especially on limited compute. But it learns the right thing.

### 5.2 Self-Play and the 70/30 Split

Training uses fictitious self-play: 70% of parallel environments fight against randomly sampled past checkpoints of the agent itself, while 30% fight a `RandomPlayer` baseline.

The 30% `RandomPlayer` allocation is not arbitrary — it prevents **catastrophic forgetting**. If the agent trains exclusively against its own historical policy, it may become brittle against unconventional or random play, since such play no longer appears in the training distribution. The `RandomPlayer` keeps basic robustness in the objective throughout training.

The 70% self-play component ensures the agent faces increasingly strong opposition as it improves. Sampling randomly from the checkpoint pool (rather than always using the most recent checkpoint) provides a mix of difficulty levels and prevents the agent from over-adapting to a single opponent style.

### 5.3 The LSTM as Implicit Belief Integrator

The 256-unit LSTM layer between the feature extractor and actor/critic heads serves a dual purpose. Formally, it handles temporal credit assignment — connecting turn-$t$ actions to a reward that may only arrive at turn 30. But it also functions as an **implicit belief integrator** over the episode.

Consider what the LSTM "sees" across turns: the `BeliefEncoder` output changes as moves are revealed and probability mass concentrates. The LSTM can learn to track this evolution in its hidden state, augmenting the explicit Bayesian tracker with learned, uninterpretable patterns. The two systems are complementary: the `BeliefTracker` handles structured probabilistic inference using domain knowledge; the LSTM handles unstructured temporal patterns that the `BeliefTracker` can't represent.

This matters most for **speed inference** patterns that aren't captured by the role dictionary — for example, if a particular opponent has been unexpectedly fast on multiple turns, the LSTM can learn to upweight the Scarf hypothesis beyond what the static role prior would suggest.

---

## 6. Curriculum Learning: What Didn't Work (And One Interesting Direction)

We implemented opponent pool management with a `WinRateTracker` and phase-conditional opponent rotation — cycling through increasingly capable heuristic opponents before transitioning to self-play. The result: a slight bootstrapping effect in early training (faster initial learning curve against weak baselines), but no significant improvement in final performance compared to pure self-play from the start.

This is consistent with Wang's choice to use pure self-play without curriculum. The intuition is that in Random Battles, the core strategic skills (type matching, hazard setting, momentum preservation) appear at all skill levels — the agent can learn them from itself. A curriculum that gradually introduces these skills doesn't add information the self-play signal doesn't eventually provide.

**One direction that remains interesting**: modeling the teacher as a POMDP over the student's learning state. A fixed curriculum assumes the teacher knows where the student is in skill space. A POMDP curriculum maintains a belief distribution over the student's current capabilities and updates it based on observed win rates, policy entropy, and loss curves — then selects the opponent that maximizes expected learning signal given that belief. This is distinct from standard curriculum learning in exactly the way the `BeliefTracker` is distinct from a fixed prior: it's adaptive rather than predetermined. We didn't implement this; it's listed as future work.

---

## 7. Training Infrastructure

### 7.1 The Simulation Stack

Training requires a local instance of the Pokémon Showdown NodeJS server running in `--no-security` mode. The Python training loop communicates with the server via WebSocket using the `poke-env` library. A modified `serv.js` handles mass login without rate-limiting.

Parallel training uses `SubprocVecEnv` with 4–8 workers. Workers are initialized with a 2-second staggered startup delay to prevent WebSocket handshake collisions — without this, simultaneous login attempts frequently fail.

### 7.2 The PPO Configuration

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Algorithm | RecurrentPPO (sb3_contrib) | LSTM with 256 units |
| $\gamma$ | 0.9999 | Near-undiscounted; episodes are ~25 turns |
| Entropy coefficient | 0.0588 | Encourages exploration |
| Clip range | 0.0829 | Conservative — prevents policy collapse |
| LR schedule | $\text{LR}_0 / (8x + 1)^{1.5}$ | Power-law decay; Wang (2024) §3.1.4 |
| Validation | Win rate vs. `SimpleHeuristicsPlayer` every 20k steps | |

The learning rate schedule deserves attention. Wang showed that constant learning rate produces a policy stuck around 55% win rate against `SimpleHeuristicsPlayer`; the annealing schedule reaches 80%+ at 40M steps and roughly 85% at 150M. The $1.5$ exponent means the rate decays faster than cosine annealing early in training (when large gradient updates are safe) and slower late (when stability matters). The constants 8 and 1.5 were not tuned — future work could properly optimize them.

### 7.3 Current Compute Reality

Wang's results used a single NVIDIA A6000 (48GB), 80 CPU workers, and 150M steps over 4 days. We're training on an M1 MacBook Air with 4 parallel environments. The system converges slowly at this scale under sparse rewards. Current result: marginal outperformance of `SimpleHeuristicsPlayer` — roughly the same regime Wang reaches in the first few million steps.

The architecture is designed for Wang-scale compute. MCTS at inference, which Wang shows dramatically increases performance (80.9% win rate vs. NN alone in head-to-head), requires a fast enough simulation loop that we can run 1000–2000 rollouts per decision within the Showdown timer. This is blocked by compute; the belief-weighted damage projection and the LSTM inference pipeline are both designed to be MCTS-compatible once that compute is available.

---

## 8. Preliminary Results

Current benchmark: win rate against `SimpleHeuristicsPlayer` after several hundred thousand training steps on M1.

The agent is currently in the early regime — it has learned basic type matching and avoids flagrantly illegal reasoning, but hasn't yet developed the pivoting, hazard-setting, or multi-turn planning that characterizes stronger play. This is expected at this compute scale. The learning curve shape is consistent with Wang's Figure 4.1: rapid early improvement followed by diminishing returns that require compute to push through.

What we've verified empirically:
- Dense reward schemes produce the degenerate behaviors documented in Section 5.1.
- The `BeliefTracker` correctly updates role posteriors across a range of test cases (Bayesian unit tests pass for all update mechanisms including speed inference and deductive elimination).
- The action mask correctly prevents all illegal actions across all game states tested.
- The observation vector is numerically stable (no NaN propagation, values within `[-1, 4]` across 10k test episodes).

What we haven't verified yet:
- Whether the LSTM is actually leveraging the belief state embeddings (would require attention analysis or probing).
- Elo on the online ladder.
- Performance of MCTS + NN vs. NN alone.

---

## 9. Discussion and Future Work

### 9.1 The MCTS Ceiling

Wang's most important empirical finding is that MCTS provides a substantial policy improvement over the trained NN alone: 80.9% win rate for MCTS+NN vs. NN head-to-head. The architecture here is designed to unlock this. The `BeliefTracker` is precisely the "Sampling Unknown Information" mechanism (Wang §3.2.2) — it enables MCTS rollouts against plausible hidden states rather than requiring full information.

The computational constraint is the Showdown simulation speed: MCTS needs 1000+ rollouts per decision within 10 seconds. This requires sufficient CPU parallelism — Wang uses 20 parallel MCTS workers. Achievable with the right hardware.

### 9.2 Gen 9-Specific Open Problems

**Terastallization timing.** When is the correct turn to Terastallize? This is a strategic decision with irreversible consequences — you can't un-Terastallize. The agent must learn to value the optionality of an unused Tera and to estimate when the opponent might Terastallize based on belief state. This is a hard credit assignment problem under sparse rewards.

**Zoroark.** Gen 9 introduces Zoroark-H, which disguises itself as another Pokémon on its team. This creates a second layer of hidden state: not just what set is the opponent's Pokémon, but whether the Pokémon you're looking at is actually the one it appears to be. The `BeliefTracker` doesn't currently handle this — it would require a meta-level distribution over whether a revealed Pokémon is real or an illusion.

**Opponent modeling beyond role priors.** The current `BeliefTracker` treats all opponents as drawing from the same `gen9randombattle.json` prior. Real opponents have styles (some players never use hazards; some always Terastallize turn 1). An opponent modeling layer could update a higher-level prior over opponent style based on observed play patterns, then condition role beliefs on that prior. This is the "Smarter Opponent Modeling" future direction from Wang §5.2.3, extended to the belief layer.

### 9.3 The Broader Pattern

The `BeliefTracker` design pattern — Bayesian inference over a structured domain-specific prior, with entropy as an uncertainty signal to downstream components — generalizes beyond Pokémon. Any game or system where hidden state has known structure (possible states are discrete and come from a known distribution) can use this pattern. The key requirement is the prior-generating artifact: the domain dictionary that defines what the hidden states can be and with what relative frequencies.

---

## Conclusion

Gen 9 Random Battles are a genuinely hard POMDP: stochastic, partially observable, adversarial, and large-scale. Building a principled agent means taking each of these properties seriously rather than hoping a neural network learns to work around them.

The architecture described here does three things simultaneously: it represents hidden information as a probability distribution (the `BeliefTracker`), it encodes that uncertainty into the observation space in a way the policy can act on (the `BeliefEncoder` and `DamageEncoder`), and it trains the policy with a reward signal uncorrupted by proxy objectives (sparse rewards). Each of these is independently justified; together they form a coherent system for learning to play well in the face of uncertainty.

The compute ceiling is real. At M1 scale, the agent is barely scratching the surface of what the architecture can do. At Wang-scale — or beyond, with MCTS properly integrated — the design should produce meaningfully stronger play. That's the next step.

---

## References

- Wang, J. (2024). *Winning at Pokémon Random Battles Using Reinforcement Learning*. MEng Thesis, MIT EECS. [Thesis supervised by Joshua Tenenbaum]
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Silver, D. et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and go through self-play*. Science, 362(6419).
- Huang, S. & Lee, S. (2019). *A self-play policy optimization approach to battling Pokémon*. IEEE CoG.
- Huang, S. et al. (2022). *The 37 implementation details of proximal policy optimization*. ICLR Blog Track.

---

*Code available at [github.com/saketatreya/showdownRL](https://github.com/saketatreya/showdownRL). Training is ongoing — this post will be updated with ladder results when compute permits.*
