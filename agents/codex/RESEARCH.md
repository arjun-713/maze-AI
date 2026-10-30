# Codex Memory PPO: research and design decision

This is an engineering hypothesis for this particular environment, not a claim of
superiority established by an experiment. The contestant has not been trained or
run. The work used primary papers, author-maintained implementations/documentation,
and the checked-out environment source. No pretrained weights or policy code were
copied. PyTorch supplies numerical primitives and Adam; the architecture, memory,
rollout collection, recurrent PPO objective, seed protocol, curriculum, and agent
integration are implemented here.

## Environment findings

The public `MazeEnv.reset(level, seed)` and `step(action)` API is the only environment
interface used by the contestant. The 35 presets use 9/13/17/21/27/33/41 generation
cells per side. The visible board has 19/27/35/43/55/67/83 blocks per side. Start,
goal, collision, generation rejection gates, and trusted shortest-path computation
remain engine concerns. Inspection of the generator established that every accepted
maze is reachable, includes branch/dead-end constraints, and has alternative routes.
No generation metrics are supplied to the policy or critic.

The policy observes a 5x5 window with `wall`, `trap`, `goal`, `start`, and permitted
outgoing `open` directions; absolute current position; map dimensions; approximate
goal bearing; moves, move limit, remaining moves, and score. Outside-board cells
are `None`. The exact goal coordinate, maze seed, unseen terrain, and solver are
not policy inputs. The runner controls seeds for experiment reproducibility but
does not pass them to the policy as maze identifiers.

Completion adds 1000 points. Entering a trap subtracts 50 every visit. Wall attempts
subtract 5; every action after the trusted optimal length subtracts 3. Blocked
actions also consume the move budget. Exceeding that budget is an actual episode
failure, not a rollout time-limit that should bootstrap forever. Consequently both
`done` and `truncated` cut the GAE recurrence and use zero terminal bootstrap.

## Alternatives considered

| Approach | Attraction here | Reason for the selected design |
| --- | --- | --- |
| Tabular Q-learning | Excellent educational fixed-maze baseline | Position-indexed values do not express a transferable exploration strategy across procedural layouts. Redbull's implementation is the user's choice and remains untouched. |
| Feed-forward DQN / Rainbow | Discrete four-action space; replay can reuse expensive experiences | Current local view aliases different histories. A stack of recent frames does not preserve discoveries from hundreds of moves ago. Distributional/double/dueling improvements do not themselves fix memory. |
| DRQN / R2D2 | Recurrent value learning handles histories and uses replay | A credible alternative, but replay must address hidden-state staleness, parameter drift, sequence storage, targets, and replay weighting. That is a larger operational surface for an independently maintained consumer-hardware contestant. |
| Plain PPO | Simple on-policy collection and stable clipped updates | Feed-forward PPO still has perceptual aliasing. GRU memory plus explicit observation storage are needed. |
| Recurrent PPO alone | Contiguous training sequences; no stale replay buffer | Long-term spatial recall is hard to learn entirely inside a small GRU. A lossless episodic observation store complements the learned recurrent state. |
| Transformer / learned world model | Longer-context attention or imagined planning | More memory, context management, and training complexity. Not selected without experiments establishing a benefit on these small structured observations. |
| RND exploration | Learned novelty for sparse-reward exploration | Every maze is new; visual novelty can remain high without useful progress. Exact episodic position/discovery counts are cheap and already permitted by the observation. No extra predictor network is needed. |
| External shortest-path policy | Easy once a map is known | Deliberately excluded. No BFS, Dijkstra, A*, DFS navigation controller, flood-fill planner, or solver demonstrations are used. |

## Selected design

**Observation memory + two-scale CNN + GRU + clipped PPO with two value heads.**

The map stores only observed cells, legal directions, visits, and visit recency.
It is reset per maze, never persisted as a cross-maze policy lookup table. A local
15x15 crop preserves corridor detail around the player. A pooled 21x21 remembered
map offers distant spatial context, with maxima keeping rare goal/frontier markers
visible. Pooling is approximate; it is not a topology-preserving planning map.
The 128-unit GRU models temporal control. The observation store has fixed writes,
unlike the learned write operator in Neural Map; the neural policy learns how to
use both representations. This is a domain-specific memory design, not a reproduction
of that paper.

Known blocked directions are masked using the current cell's public `open` list.
Traps are **not** masked: they may be unavoidable. There is no goal-seeking fallback
or forced exploration controller. Both training and inference select from learned
policy logits. Stochastic inference is the default to avoid making a tie or a
bad argmax choice into a permanent deterministic cycle; argmax is an explicit option.

Task rewards are raw score deltas divided by 100, with a small training-only action
cost. An intrinsic head predicts first-visit and newly observed floor bonuses.
The policy combines their advantages with an annealed exploration weight. The
official score and replay rewards remain raw and unmodified. Evaluation disables
all bonuses and updates; the frozen policy still retains episode memory.

PPO training uses contiguous time sequences, pre-observation hidden states,
episode-start masks, detached burn-in prefixes, dual GAE, value clipping, entropy,
gradient clipping, advantage normalization, an annealed learning rate, and KL early
stopping. Burn-in at the start of a rollout has only its cached hidden state, so it
reduces rather than eliminates recurrent state staleness after updates. The spatial
store reduces dependence on exact long GRU histories.

The curriculum unlocks one level at a time only after sufficient successful training
episodes **and** validation performance. Earlier weak levels receive rehearsal.
Procedural seeds occupy disjoint integer domains for training, validation, and final
evaluation. Validation is reused for promotion/model selection and is therefore
not an unbiased championship test. Five secret mazes must remain outside tuning.

## Primary sources and how they informed the implementation

1. Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
   Source of the clipped policy objective and repeated minibatch optimization. It
   does not establish PPO as best for this maze; that choice is an engineering judgment.
2. Schulman et al., [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
   Basis for the bias/variance-controlled advantage recursion.
3. Hausknecht and Stone, [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527).
   Motivates recurrent histories rather than treating the visible window as Markov.
4. Hessel et al., [Rainbow](https://arxiv.org/abs/1710.02298).
   Reviewed as a strong discrete-action alternative; it combines improvements to DQN.
5. Kapturowski et al., [R2D2](https://openreview.net/forum?id=r1lyTjAqYX).
   Highlights recurrent replay state/representation drift. The indexed abstract was
   available; full PDF access was blocked by OpenReview's browser challenge. This
   implementation does not rely on unverified numerical claims from that PDF.
6. Parisotto and Salakhutdinov, [Neural Map](https://arxiv.org/abs/1702.08360).
   Motivates long-lived spatial memory for partially observed navigation. Our
   memory writes are deterministic observed-data writes, not that learned memory.
7. Cobbe et al., [Procgen](https://arxiv.org/abs/1912.01588).
   Motivates training-layout diversity and explicitly separate generalization tests.
8. Pleines et al., [Generalization, Mayhems and Limits in Recurrent PPO](https://arxiv.org/abs/2205.11104).
   Informs sequence arrangement, initial states, boundary masks, and caution about
   claiming recurrence solves every memory task.
9. Bellemare et al., [Count-Based Exploration](https://arxiv.org/abs/1606.01868).
   Motivation for novelty rewards. Here exact episode-local observed counts replace
   pseudo-count density estimation; the bonus formula is our environment-specific choice.
10. Burda et al., [Random Network Distillation](https://arxiv.org/abs/1810.12894).
    Informs the separation of intrinsic and task value estimates; the RND algorithm
    itself is not included.
11. [SB3-Contrib recurrent PPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
    and [CleanRL recurrent PPO notes](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstm.py).
    Reviewed established implementations for recurrent-state and episode-reset
    conventions. Neither trainer is imported or required.
12. PyTorch official [GRUCell](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRUCell.html),
    [serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html), and
    [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) documentation.
    Used for API shape, checkpoint loading, and reproducibility limits.

## Unresolved empirical questions

These require authorized future experiments: sparse-reward learning speed; whether
the curriculum reaches level 35 within the chosen budget; whether 21x21 pooling
loses necessary global topology; the best intrinsic decay and entropy; transfer to
secret maze distributions; CPU/GPU speed; and stochastic versus greedy performance.
No success rate, convergence, or speed measurement has been produced.
