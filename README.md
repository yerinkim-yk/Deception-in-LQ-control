# Deception in LQ control

This repository contains the simulation code for our paper
**[Deception in Linear-Quadratic Control](https://arxiv.org/abs/2604.00227)** (arXiv:2604.00227, 2026),
including the numerical experiments in its Section V.

A **blue team** solves a linear-quadratic control problem while deliberately perturbing its trajectory to mislead an observer.
A **red team** observes the full state and runs a sequential probability ratio test (SPRT) to detect the deception.
The blue team trades off its primary cost against deception through the intensity parameter λ, and checks how likely it is to be detected.

## Files

| File | Contents |
|---|---|
| `blue_team_original.py` | `BlueTeam`: optimal deceptive controller, simulation, detection-probability estimates and bounds |
| `red_team_original.py` | `RedTeam`: SPRT statistic log L_t computed from observed trajectories |
| `main.py` | Experiment settings, a sweep over λ, and plots |

## Usage

```bash
pip install numpy scipy matplotlib tqdm
python main.py
```

All settings are at the top of `main.py`:
- `n_dim`: dimension of the velocity v and position p
- `terminal_time`: horizon T
- `lambda_list`: deception intensities λ to sweep
- `s_b_matrix`, `s_c_matrix`: deception pattern matrices F^b, F^c
- `var_z`, `var_y`: noise covariances (a scalar means scalar·I)
- `alpha_error`, `beta_error`: SPRT error probabilities a, b
- `sample_size`, `significance`: Monte Carlo sample size and confidence level
- `random_noise`: `False` fixes the seed so every λ sees the same noise, `True` uses fresh noise

The defaults reproduce the paper's numerical setup (Section V).

## Correspondence with the paper

| Paper | Code |
|---|---|
| Dynamics (1), primary cost (2)–(3) | `BlueTeam.run_dynamics`, `calculate_primary_cost` |
| SPRT statistic log L_t (8), thresholds (Lemma 1) | `RedTeam.sequential_hypothesis_test`, `RedTeam.U` |
| Augmented cost (11), optimal control (13), recursions (14) | `BlueTeam.induction_step`, `backward_recursion`, `compute_control_input` |
| Well-posedness range of λ (Theorem 1) | `BlueTeam.margin` |
| Deception measure D(λ) (15) | `BlueTeam.calculate_deception_measure` |
| Mean, covariance and E[log L_t] (Lemma 3) | `BlueTeam.run_mean_and_covariance_dynamics_under_optimal_control` |
| Sampling-based detection probability, confidence interval (Lemma 2) | `BlueTeam.calculate_sampling_based_probability`, `score_confidence_interval` |
| Probabilistic bound (Theorem 2) | `BlueTeam.compute_C_D_matrices`, `compute_L_matrices`, `get_theorem_2_bounds` |

## Notes

- The code supports v, p ∈ ℝⁿ for any `n_dim`; the paper's experiments use n = 1.
- Lemma 3 and Theorem 2 assume f^d ≡ 0 (Assumption 1), so keep `BlueTeam.s_d` at zero when using them.
- The blue and red teams are kept independent: each holds its own copy of the deception pattern and computes log L_t from its own knowledge.
