# Code repository

This is the accompanying code repository for the paper: "Tactile suppression during movement as optimal integration of somatosensory feedback across time" Tatai et al. (2026)

## Citation

If you use this code or the data in this repository, please cite:

Tatai, F., Voudouris, D., Straub, D., Fiehler, K., & Rothkopf, C. A. (2026). Tactile suppression during movement as optimal integration of somatosensory feedback across time. _bioRxiv_.

```bibtex
@article{
  tatai2026tactile,
  title={Tactile suppression during movement as optimal integration of somatosensory feedback across time},
  author={Tatai, Fabian and Voudouris, Dimitris and Straub, Dominik and Fiehler, Katja and Rothkopf, Constantin A},
  journal={bioRxiv},
  pages={2026--02},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## How to get started

Use the Python package manager _uv_ to set up the environment. This creates a
virtual environment and installs all dependencies, including the local
`suppression_oc` package (editable, from `src/`):

> uv sync

For exact reproducibility, use the lockfile:

> uv sync --frozen

Run any script or notebook with `uv run`, e.g. `uv run jupyter lab`; this uses
the synced environment automatically.

## Data

### colino_plot_data.csv

Digitized plot coordinates of Figure 3 (Right Arm) from

Colino, F. L., Lee, J.-H. & Binsted, G. Availability of vision and tactile gating: vision enhances tactile sensitivity. Exp Brain Res 235, 341–348 (2017).

| Column    | Description                               |
| --------- | ----------------------------------------- |
| bin       | bin number, starting from the left        |
| d_prime   | d-prime value                             |
| condition | vision or no-vision                       |
| identity  | value, upper or lower error bar delimiter |

### tatai_et_al_detection.csv

Data from the detection task, augmented with reach summary statistics.

| Column              | Description                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| trial               | trial index within subject                                                                                            |
| subject             | subject identifier                                                                                                    |
| condition           | experimental condition: normal or uncertain                                                                           |
| uncertain_first     | whether the uncertain condition was presented first for the subject                                                   |
| intensity           | stimulus intensity (0 indicates catch/no-stimulus trials)                                                             |
| response            | participant response in the detection task (1 = detected, 0 = not detected)                                           |
| correct             | whether response correctness criterion was met                                                                        |
| hit                 | hit trial flag (stimulus present and detected)                                                                        |
| false_alarm         | false-alarm trial flag (no stimulus but reported as detected)                                                         |
| reaction_time       | reaction time in the reaching task: time from trial start until movement initiation (i.e., finger leaves the button)  |
| max_v               | maximum reach velocity                                                                                                |
| t_max_v             | time of maximum reach velocity in seconds                                                                             |
| max_a               | maximum reach acceleration                                                                                            |
| t_max_a             | time of maximum reach acceleration in seconds                                                                         |
| max_dec_a           | maximum deceleration magnitude                                                                                        |
| t_max_dec_a         | time of maximum deceleration in seconds                                                                               |
| vertical_error      | vertical endpoint error of the reach                                                                                  |
| reach_time          | total movement duration (reach time) in seconds                                                                       |
| rel_timepoint_onset | stimulus time relative to movement onset                                                                              |
| rel_timepoint_norm  | stimulus time normalized by reach duration (absolute time is retained if the stimulus occurred before movement onset) |
| timepoint_bin       | binned normalized stimulus timepoint label                                                                            |

### tatai_et_al_reaches.csv

Preprocessed reaches as described in the Methods section of the paper.

| Column          | Description                                           |
| --------------- | ----------------------------------------------------- |
| trial           | trial index within subject                            |
| subject         | subject identifier                                    |
| block           | experimental block index                              |
| condition       | experimental condition: normal or uncertain           |
| x               | x-position of the fingertip trajectory sample         |
| y               | y-position of the fingertip trajectory sample         |
| z               | z-position of the fingertip trajectory sample         |
| d               | distance-to-target (or traveled-distance) sample      |
| v               | instantaneous velocity sample                         |
| a               | instantaneous acceleration sample                     |
| reach_t         | elapsed time since movement onset (seconds)           |
| percent_reach_t | normalized reach time (fraction of movement duration) |

### signal_detection.csv

Signal detection theory estimates (d-prime and criterion) per subject, condition, and timepoint bin. Computed from `tatai_et_al_detection.csv` by `scripts/sdt.py`.

| Column        | Description                                 |
| ------------- | ------------------------------------------- |
| subject       | subject identifier                          |
| condition     | experimental condition: normal or uncertain |
| timepoint_bin | binned normalized stimulus timepoint label  |
| d_prime       | sensitivity (d') for the timepoint bin      |
| criterion     | response criterion for the timepoint bin    |

### reaches_simulated.csv

Simulated reaching trajectories from the optimal feedback control model, based on individual subjects' movement times. Generated by `scripts/simulate_reaches.py` (100 simulated reaches per subject and condition, random seed 0). Velocities and accelerations are sign-flipped such that movement is positive.

| Column          | Description                                                        |
| --------------- | ------------------------------------------------------------------ |
| reach_i         | index of the simulated reach within subject and condition (0–99)   |
| time            | elapsed time since trial start (seconds)                           |
| percent_reach_t | normalized reach time; negative values = pre-movement period       |
| d               | simulated hand distance-to-target (or traveled distance) sample    |
| v               | simulated velocity sample                                          |
| a               | simulated acceleration sample                                      |
| subject         | subject identifier whose movement time was used for the simulation |
| condition       | experimental condition: normal or uncertain                        |

### kalman_gains_initial_uncertainty.csv

Kalman gain of the optimal feedback control model (position, velocity, and acceleration gain on observed position) over time, per subject and condition. Generated by `scripts/simulate_reaches.py`.

| Column       | Description                                                           |
| ------------ | --------------------------------------------------------------------- |
| time         | elapsed time since trial start (seconds)                              |
| percent_time | normalized time; negative values = pre-movement period                |
| K_pos        | Kalman gain of the position estimate on observed position             |
| K_vel        | Kalman gain of the velocity estimate on observed position             |
| K_acc        | Kalman gain of the acceleration (force) estimate on observed position |
| subject      | subject identifier                                                    |
| condition    | experimental condition: normal or uncertain                           |

### robustness_oc_gains.csv

Dataset for the robustness analysis of the optimal control model parameters (SI): 1000 random parameter combinations with the resulting position Kalman gain trajectory (`K_pos_0` … `K_pos_89`). Generated by `scripts/robustness_oc_gains.py`.

| Column                          | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| i                               | parameter combination index                                  |
| sigma_pos, sigma_vel, sigma_frc | sampled sensory noise parameters                             |
| c, r, v, f                      | sampled cost weights                                         |
| sigma0                          | sampled initial position uncertainty                         |
| K_pos_0 … K_pos_89              | position Kalman gain over time for the sampled parameter set |

## Notebooks

### figures.ipynb

Jupyter Notebook containing code to reproduce all the plots in the paper, including the masking-model comparison panels and SI figures.

### paper_stats.ipynb

Jupyter Notebook producing all statistical tests and summary statistics reported in the paper.

## Scripts

### simulate_reaches.py

Run this script to simulate reaches with a model based on individual subjects' movement times. Writes `data/reaches_simulated.csv` (100 simulated reaches per subject and condition, random seed 0) and `data/kalman_gains_initial_uncertainty.csv` (model Kalman gains per subject and condition). The model parameters are defined in `src/suppression_oc/constants.py` and `src/suppression_oc/oc/models/todorov.py`.

### robustness_oc_gains.py

Run this script to create the dataset for the robustness analysis of the optimal control model parameters.

### sdt.py

Run this script to create the signal detection theory dataset containing d_primes per subject and bin.
