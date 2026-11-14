# StatF

[![DOI](https://zenodo.org/badge/465050837.svg)](https://doi.org/10.5281/zenodo.18848634)

Replication package for the paper "Predicting the Load-Time Energy Consumption of Web Apps from Static Design Features" by Muhammad Sohaib Ayub, Nimrah Mustafa, Umair ul Hassan, Muhammad Asad Khan, Safiullah Faizullah and Imdad Ullah Khan.

The repository holds the raw data collected for the top Alexa websites, the dataset used in the paper, and the scripts that produce the reported results.

## Contents

| Folder or file | What it holds |
|---|---|
| `Lighthouse/` | Google Lighthouse reports of the websites (JSON, CSV and HTML) |
| `WebPageTest/` | WebPageTest summary and aggregate results |
| `PowerAPI/` | SmartWatts power readings of the page loads in Chrome, three runs per website |
| `analysis/` | Dataset of 689 websites and the scripts and result files of the paper |
| `FeatureBasedDistributionOfPower.py`, `LHperfScrAndPow.py`, `correlationOfFeaturesWithPower.py` | Plotting scripts of the first version of the analysis |
| `*DataColumnsList.txt` | Column lists used when extracting data from the tools |

## Dataset

`analysis/sites_dataset.csv` has one row per website.

| Columns | Meaning |
|---|---|
| first 33 columns | static features from WebPageTest |
| next 15 columns | static features from Google Lighthouse |
| `power1`, `power2`, `power3`, `avgPow` | sum of the power readings of each page load and their average |
| `energyJ` | energy of the page load in joules |
| `meanPowerW`, `durationS` | average power in watts and active time in seconds |
| `LOGavgPowClusters3`, `4`, `5` | energy tier labels from k-means on log energy (empty for the 4 outliers) |
| `performance-score` and the six metric columns | Lighthouse performance score and its six metrics |
| `url` | website |

SmartWatts reports one power reading for every sampling interval of 0.5 seconds. The energy of a page load is the sum of its readings multiplied by 0.5. The column `avgPow`, which the older scripts use, is that sum before the multiplication, so the energy in joules is half of it.

`analysis/feature_selection.csv` lists the candidate features of both tools and whether each one was selected or why it was left out.

## Reproducing the results

Python 3 with the packages in `analysis/requirements.txt` is needed. Run the scripts from inside the `analysis` folder.

| Script | What it does | Output |
|---|---|---|
| `build_dataset.py` | Builds the dataset from the raw files | `sites_dataset.csv` |
| `classification_cv.py` | Energy tier classification for StatF and both PerfS baselines with repeated 10-fold cross-validation and the corrected t-test | `cv_summary.csv`, `significance_tests.csv`, `confusion_matrix_all_rf.csv` |
| `combined.py` | Static features together with the performance metrics | printed |
| `robustness.py` | Regression on log energy, split of energy into power and time, tier classification for each | `regression_cv.csv`, `tiers_power_time_cv.csv` |
| `power_tier_tests.py` | Significance tests for the average power tiers and the regression | printed |
| `k_sensitivity_cv.py` | Classification with 3, 4 and 5 tiers | `k_sensitivity.txt` |
| `feature_importance_cv.py` | Random Forest feature importance averaged over the cross-validation models | `feature_importance_cv.csv` |

`analysis/results_summary.txt` lists the main numbers reported in the paper.

## License

The code is released under the MIT License (see `LICENSE`). The data is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Citation

If you use the data or the code, please cite the paper and this repository (see `CITATION.cff`).

## Contact

Muhammad Sohaib Ayub, muhammadsohaib.ayub@universityofgalway.ie
