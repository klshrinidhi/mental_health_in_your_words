# Supplemental analyses

These scripts reproduce supplemental analyses reported in the paper from
preserved model-output artifacts. They do not call an LLM or an external API,
and they do not require raw interview audio, video, or transcripts.

## Scripts

- `paired_model_comparisons.py` — paired omnibus and model-pair comparisons.
- `item_profile_association.py` — item-profile association after removing
  participant-level and severity-conditioned effects.
- `run_preserving_item_analysis.py` — complementary item analysis retaining
  the ten prompt-sampling runs through estimation and inference.
- `ordinal_severity_metrics.py` — ordinal error measures and controlled label
  collapse.
- `screen10_analysis.py` — performance at GAD-7 or PHQ-9 total score >= 10.
- `individual_prompt_icl.py` — individual-prompt ICL distributions and
  aggregation sensitivities.
- `mean_confusion_matrices.py` — all-run mean normalized confusion matrices.

## Expected inputs

The scripts read preserved, participant-level model outputs from a local data
directory. The expected layout includes:

```text
DATA_DIR/
├── questionnaire_data_clean.csv
├── prompts_1_11/results.csv
├── prompts_1_12/results.csv
├── prompts_1_13/results.csv
└── prompts_1_17/results.csv
```

`paired_model_comparisons.py` additionally reads each model directory's
`ml_algos_binary/metrics.pkl`. The data files are not included because they
contain restricted participant-level study information.

Run `paired_model_comparisons.py` first when using the validation chain. Its
SHA-256 manifest is checked by `screen10_analysis.py` and
`mean_confusion_matrices.py`; `item_profile_association.py` uses the same
manifest when it is present. Every script accepts or documents its data and
output locations near the top of the file.

The reported seeded draws in all scripts except
`run_preserving_item_analysis.py` require Polars 1.19.0. That complementary
run-preserving analysis used Polars 1.43.2 together with NumPy 2.5.1, pandas
3.0.5, SciPy 1.18.0, and Matplotlib 3.11.1. The other analyses used NumPy
2.5.1; those involving inference used SciPy 1.18.0, and
`individual_prompt_icl.py` used scikit-learn 1.9.0.
