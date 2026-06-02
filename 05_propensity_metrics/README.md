## 5. Compute PropMe metrics from traced summaries

PropMe compares non-prefix settings such as `generic` and `specific` against a `prefix` capability setting.

Example:

```bash
python 05_propensity_metrics/compute_propensity_metrics.py \
    --generic-summary outputs/generic_summary.json \
    --specific-summary outputs/specific_summary.json \
    --prefix-summary outputs/prefix_summary.json \
    --metrics avg_nv_recall generations_full_matches_ratio \
    --output outputs/propensity_metrics.json \
    --plot
```

This produces a report comparing each non-prefix setting to the prefix baseline using the current propensity transformation implemented in the codebase.

You can also use existing presets, check them running the following:

```bash
python 05_propensity_metrics/compute_propensity_metrics.py \
    --list
```
