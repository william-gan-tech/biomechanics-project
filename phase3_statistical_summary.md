# Phase 3 Statistical Summary

*Auto-generated -- verify before citing.*

## Phase 3a: General Motion-Reconstruction Variance

*Question: does bone-length scaling reduce cross-subject variance in general reconstruction loss?*

Conditions found: ['unscaled', 'scaled', 'zscore_only']

### Unscaled vs. Scaled

- Paired skaters: **7** (dropped 0 not present in both conditions)
- Mean difference: `0.11296`
- Median difference: `0.00194`
- Wilcoxon signed-rank test: statistic=`13.000`, p=`0.9375`
- **Statistically significant at α=0.05: NO**
  - ⚠️ n=7 is very small -- treat this p-value as suggestive, not conclusive. A non-significant result here does NOT prove there's no effect, only that this sample can't detect one reliably.

| Skater | unscaled | scaled |
|---|---|---|
| Haralds Silovs | 0.2944 | 0.4333 |
| Jan Blokhuijsen | 0.1183 | 0.0515 |
| Jorrit Bergsma | 0.1347 | 0.1367 |
| Mia Manganello Kilburg | 0.4359 | 1.3596 |
| Patrick Meek | 0.2546 | 0.1982 |
| Ragne Wiklund | 0.5429 | 0.3550 |
| Sven Kramer | 0.1230 | 0.1602 |

### Unscaled vs. Z-Score-Only

- Paired skaters: **7** (dropped 0 not present in both conditions)
- Mean difference: `0.01565`
- Median difference: `-0.01414`
- Wilcoxon signed-rank test: statistic=`14.000`, p=`1.0000`
- **Statistically significant at α=0.05: NO**
  - ⚠️ n=7 is very small -- treat this p-value as suggestive, not conclusive. A non-significant result here does NOT prove there's no effect, only that this sample can't detect one reliably.

| Skater | unscaled | zscore_only |
|---|---|---|
| Haralds Silovs | 0.2944 | 0.1925 |
| Jan Blokhuijsen | 0.1183 | 0.0445 |
| Jorrit Bergsma | 0.1347 | 0.1977 |
| Mia Manganello Kilburg | 0.4359 | 0.4878 |
| Patrick Meek | 0.2546 | 0.2058 |
| Ragne Wiklund | 0.5429 | 0.7761 |
| Sven Kramer | 0.1230 | 0.1089 |

### Z-Score-Only vs. Scaled (does bone geometry add anything beyond standardization?)

- Paired skaters: **7** (dropped 0 not present in both conditions)
- Mean difference: `0.09732`
- Median difference: `0.00697`
- Wilcoxon signed-rank test: statistic=`12.000`, p=`0.8125`
- **Statistically significant at α=0.05: NO**
  - ⚠️ n=7 is very small -- treat this p-value as suggestive, not conclusive. A non-significant result here does NOT prove there's no effect, only that this sample can't detect one reliably.

| Skater | zscore_only | scaled |
|---|---|---|
| Haralds Silovs | 0.1925 | 0.4333 |
| Jan Blokhuijsen | 0.0445 | 0.0515 |
| Jorrit Bergsma | 0.1977 | 0.1367 |
| Mia Manganello Kilburg | 0.4878 | 1.3596 |
| Patrick Meek | 0.2058 | 0.1982 |
| Ragne Wiklund | 0.7761 | 0.3550 |
| Sven Kramer | 0.1089 | 0.1602 |

## Phase 3b: Cross-Subject Fatigue-Detection Separability

*Question: does bone-length scaling improve an autoencoder's ability to separate fresh from fatigued movement in an unseen athlete?*

Conditions found: ['unscaled', 'scaled', 'zscore_only']

**Effective sample size per condition:**

- `scaled`: n=7 (Sven Kramer, Patrick Meek, Haralds Silovs, Ragne Wiklund, Mia Manganello Kilburg, Jorrit Bergsma, Jan Blokhuijsen)
- `unscaled`: n=7 (Sven Kramer, Patrick Meek, Haralds Silovs, Ragne Wiklund, Mia Manganello Kilburg, Jorrit Bergsma, Jan Blokhuijsen)
- `zscore_only`: n=7 (Sven Kramer, Patrick Meek, Haralds Silovs, Ragne Wiklund, Mia Manganello Kilburg, Jorrit Bergsma, Jan Blokhuijsen)

### Unscaled vs. Scaled (separability gap)

- Paired skaters: **7** (dropped 0 not present in both conditions)
- Mean difference: `0.27603`
- Median difference: `0.02676`
- Wilcoxon signed-rank test: statistic=`12.000`, p=`0.8125`
- **Statistically significant at α=0.05: NO**
  - ⚠️ n=7 is very small -- treat this p-value as suggestive, not conclusive. A non-significant result here does NOT prove there's no effect, only that this sample can't detect one reliably.

| Skater | unscaled | scaled |
|---|---|---|
| Haralds Silovs | 0.3008 | 0.3285 |
| Jan Blokhuijsen | -0.2545 | 0.0288 |
| Jorrit Bergsma | 0.2595 | 0.2863 |
| Mia Manganello Kilburg | 0.5231 | 2.6023 |
| Patrick Meek | 0.0907 | -0.1601 |
| Ragne Wiklund | 0.2392 | 0.0856 |
| Sven Kramer | -0.2568 | -0.3373 |

### Unscaled vs. Z-Score-Only (separability gap)

- Paired skaters: **7** (dropped 0 not present in both conditions)
- Mean difference: `0.16680`
- Median difference: `0.15485`
- Wilcoxon signed-rank test: statistic=`9.000`, p=`0.4688`
- **Statistically significant at α=0.05: NO**
  - ⚠️ n=7 is very small -- treat this p-value as suggestive, not conclusive. A non-significant result here does NOT prove there's no effect, only that this sample can't detect one reliably.

| Skater | unscaled | zscore_only |
|---|---|---|
| Haralds Silovs | 0.3008 | 0.0407 |
| Jan Blokhuijsen | -0.2545 | -0.0001 |
| Jorrit Bergsma | 0.2595 | 0.6246 |
| Mia Manganello Kilburg | 0.5231 | 0.4488 |
| Patrick Meek | 0.0907 | -0.1051 |
| Ragne Wiklund | 0.2392 | 1.1626 |
| Sven Kramer | -0.2568 | -0.1020 |

### Z-Score-Only vs. Scaled (separability gap)

- Paired skaters: **7** (dropped 0 not present in both conditions)
- Mean difference: `0.10923`
- Median difference: `-0.05501`
- Wilcoxon signed-rank test: statistic=`12.000`, p=`0.8125`
- **Statistically significant at α=0.05: NO**
  - ⚠️ n=7 is very small -- treat this p-value as suggestive, not conclusive. A non-significant result here does NOT prove there's no effect, only that this sample can't detect one reliably.

| Skater | zscore_only | scaled |
|---|---|---|
| Haralds Silovs | 0.0407 | 0.3285 |
| Jan Blokhuijsen | -0.0001 | 0.0288 |
| Jorrit Bergsma | 0.6246 | 0.2863 |
| Mia Manganello Kilburg | 0.4488 | 2.6023 |
| Patrick Meek | -0.1051 | -0.1601 |
| Ragne Wiklund | 1.1626 | 0.0856 |
| Sven Kramer | -0.1020 | -0.3373 |

### Fraction of skaters with a positive separability gap (any fatigue signal)

- `unscaled`: 71% of held-out skaters (5/7)
- `scaled`: 71% of held-out skaters (5/7)
- `zscore_only`: 57% of held-out skaters (4/7)

## How to read this report

- A non-significant p-value with n≈7 is NOT proof of 'no effect' -- it means this sample is too small to distinguish a real small/moderate effect from noise. Report p-values honestly alongside the sample size, don't claim a null result is proven.
- The 'Z-Score-Only vs. Scaled' comparisons are the most directly informative for your core question: they isolate whether bone-length GEOMETRY specifically adds value beyond what simple image-size normalization and standard z-scoring already provide.
