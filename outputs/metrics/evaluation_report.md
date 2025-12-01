# Brooklyn Crosswalk Detection - Evaluation Report

Generated: 2025-11-30 15:42:18

## Summary
- **Tiles evaluated**: 120
- **Mean IoU**: 0.0000 (±0.0000)
- **Mean F1**: 0.0000 (±0.0000)
- **Mean Precision**: 0.0000
- **Mean Recall**: 0.0000

## Calibration
- **Mean ECE**: 0.1864
- **Mean MCE**: 0.9500

## Recommendations
- **Optimal Threshold**: 0.30

## Per-Tile Statistics

|       |   iou |   f1 |   precision |   recall |          ece |   optimal_threshold |   tp_pixels |   fp_pixels |   fn_pixels |
|:------|------:|-----:|------------:|---------:|-------------:|--------------------:|------------:|------------:|------------:|
| count |   120 |  120 |         120 |      120 | 120          |       120           |         120 |         120 |      120    |
| mean  |     0 |    0 |           0 |        0 |   0.186436   |         0.3         |           0 |           0 |     1622.91 |
| std   |     0 |    0 |           0 |        0 |   0.00473375 |         5.57439e-17 |           0 |           0 |     1279.82 |
| min   |     0 |    0 |           0 |        0 |   0.171325   |         0.3         |           0 |           0 |        0    |
| 25%   |     0 |    0 |           0 |        0 |   0.183247   |         0.3         |           0 |           0 |      430.75 |
| 50%   |     0 |    0 |           0 |        0 |   0.186374   |         0.3         |           0 |           0 |     1677    |
| 75%   |     0 |    0 |           0 |        0 |   0.190366   |         0.3         |           0 |           0 |     2475.25 |
| max   |     0 |    0 |           0 |        0 |   0.193716   |         0.3         |           0 |           0 |     5696    |

## Notes
- IoU (Intersection over Union) measures overlap between prediction and ground truth
- F1 Score balances precision and recall
- ECE (Expected Calibration Error) measures how well probabilities match actual frequencies
- Lower ECE indicates better calibration