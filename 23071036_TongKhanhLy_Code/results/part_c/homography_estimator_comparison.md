| Estimator           | Data                 |   Inliers |   Total |   RMSE_on_true_inliers_px |   Time_ms |
|:--------------------|:---------------------|----------:|--------:|--------------------------:|----------:|
| Least squares (all) | ratio-test matches   |       693 |     693 |                   18.62   |    0.8085 |
| Least squares (all) | +20% random outliers |       831 |     831 |                  195.3    |    0.8243 |
| RANSAC thr=1px      | ratio-test matches   |       463 |     693 |                    0.4638 |    1.576  |
| RANSAC thr=1px      | +20% random outliers |       469 |     831 |                    0.4751 |    2.483  |
| RANSAC thr=3px      | ratio-test matches   |       642 |     693 |                    0.8494 |    0.6478 |
| RANSAC thr=3px      | +20% random outliers |       623 |     831 |                    0.6273 |    0.9893 |
| RANSAC thr=10px     | ratio-test matches   |       654 |     693 |                    0.8438 |    0.7152 |
| RANSAC thr=10px     | +20% random outliers |       654 |     831 |                    0.847  |    0.7823 |
| LMedS               | ratio-test matches   |       608 |     693 |                    0.5005 |    2.175  |
| LMedS               | +20% random outliers |       616 |     831 |                    0.5966 |    2.31   |