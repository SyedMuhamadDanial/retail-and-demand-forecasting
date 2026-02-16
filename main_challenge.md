# Summary of Forecast Discrepancy Fix
Tried to resolved the issue where forecasted sales were significantly lower than historical sales.

## Problem Statement
The model generated predictions that were ~76% lower than historical sales. This was caused by the model's reliance on lag features (e.g., lag_1), which depend on the previous day's sales. Since the test set has no sales data, these features were NaN (filled as 0), breaking the forecast. Additionally, the test set contained 23% fewer stores than the training set, and the Customers feature was missing from the test data, causing tool errors and leakage.

## Solution
Removed Broken Lags: Deleted the dependency on recursive lag features that require unavailable sales data.
Added Aggregate Features: Implemented Store-DayOfWeek and Store-Month historical averages to provide stable signals.
Feature Alignment: Excluded the Customers column and other training-only fields from the modeling set.
Normalized Diagnostics: Updated the check to compare Per-Store averages rather than total sums to account for varying store counts.
Technical Fix: Removed unicode icons from print statements to fix UnicodeEncodeError in Windows.

## **Reason of Solution**
Historical averages (Store-Day/Store-Month) are "Horizon-Safe"—they don't depend on the immediate past sales. This allows the model to predict accurately for the entire 6-week window without needing recursive feedback. Normalizing by store count ensures that the diagnostic "performance ratio" is mathematically sound.

## **Achievements**
Accuracy Boost: Improved MAPE to 11.00% (89% prediction accuracy on average across all stores).
Stability: The script now runs from start to finish on Windows without encoding or key errors.
Improved Insights: The "Per-Store" performance ratio is now stable, showing a healthy forecast that aligns more with historical trends.

## Before fix:
![alt text](image.png)

## After fix:
![alt text](image-1.png)