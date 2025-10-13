
Im trying to forecast the influenza incidence rates in Kreisen in Germany using weekly casedata in a GNN. For now, in each epoch, for each timestep, I'm adding the mse between predictions and ground truth, and average those out over all timesteps (Datapoints) so that I have an averaged mse per epoch.

Does that make sense? I'm thinking that I should get the entire predictions, and then get a loss metric that represents the correlation coefficients (increase is an actual increase) and maybe even punish the model upon mirrorring the ground truth too much using the most recent lagged value as most important predcitor. Any thoughts?

Your current approach of averaging MSE over timesteps is reasonable and standard, but your intuitions about its limitations are spot-on. Let me break down some thoughts:

Current MSE Approach
Averaging MSE across timesteps per epoch is fine for basic training, but it has weaknesses for time series forecasting:

It treats all errors equally (missing a peak by 10 cases = being off by 10 during a trough)
It doesn't capture temporal dynamics or trend-following ability
It can encourage "safe" predictions that lag reality
Your Proposed Improvements
1. Correlation-based metrics - Excellent idea Consider adding a component that measures how well your predictions track the direction of change:

Pearson correlation between predictions and ground truth across the forecast horizon
Directional accuracy: What % of time does your model correctly predict increase/decrease?
You could create a composite loss: loss = MSE - λ * correlation (where λ is a weighting factor)
2. Penalizing naive lag-based predictions - Smart! This addresses "lazy" models that just shift recent values forward. Options:

Compute predictions from a naive baseline (e.g., "tomorrow = today")
Add a penalty term if your predictions are too correlated with the t-1 lag
Use a skill score: 1 - (your_MSE / naive_MSE) to ensure you beat simple baselines
Practical Suggestions
Combined loss function:

loss = MSE + α * (1 - correlation) + β * max(0, corr(pred, lag1) - threshold)
Where:

α weights the correlation penalty
β penalizes excessive correlation with lag-1 values (only if above some threshold like 0.9)
Alternative metrics to track (not necessarily in loss):

MAPE (Mean Absolute Percentage Error) - handles scale differences between Kreisen
Peak timing error - How many weeks off are your predicted peaks?
Peak magnitude error - How accurate are the peak heights?
Multi-horizon consideration: If you're forecasting multiple weeks ahead, you might weight recent timesteps more heavily:

weighted_loss = Σ(w_t * MSE_t) where w_t decreases with forecast horizon
The correlation idea is particularly good for epidemiological forecasting where capturing the trend matters more than exact values. Would you like help implementing any of these in your training loop?






