"""
Quantitative Accuracy Comparison: Baseline vs EWMA Ensemble
============================================================
QA = 1 - |predicted_price - actual_price| / |actual_price|

This measures how CLOSE our price predictions are, not just direction.
"""
import pandas as pd
import numpy as np
import os

# Import project configuration
from config import PARQUET_PATH, QA_RESULTS, TRAIN_END, TEST_START, PROJECT_ID, PROJECT_NAME

print("=" * 80)
print("QUANTITATIVE ACCURACY: BASELINE vs EWMA ENSEMBLE")
print("=" * 80)
print(f"Project: {PROJECT_ID} - {PROJECT_NAME}")

# Load data
print("\n[1] Loading data...")
if not os.path.exists(PARQUET_PATH):
    print(f"\n❌ ERROR: Data file not found: {PARQUET_PATH}")
    exit(1)

df = pd.read_parquet(PARQUET_PATH)
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
print(f"    Total records: {len(df):,}")

# Constants
TRAIN_END = pd.Timestamp(TRAIN_END)
TEST_START = pd.Timestamp(TEST_START)
TOP_N = 5

optimal_params = {
    30: {'lookback': 4, 'alpha': 0.2},
    60: {'lookback': 6, 'alpha': 0.3},
    90: {'lookback': 6, 'alpha': 0.5},
    180: {'lookback': 6, 'alpha': 0.5},
    270: {'lookback': 5, 'alpha': 0.5},
    360: {'lookback': 3, 'alpha': 0.2},
    450: {'lookback': 10, 'alpha': 0.5},
}

# ============= HELPER FUNCTIONS =============
def ewma_weights(accuracies, alpha=0.3):
    n = len(accuracies)
    if n == 0:
        return []
    weights = []
    for i in range(n):
        w = alpha * ((1 - alpha) ** (n - 1 - i))
        weights.append(w)
    total = sum(weights)
    return [w / total for w in weights]

def get_top_models(df_horizon, n=5):
    """Get top N models based on training DA"""
    train_data = df_horizon[df_horizon['time'] <= TRAIN_END]
    model_performance = []
    for symbol in df_horizon['symbol'].unique():
        train_model = train_data[train_data['symbol'] == symbol]
        pred_dir = np.sign(train_model['close_predict'] - train_model['target_var_price'])
        actual_dir = np.sign(train_model['yn_actual'])
        mask = (pred_dir != 0) & (actual_dir != 0)
        if mask.sum() >= 100:
            correct = (pred_dir[mask] == actual_dir[mask]).sum()
            train_da = correct / mask.sum()
            model_performance.append({'symbol': symbol, 'train_da': train_da})
    perf_df = pd.DataFrame(model_performance).sort_values('train_da', ascending=False)
    return perf_df.head(n)['symbol'].tolist()

def calculate_quantitative_accuracy(predicted_price, actual_price):
    """
    QA = 1 - |predicted - actual| / |actual|
    Returns accuracy as percentage (0-100%)
    """
    if actual_price == 0:
        return 0
    qa = 1 - abs(predicted_price - actual_price) / abs(actual_price)
    return max(0, qa) * 100  # Clamp to 0-100%

def calculate_baseline_qa(df_horizon, n_predict):
    """Calculate Quantitative Accuracy for baseline (average of all models)"""
    test_data = df_horizon[df_horizon['time'] >= TEST_START].copy()
    
    # Group by date and average predictions
    daily_avg = test_data.groupby('time').agg({
        'close_predict': 'mean',  # Average predicted price across all models
        'target_var_price': 'first',  # Current price (same for all)
    }).reset_index()
    
    # We need actual future price - get it from yn_actual
    # yn_actual is the log return, so actual_future_price = current_price * exp(yn_actual)
    # But we need to look n_predict days ahead
    
    results = []
    dates = sorted(test_data['time'].unique())
    
    for i, date in enumerate(dates):
        day_data = test_data[test_data['time'] == date]
        
        # Average prediction across all models
        avg_close_predict = day_data['close_predict'].mean()
        current_price = day_data['target_var_price'].iloc[0]
        
        # Get actual price n_predict days later
        future_idx = i + n_predict
        if future_idx >= len(dates):
            continue
        future_date = dates[future_idx]
        future_data = test_data[test_data['time'] == future_date]
        if len(future_data) == 0:
            continue
        actual_future_price = future_data['target_var_price'].iloc[0]
        
        qa = calculate_quantitative_accuracy(avg_close_predict, actual_future_price)
        results.append({
            'date': date,
            'predicted': avg_close_predict,
            'actual': actual_future_price,
            'qa': qa
        })
    
    return pd.DataFrame(results)

def run_ewma_ensemble_qa(df_horizon, top_models, lookback, alpha, n_predict):
    """Run EWMA ensemble and calculate Quantitative Accuracy"""
    df_h = df_horizon[df_horizon['symbol'].isin(top_models)].copy()
    df_h = df_h[['time', 'symbol', 'close_predict', 'target_var_price', 'yn_actual']].copy()
    
    dates = sorted(df_h['time'].unique())
    results = []
    model_recent_correct = {m: [] for m in top_models}
    
    for i, date in enumerate(dates):
        if i < lookback:
            continue
        if date < TEST_START:
            # Still need to track model performance during training
            day_data = df_h[df_h['time'] == date]
            for model in top_models:
                model_day = day_data[day_data['symbol'] == model]
                if len(model_day) == 0:
                    continue
                close_pred = float(model_day['close_predict'].values[0])
                target_price = float(model_day['target_var_price'].values[0])
                yn_actual = float(model_day['yn_actual'].values[0])
                pred_dir = np.sign(close_pred - target_price)
                act_dir = np.sign(yn_actual)
                if pred_dir != 0 and act_dir != 0:
                    was_correct = 1 if pred_dir == act_dir else 0
                    model_recent_correct[model].append(was_correct)
            continue
            
        day_data = df_h[df_h['time'] == date]
        if len(day_data) < len(top_models):
            continue
        
        preds = {}
        pred_prices = {}
        
        for model in top_models:
            model_day = day_data[day_data['symbol'] == model]
            if len(model_day) == 0:
                continue
            close_pred = float(model_day['close_predict'].values[0])
            target_price = float(model_day['target_var_price'].values[0])
            yn_actual = float(model_day['yn_actual'].values[0])
            pred_dir = np.sign(close_pred - target_price)
            act_dir = np.sign(yn_actual)
            if pred_dir != 0 and act_dir != 0:
                preds[model] = pred_dir
                pred_prices[model] = close_pred
        
        if len(preds) < 2:
            continue
        
        # Calculate EWMA weights
        model_weights = {}
        for model in preds.keys():
            recent = model_recent_correct.get(model, [])[-lookback:]
            if len(recent) >= lookback:
                ewma_w = ewma_weights(recent, alpha)
                weighted_acc = sum(r * w for r, w in zip(recent, ewma_w))
                model_weights[model] = weighted_acc
            else:
                model_weights[model] = 0.5
        
        total_weight = sum(model_weights.values()) + 0.001
        norm_weights = {m: w / total_weight for m, w in model_weights.items()}
        
        # EWMA weighted price prediction
        ewma_price_pred = sum(pred_prices[m] * norm_weights[m] for m in preds.keys())
        current_price = day_data['target_var_price'].iloc[0]
        
        # Get actual price n_predict days later
        future_idx = i + n_predict
        if future_idx >= len(dates):
            # Update model tracking and continue
            for model in preds.keys():
                was_correct = 1 if preds[model] == np.sign(day_data[day_data['symbol'] == model]['yn_actual'].values[0]) else 0
                model_recent_correct[model].append(was_correct)
            continue
            
        future_date = dates[future_idx]
        future_data = df_h[df_h['time'] == future_date]
        if len(future_data) == 0:
            for model in preds.keys():
                was_correct = 1 if preds[model] == np.sign(day_data[day_data['symbol'] == model]['yn_actual'].values[0]) else 0
                model_recent_correct[model].append(was_correct)
            continue
        actual_future_price = future_data['target_var_price'].iloc[0]
        
        qa = calculate_quantitative_accuracy(ewma_price_pred, actual_future_price)
        
        results.append({
            'date': date,
            'predicted': ewma_price_pred,
            'actual': actual_future_price,
            'qa': qa
        })
        
        # Update model tracking
        for model in preds.keys():
            model_day_data = day_data[day_data['symbol'] == model]
            if len(model_day_data) > 0:
                yn_actual = model_day_data['yn_actual'].values[0]
                was_correct = 1 if preds[model] == np.sign(yn_actual) else 0
                model_recent_correct[model].append(was_correct)
    
    return pd.DataFrame(results)

# ============= MAIN COMPARISON =============
print("\n[2] Calculating Quantitative Accuracy for each horizon...")
print("=" * 80)

all_results = []

for horizon in [30, 60, 90, 180]:  # Main horizons
    params = optimal_params[horizon]
    lookback = params['lookback']
    alpha = params['alpha']
    
    print(f"\n{'='*80}")
    print(f"  HORIZON: {horizon} DAYS")
    print(f"{'='*80}")
    
    df_h = df[df['n_predict'] == horizon].copy()
    
    # Calculate BASELINE QA
    print("  Calculating Baseline QA...")
    baseline_qa_df = calculate_baseline_qa(df_h, horizon)
    
    # Calculate EWMA QA
    print("  Calculating EWMA QA...")
    top_models = get_top_models(df_h, TOP_N)
    ewma_qa_df = run_ewma_ensemble_qa(df_h, top_models, lookback, alpha, horizon)
    
    if len(baseline_qa_df) == 0 or len(ewma_qa_df) == 0:
        print("  Insufficient data, skipping...")
        continue
    
    baseline_avg_qa = baseline_qa_df['qa'].mean()
    ewma_avg_qa = ewma_qa_df['qa'].mean()
    
    baseline_median_qa = baseline_qa_df['qa'].median()
    ewma_median_qa = ewma_qa_df['qa'].median()
    
    # Calculate MAPE (Mean Absolute Percentage Error) as well
    baseline_mape = ((baseline_qa_df['predicted'] - baseline_qa_df['actual']).abs() / baseline_qa_df['actual'].abs()).mean() * 100
    ewma_mape = ((ewma_qa_df['predicted'] - ewma_qa_df['actual']).abs() / ewma_qa_df['actual'].abs()).mean() * 100
    
    print(f"\n  {'Metric':<30} {'BASELINE':>15} {'EWMA':>15} {'Diff':>12}")
    print(f"  {'-'*75}")
    print(f"  {'Avg Quantitative Accuracy':<30} {baseline_avg_qa:>14.2f}% {ewma_avg_qa:>14.2f}% {ewma_avg_qa - baseline_avg_qa:>+11.2f}%")
    print(f"  {'Median Quantitative Accuracy':<30} {baseline_median_qa:>14.2f}% {ewma_median_qa:>14.2f}% {ewma_median_qa - baseline_median_qa:>+11.2f}%")
    print(f"  {'Mean Abs % Error (MAPE)':<30} {baseline_mape:>14.2f}% {ewma_mape:>14.2f}% {ewma_mape - baseline_mape:>+11.2f}%")
    print(f"  {'Sample Size':<30} {len(baseline_qa_df):>15} {len(ewma_qa_df):>15}")
    
    all_results.append({
        'horizon': horizon,
        'baseline_qa': baseline_avg_qa,
        'ewma_qa': ewma_avg_qa,
        'qa_improvement': ewma_avg_qa - baseline_avg_qa,
        'baseline_mape': baseline_mape,
        'ewma_mape': ewma_mape,
        'mape_improvement': baseline_mape - ewma_mape,  # Lower is better
    })

# ============= SUMMARY =============
print("\n")
print("=" * 80)
print("SUMMARY: QUANTITATIVE ACCURACY COMPARISON")
print("=" * 80)

summary_df = pd.DataFrame(all_results)

print("\n┌───────────────────────────────────────────────────────────────────────────────┐")
print("│                    QUANTITATIVE ACCURACY COMPARISON                           │")
print("│         (How close are predicted prices to actual prices?)                    │")
print("├──────────┬──────────────────┬──────────────────┬───────────────────────────────┤")
print("│ Horizon  │     BASELINE     │       EWMA       │          IMPROVEMENT          │")
print("├──────────┼──────────────────┼──────────────────┼───────────────────────────────┤")
for _, row in summary_df.iterrows():
    imp_sign = "+" if row['qa_improvement'] >= 0 else ""
    print(f"│  {row['horizon']:>4}d   │      {row['baseline_qa']:>6.2f}%      │      {row['ewma_qa']:>6.2f}%      │          {imp_sign}{row['qa_improvement']:>5.2f} pp              │")
print("└──────────┴──────────────────┴──────────────────┴───────────────────────────────┘")

print("\n┌───────────────────────────────────────────────────────────────────────────────┐")
print("│                    MEAN ABSOLUTE % ERROR (MAPE)                               │")
print("│                   (Lower is better - prediction error)                        │")
print("├──────────┬──────────────────┬──────────────────┬───────────────────────────────┤")
print("│ Horizon  │     BASELINE     │       EWMA       │       ERROR REDUCTION         │")
print("├──────────┼──────────────────┼──────────────────┼───────────────────────────────┤")
for _, row in summary_df.iterrows():
    imp_sign = "+" if row['mape_improvement'] >= 0 else ""
    print(f"│  {row['horizon']:>4}d   │      {row['baseline_mape']:>6.2f}%      │      {row['ewma_mape']:>6.2f}%      │          {imp_sign}{row['mape_improvement']:>5.2f} pp              │")
print("└──────────┴──────────────────┴──────────────────┴───────────────────────────────┘")

# Averages
avg_qa_baseline = summary_df['baseline_qa'].mean()
avg_qa_ewma = summary_df['ewma_qa'].mean()
avg_mape_baseline = summary_df['baseline_mape'].mean()
avg_mape_ewma = summary_df['ewma_mape'].mean()

print(f"\n  📊 OVERALL AVERAGES:")
print(f"     • Baseline Quantitative Accuracy: {avg_qa_baseline:.2f}%")
print(f"     • EWMA Quantitative Accuracy:     {avg_qa_ewma:.2f}%")
print(f"     • Improvement:                    {avg_qa_ewma - avg_qa_baseline:+.2f} pp")
print(f"")
print(f"     • Baseline MAPE:                  {avg_mape_baseline:.2f}%")
print(f"     • EWMA MAPE:                      {avg_mape_ewma:.2f}%")
print(f"     • Error Reduction:                {avg_mape_baseline - avg_mape_ewma:+.2f} pp")

print("\n" + "=" * 80)
output_path = QA_RESULTS
summary_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")
print("=" * 80)

