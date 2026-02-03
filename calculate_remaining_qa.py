"""
Calculate QA for remaining horizons: 270d, 360d, 450d
Using the optimal parameters already found for DA
"""
import pandas as pd
import numpy as np
import os

# Import project configuration
from config import PARQUET_PATH, TRAIN_END, TEST_START, PROJECT_ID, PROJECT_NAME

print("=" * 60)
print("QA CALCULATION FOR 270d, 360d, 450d")
print("=" * 60)
print(f"Project: {PROJECT_ID} - {PROJECT_NAME}")

# Load data
if not os.path.exists(PARQUET_PATH):
    print(f"\n❌ ERROR: Data file not found: {PARQUET_PATH}")
    exit(1)

df = pd.read_parquet(PARQUET_PATH)
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
print(f"Loaded {len(df):,} records")

# Constants
TRAIN_END = pd.Timestamp(TRAIN_END)
TEST_START = pd.Timestamp(TEST_START)

# Optimal parameters from DA optimization (from dashboard)
HORIZON_PARAMS = {
    270: {'L': 5, 'alpha': 0.5, 'N': 5},
    360: {'L': 3, 'alpha': 0.2, 'N': 5},
    450: {'L': 10, 'alpha': 0.5, 'N': 5},
}

def ewma_weights(accuracies, alpha=0.3):
    n = len(accuracies)
    if n == 0:
        return []
    weights = [(alpha * ((1 - alpha) ** (n - 1 - i))) for i in range(n)]
    total = sum(weights)
    return [w / total for w in weights] if total > 0 else [1/n] * n

def calculate_qa(predicted_price, actual_price):
    if actual_price == 0:
        return 0
    return max(0, 1 - abs(predicted_price - actual_price) / abs(actual_price))

def get_top_models_by_da(df_horizon, n=5):
    """Get top N models by training DA"""
    train_data = df_horizon[df_horizon['time'] <= TRAIN_END]
    model_da = []
    
    for symbol in df_horizon['symbol'].unique():
        train_model = train_data[train_data['symbol'] == symbol]
        if len(train_model) < 50:
            continue
        
        pred_dir = np.sign(train_model['close_predict'].values - train_model['target_var_price'].values)
        actual_dir = np.sign(train_model['yn_actual'].values)
        mask = (pred_dir != 0) & (actual_dir != 0)
        if mask.sum() < 30:
            continue
        da = (pred_dir[mask] == actual_dir[mask]).sum() / mask.sum()
        model_da.append((symbol, da))
    
    model_da.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in model_da[:n]]

def calculate_baseline_qa(df_horizon, n_predict):
    """Calculate baseline QA (average of all models)"""
    test_data = df_horizon[df_horizon['time'] >= TEST_START].copy()
    dates = sorted(test_data['time'].unique())
    qa_values = []
    
    for i, date in enumerate(dates):
        day_data = test_data[test_data['time'] == date]
        avg_pred = day_data['close_predict'].mean()
        
        future_idx = i + n_predict
        if future_idx < len(dates):
            future_date = dates[future_idx]
            future_data = test_data[test_data['time'] == future_date]
            if len(future_data) > 0:
                actual_price = future_data['target_var_price'].iloc[0]
                qa = calculate_qa(avg_pred, actual_price)
                qa_values.append(qa)
    
    return np.mean(qa_values) * 100 if qa_values else 0

def calculate_ewma_qa(df_horizon, top_models, lookback, alpha, n_predict):
    """Calculate EWMA ensemble QA"""
    df_h = df_horizon[df_horizon['symbol'].isin(top_models)].copy()
    df_h = df_h[['time', 'symbol', 'close_predict', 'target_var_price', 'yn_actual']].copy()
    
    dates = sorted(df_h['time'].unique())
    model_recent_correct = {m: [] for m in top_models}
    qa_values = []
    
    for i, date in enumerate(dates):
        if i < lookback:
            continue
            
        day_data = df_h[df_h['time'] == date]
        
        # Track model performance before test period
        if date < TEST_START:
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
        
        if len(day_data) < len(top_models) // 2:
            continue
        
        pred_prices = {}
        preds = {}
        
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
                model_weights[model] = max(0.01, weighted_acc)
            else:
                model_weights[model] = 0.5
        
        total_weight = sum(model_weights.values()) + 0.001
        norm_weights = {m: w / total_weight for m, w in model_weights.items()}
        
        # EWMA weighted price prediction
        ewma_price_pred = sum(pred_prices[m] * norm_weights[m] for m in preds.keys())
        
        # Get actual future price for QA
        future_idx = i + n_predict
        if future_idx < len(dates):
            future_date = dates[future_idx]
            future_data = df_h[df_h['time'] == future_date]
            if len(future_data) > 0:
                actual_future_price = future_data['target_var_price'].iloc[0]
                qa = calculate_qa(ewma_price_pred, actual_future_price)
                qa_values.append(qa)
        
        # Update model tracking
        for model in preds.keys():
            model_day_data = day_data[day_data['symbol'] == model]
            if len(model_day_data) > 0:
                yn_actual = model_day_data['yn_actual'].values[0]
                was_correct = 1 if preds[model] == np.sign(yn_actual) else 0
                model_recent_correct[model].append(was_correct)
    
    return np.mean(qa_values) * 100 if qa_values else 0

# Calculate QA for each horizon
print("\n" + "=" * 60)
results = []

for horizon, params in HORIZON_PARAMS.items():
    print(f"\nProcessing {horizon}d horizon...")
    
    df_h = df[df['n_predict'] == horizon].copy()
    
    if len(df_h) == 0:
        print(f"  No data for {horizon}d horizon!")
        continue
    
    # Get top models
    top_models = get_top_models_by_da(df_h, n=params['N'])
    print(f"  Top {len(top_models)} models selected")
    
    # Calculate baseline QA
    print(f"  Calculating baseline QA...")
    baseline_qa = calculate_baseline_qa(df_h, horizon)
    
    # Calculate EWMA QA
    print(f"  Calculating EWMA QA (L={params['L']}, α={params['alpha']})...")
    ewma_qa = calculate_ewma_qa(df_h, top_models, params['L'], params['alpha'], horizon)
    
    qa_diff = ewma_qa - baseline_qa
    
    results.append({
        'horizon': horizon,
        'baseline_qa': baseline_qa,
        'ewma_qa': ewma_qa,
        'qa_diff': qa_diff
    })
    
    print(f"  ✅ Baseline QA: {baseline_qa:.2f}%")
    print(f"  ✅ EWMA QA: {ewma_qa:.2f}%")
    print(f"  ✅ Difference: {qa_diff:+.2f} pp")

# Final summary
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print("\n┌──────────┬─────────────┬──────────┬─────────┐")
print("│ Horizon  │ Baseline QA │ EWMA QA  │  QA Δ   │")
print("├──────────┼─────────────┼──────────┼─────────┤")
for r in results:
    sign = "+" if r['qa_diff'] >= 0 else ""
    print(f"│  {r['horizon']:>4}d   │   {r['baseline_qa']:>6.2f}%   │  {r['ewma_qa']:>6.2f}% │ {sign}{r['qa_diff']:>5.2f}  │")
print("└──────────┴─────────────┴──────────┴─────────┘")

print("\n✅ Done! Use these values to update the dashboard.")

