"""
Visualize WHERE baseline vs EWMA got direction RIGHT vs WRONG
This will show the actual DA difference visually
"""
import pandas as pd
import numpy as np
import json
import os

# Import project configuration
from config import PARQUET_PATH, DA_COMPARISON, TRAIN_END, TEST_START, PROJECT_ID, PROJECT_NAME

print("=" * 70)
print("GENERATING DIRECTIONAL ACCURACY VISUALIZATION DATA")
print("=" * 70)
print(f"Project: {PROJECT_ID} - {PROJECT_NAME}")

# Load data
if not os.path.exists(PARQUET_PATH):
    print(f"\n❌ ERROR: Data file not found: {PARQUET_PATH}")
    exit(1)

df = pd.read_parquet(PARQUET_PATH)
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

TRAIN_END = pd.Timestamp(TRAIN_END)
TEST_START = pd.Timestamp(TEST_START)
TOP_N = 5

optimal_params = {
    30: {'lookback': 4, 'alpha': 0.2},
    60: {'lookback': 6, 'alpha': 0.3},
    90: {'lookback': 6, 'alpha': 0.5},
    180: {'lookback': 6, 'alpha': 0.5},
}

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

def generate_da_comparison(df_horizon, horizon, top_models, lookback, alpha):
    """Generate data showing where each method got direction right/wrong"""
    
    # =========== BASELINE ===========
    test_data = df_horizon[df_horizon['time'] >= TEST_START].copy()
    test_data['pred_dir'] = np.sign(test_data['close_predict'] - test_data['target_var_price'])
    test_data['actual_dir'] = np.sign(test_data['yn_actual'])
    valid = test_data[(test_data['pred_dir'] != 0) & (test_data['actual_dir'] != 0)].copy()
    
    # Aggregate by date
    daily_baseline = valid.groupby('time').agg({
        'pred_dir': lambda x: np.sign(x.mean()),
        'actual_dir': 'first',
        'target_var_price': 'first',  # Current price
        'yn_actual': 'first'  # Actual return
    }).reset_index()
    daily_baseline['correct'] = (daily_baseline['pred_dir'] == daily_baseline['actual_dir']).astype(int)
    daily_baseline = daily_baseline.sort_values('time')
    
    baseline_results = []
    for _, row in daily_baseline.iterrows():
        baseline_results.append({
            'date': row['time'].strftime('%Y-%m-%d'),
            'correct': int(row['correct']),
            'actual_dir': int(row['actual_dir']),
            'pred_dir': int(row['pred_dir']),
            'price': float(row['target_var_price'])
        })
    
    # =========== EWMA ===========
    df_h = df_horizon[df_horizon['symbol'].isin(top_models)].copy()
    df_h = df_h[['time', 'symbol', 'close_predict', 'target_var_price', 'yn_actual']].copy()
    
    dates = sorted(df_h['time'].unique())
    ewma_results = []
    model_recent_correct = {m: [] for m in top_models}
    
    for i, date in enumerate(dates):
        if i < lookback:
            continue
        day_data = df_h[df_h['time'] == date]
        if len(day_data) < len(top_models):
            continue
        
        preds = {}
        actual_dir = None
        price = None
        
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
                actual_dir = act_dir
                price = target_price
        
        if len(preds) < 2 or actual_dir is None:
            continue
        
        # EWMA weights
        model_weights = {}
        for model in preds.keys():
            recent = model_recent_correct.get(model, [])[-lookback:]
            if len(recent) >= lookback:
                ew = ewma_weights(recent, alpha)
                weighted_acc = sum(r * w for r, w in zip(recent, ew))
                model_weights[model] = weighted_acc
            else:
                model_weights[model] = 0.5
        
        total_weight = sum(model_weights.values()) + 0.001
        norm_weights = {m: w / total_weight for m, w in model_weights.items()}
        
        ewma_vote = sum(preds[m] * norm_weights[m] for m in preds.keys())
        ensemble_pred = np.sign(ewma_vote)
        
        if date >= TEST_START:
            ewma_results.append({
                'date': date.strftime('%Y-%m-%d'),
                'correct': 1 if ensemble_pred == actual_dir else 0,
                'actual_dir': int(actual_dir),
                'pred_dir': int(ensemble_pred),
                'price': float(price)
            })
        
        # Update tracking
        for model in preds.keys():
            was_correct = 1 if preds[model] == actual_dir else 0
            model_recent_correct[model].append(was_correct)
    
    return baseline_results, ewma_results

# Generate for all horizons
da_data = {}

for horizon, params in optimal_params.items():
    print(f"\nProcessing {horizon}d...")
    df_h = df[df['n_predict'] == horizon].copy()
    top_models = get_top_models(df_h, TOP_N)
    
    baseline_results, ewma_results = generate_da_comparison(
        df_h, horizon, top_models, params['lookback'], params['alpha']
    )
    
    h_key = f"{horizon}d"
    da_data[h_key] = {
        'baseline': baseline_results,
        'ewma': ewma_results
    }
    
    # Calculate DA
    if baseline_results:
        base_da = sum(r['correct'] for r in baseline_results) / len(baseline_results) * 100
    else:
        base_da = 0
    
    if ewma_results:
        ewma_da = sum(r['correct'] for r in ewma_results) / len(ewma_results) * 100
    else:
        ewma_da = 0
    
    base_wrong = sum(1 for r in baseline_results if r['correct'] == 0)
    ewma_wrong = sum(1 for r in ewma_results if r['correct'] == 0)
    
    print(f"  Baseline: {base_da:.1f}% DA ({base_wrong} wrong predictions)")
    print(f"  EWMA:     {ewma_da:.1f}% DA ({ewma_wrong} wrong predictions)")

# Save
with open(DA_COMPARISON, 'w') as f:
    json.dump(da_data, f)

print(f"\n✓ Saved to {DA_COMPARISON}")

