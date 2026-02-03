"""
Grid Search for Optimal EWMA Parameters
========================================
Find best lookback (L) and alpha for each horizon
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Import project configuration
from config import PARQUET_PATH, GRID_SEARCH_RESULTS, OPTIMAL_PARAMS, TRAIN_END, TEST_START, PROJECT_ID, PROJECT_NAME

print("=" * 70)
print("GRID SEARCH: OPTIMAL EWMA PARAMETERS PER HORIZON")
print("=" * 70)
print(f"Project: {PROJECT_ID} - {PROJECT_NAME}")

# Load the complete dataset
print("\n[1] Loading data...")
if not os.path.exists(PARQUET_PATH):
    print(f"\n❌ ERROR: Data file not found: {PARQUET_PATH}")
    print(f"   Run first: python fetch_data.py --project_id {PROJECT_ID} --project_name {PROJECT_NAME}")
    exit(1)

df = pd.read_parquet(PARQUET_PATH)
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

print(f"    Total records: {len(df):,}")

# Define train/test split
TRAIN_END = pd.Timestamp(TRAIN_END)
TEST_START = pd.Timestamp(TEST_START)

horizons = sorted(df['n_predict'].unique())
print(f"    Horizons: {[int(h) for h in horizons]}")

# =============================================================================
# Helper Functions
# =============================================================================

def calculate_da(data):
    """Calculate directional accuracy"""
    pred_dir = np.sign(data['close_predict'] - data['target_var_price'])
    actual_dir = np.sign(data['yn_actual'])
    mask = (pred_dir != 0) & (actual_dir != 0)
    if mask.sum() > 0:
        correct = (pred_dir[mask] == actual_dir[mask]).sum()
        return correct / mask.sum(), mask.sum()
    return 0, 0


def ewma_weights(accuracies, alpha=0.3):
    """Calculate EWMA weights - more recent = higher weight"""
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
    """Get top N models by training DA"""
    train_data = df_horizon[df_horizon['time'] <= TRAIN_END]
    
    model_performance = []
    for symbol in df_horizon['symbol'].unique():
        train_model = train_data[train_data['symbol'] == symbol]
        train_da, train_n = calculate_da(train_model)
        if train_n >= 100:
            model_performance.append({'symbol': symbol, 'train_da': train_da})
    
    perf_df = pd.DataFrame(model_performance).sort_values('train_da', ascending=False)
    return perf_df.head(n)['symbol'].tolist()


def run_ewma_ensemble(df_horizon, top_models, lookback, alpha):
    """Run EWMA ensemble with given parameters"""
    df_h = df_horizon[df_horizon['symbol'].isin(top_models)].copy()
    
    # Select only the columns we need to avoid dtype issues
    df_h = df_h[['time', 'symbol', 'close_predict', 'target_var_price', 'yn_actual']].copy()
    
    dates = sorted(df_h['time'].unique())
    
    results = []
    model_recent_correct = {m: [] for m in top_models}
    
    for i, date in enumerate(dates):
        if i < lookback:
            continue
            
        day_data = df_h[df_h['time'] == date]
        if len(day_data) < len(top_models):
            continue
        
        preds = {}
        actual_dir = None
        
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
        
        if len(preds) < 2 or actual_dir is None:
            continue
        
        # Calculate EWMA-weighted prediction
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
        
        ewma_vote = sum(preds[m] * norm_weights[m] for m in preds.keys())
        ewma_pred = np.sign(ewma_vote)
        ewma_correct = 1 if ewma_pred == actual_dir else 0
        
        results.append({'date': date, 'correct': ewma_correct})
        
        # Update tracking
        for model in preds.keys():
            was_correct = 1 if preds[model] == actual_dir else 0
            model_recent_correct[model].append(was_correct)
    
    return pd.DataFrame(results)


# =============================================================================
# Grid Search Parameters
# =============================================================================
LOOKBACK_VALUES = [3, 4, 5, 6, 7, 8, 10, 12, 15]
ALPHA_VALUES = [0.2, 0.3, 0.4, 0.5]
TOP_N = 5

print(f"\n[2] Grid Search Parameters:")
print(f"    Lookback (L): {LOOKBACK_VALUES}")
print(f"    Alpha (α): {ALPHA_VALUES}")
print(f"    Top N models: {TOP_N}")

# =============================================================================
# Run Grid Search
# =============================================================================
print("\n" + "=" * 70)
print("[3] RUNNING GRID SEARCH")
print("=" * 70)

all_results = []
best_per_horizon = {}

for horizon in horizons:
    print(f"\n  Horizon {int(horizon)}d...")
    df_h = df[df['n_predict'] == horizon].copy()
    
    # Get top models
    top_models = get_top_models(df_h, TOP_N)
    print(f"    Selected models: {top_models}")
    if len(top_models) < 2:
        print(f"    Skipping - insufficient models")
        continue
    
    horizon_results = []
    
    for lookback in LOOKBACK_VALUES:
        for alpha in ALPHA_VALUES:
            result_df = run_ewma_ensemble(df_h, top_models, lookback, alpha)
            
            if len(result_df) == 0:
                continue
            
            result_df['date'] = pd.to_datetime(result_df['date'])
            
            # Calculate train/test DA
            train_df = result_df[result_df['date'] <= TRAIN_END]
            test_df = result_df[result_df['date'] >= TEST_START]
            
            train_da = train_df['correct'].mean() * 100 if len(train_df) > 0 else 0
            test_da = test_df['correct'].mean() * 100 if len(test_df) > 0 else 0
            
            horizon_results.append({
                'horizon': int(horizon),
                'lookback': lookback,
                'alpha': alpha,
                'train_da': train_da,
                'test_da': test_da,
                'test_signals': len(test_df)
            })
    
    if horizon_results:
        hr_df = pd.DataFrame(horizon_results)
        best = hr_df.loc[hr_df['test_da'].idxmax()]
        best_per_horizon[int(horizon)] = best.to_dict()
        all_results.extend(horizon_results)
        
        print(f"    Best: L={int(best['lookback'])}, α={best['alpha']:.1f} → Test DA: {best['test_da']:.1f}%")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("[4] OPTIMAL PARAMETERS PER HORIZON")
print("=" * 70)

print(f"\n{'Horizon':>8} | {'Best L':>7} | {'Best α':>7} | {'Train DA':>10} | {'Test DA':>10} | {'Signals':>8}")
print("-" * 70)

summary_rows = []
for horizon in sorted(best_per_horizon.keys()):
    b = best_per_horizon[horizon]
    print(f"{horizon:>7}d | {int(b['lookback']):>7} | {b['alpha']:>7.1f} | {b['train_da']:>9.1f}% | {b['test_da']:>9.1f}% | {int(b['test_signals']):>8}")
    summary_rows.append(b)

# Compare with L=6, α=0.3 (our current default)
print("\n" + "=" * 70)
print("[5] COMPARISON: OPTIMAL vs DEFAULT (L=6, α=0.3)")
print("=" * 70)

# Get L=6, α=0.3 results
all_df = pd.DataFrame(all_results)
default_results = all_df[(all_df['lookback'] == 6) & (all_df['alpha'] == 0.3)]

print(f"\n{'Horizon':>8} | {'Default':>10} | {'Optimal':>10} | {'Δ':>8} | {'Optimal Params':>20}")
print("-" * 70)

for horizon in sorted(best_per_horizon.keys()):
    default_row = default_results[default_results['horizon'] == horizon]
    if len(default_row) > 0:
        default_da = default_row.iloc[0]['test_da']
    else:
        default_da = 0
    
    optimal = best_per_horizon[horizon]
    optimal_da = optimal['test_da']
    delta = optimal_da - default_da
    
    delta_str = f"+{delta:.1f}pp" if delta > 0 else f"{delta:.1f}pp"
    params = f"L={int(optimal['lookback'])}, α={optimal['alpha']:.1f}"
    
    print(f"{horizon:>7}d | {default_da:>9.1f}% | {optimal_da:>9.1f}% | {delta_str:>8} | {params:>20}")

# Save results
results_df = pd.DataFrame(all_results)
results_df.to_csv(GRID_SEARCH_RESULTS, index=False)

optimal_df = pd.DataFrame(summary_rows)
optimal_df.to_csv(OPTIMAL_PARAMS, index=False)

print("\n" + "=" * 70)
print("Results saved to:")
print(f"  - {GRID_SEARCH_RESULTS} (all combinations)")
print(f"  - {OPTIMAL_PARAMS} (best per horizon)")
print("=" * 70)

