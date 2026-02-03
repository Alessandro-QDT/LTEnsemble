"""
Generate CORRECT equity curves matching the methodology in baseline_vs_ewma.py
"""
import pandas as pd
import numpy as np
import json

print("=" * 70)
print("GENERATING CORRECT EQUITY CURVES")
print("=" * 70)

# Load data
df = pd.read_parquet('data/21_USA_Beef_Tallow/all_children_data.parquet')
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

# Constants
TRAIN_END = pd.Timestamp('2024-12-31')
TEST_START = pd.Timestamp('2025-01-01')
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

def calculate_equity_curves(df_horizon, horizon, top_models, lookback, alpha):
    """Calculate proper equity curves for baseline and EWMA"""
    
    # =========== BASELINE EQUITY ===========
    # Use all forecasts in the dataset
    test_data = df_horizon[df_horizon['time'] >= TEST_START].copy()
    test_data['pred_dir'] = np.sign(test_data['close_predict'] - test_data['target_var_price'])
    test_data['actual_dir'] = np.sign(test_data['yn_actual'])
    valid = test_data[(test_data['pred_dir'] != 0) & (test_data['actual_dir'] != 0)].copy()
    
    # Aggregate by date (average across all models for that day)
    daily_baseline = valid.groupby('time').agg({
        'yn_actual': 'mean',  # Average actual return
        'pred_dir': lambda x: np.sign(x.mean())  # Average direction prediction
    }).reset_index()
    daily_baseline['trade_return'] = daily_baseline['pred_dir'] * daily_baseline['yn_actual']
    daily_baseline = daily_baseline.sort_values('time')
    
    baseline_equity = [100]
    baseline_dates = []
    for _, row in daily_baseline.iterrows():
        baseline_equity.append(baseline_equity[-1] * (1 + row['trade_return'] / 100))
        baseline_dates.append(row['time'].strftime('%Y-%m-%d'))
    baseline_equity = baseline_equity[1:]  # Remove initial 100
    
    # =========== EWMA EQUITY ===========
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
        actual_return = None
        
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
                actual_return = yn_actual
        
        if len(preds) < 2 or actual_dir is None:
            continue
        
        # Calculate EWMA weights
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
        
        # EWMA weighted direction
        ewma_vote = sum(preds[m] * norm_weights[m] for m in preds.keys())
        ensemble_pred = np.sign(ewma_vote)
        
        if date >= TEST_START:
            ewma_results.append({
                'date': date,
                'pred_dir': ensemble_pred,
                'actual_dir': actual_dir,
                'actual_return': actual_return,
                'trade_return': ensemble_pred * actual_return
            })
        
        # Update tracking
        for model in preds.keys():
            was_correct = 1 if preds[model] == actual_dir else 0
            model_recent_correct[model].append(was_correct)
    
    ewma_df = pd.DataFrame(ewma_results).sort_values('date')
    ewma_equity = [100]
    ewma_dates = []
    for _, row in ewma_df.iterrows():
        ewma_equity.append(ewma_equity[-1] * (1 + row['trade_return'] / 100))
        ewma_dates.append(row['date'].strftime('%Y-%m-%d'))
    ewma_equity = ewma_equity[1:]
    
    return {
        'baseline': {'dates': baseline_dates, 'equity': [round(e, 2) for e in baseline_equity]},
        'ewma': {'dates': ewma_dates, 'equity': [round(e, 2) for e in ewma_equity]}
    }

# Generate equity data for all horizons
equity_data = {}

for horizon, params in optimal_params.items():
    print(f"\nProcessing {horizon}d horizon...")
    df_h = df[df['n_predict'] == horizon].copy()
    top_models = get_top_models(df_h, TOP_N)
    
    curves = calculate_equity_curves(df_h, horizon, top_models, params['lookback'], params['alpha'])
    
    h_key = f"{horizon}d"
    equity_data[h_key] = curves
    
    base_final = curves['baseline']['equity'][-1] if curves['baseline']['equity'] else 100
    ewma_final = curves['ewma']['equity'][-1] if curves['ewma']['equity'] else 100
    
    print(f"  Baseline final: ${base_final:.2f} ({base_final - 100:+.1f}%)")
    print(f"  EWMA final:     ${ewma_final:.2f} ({ewma_final - 100:+.1f}%)")
    print(f"  Winner: {'EWMA' if ewma_final > base_final else 'BASELINE'}")

# Save to JSON
output_path = 'data/21_USA_Beef_Tallow/equity_curves.json'
with open(output_path, 'w') as f:
    json.dump(equity_data, f)

print(f"\n✓ Saved to {output_path}")

