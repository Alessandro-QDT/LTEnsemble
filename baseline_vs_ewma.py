"""
BASELINE (Raw Dataset Forecasts) vs EWMA Ensemble
=================================================
Baseline = The forecasts as they exist in the dataset
EWMA = Our optimized ensemble approach
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("BASELINE (Dataset Forecasts) vs EWMA ENSEMBLE - FULL COMPARISON")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_parquet('data/21_USA_Beef_Tallow/all_children_data.parquet')
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
print(f"    Total records: {len(df):,}")

# Constants
TRAIN_END = pd.Timestamp('2024-12-31')
TEST_START = pd.Timestamp('2025-01-01')
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

def calculate_metrics(results_df, name="Strategy"):
    """Calculate comprehensive financial metrics"""
    if len(results_df) == 0:
        return None
    
    metrics = {'name': name}
    n = len(results_df)
    wins = results_df[results_df['trade_return'] > 0]
    losses = results_df[results_df['trade_return'] < 0]
    
    metrics['n_signals'] = n
    metrics['n_wins'] = len(wins)
    metrics['n_losses'] = len(losses)
    metrics['da'] = results_df['correct'].mean() * 100
    metrics['win_rate'] = len(wins) / n * 100 if n > 0 else 0
    metrics['total_return'] = results_df['trade_return'].sum()
    metrics['avg_return'] = results_df['trade_return'].mean()
    metrics['avg_win'] = wins['trade_return'].mean() if len(wins) > 0 else 0
    metrics['avg_loss'] = losses['trade_return'].mean() if len(losses) > 0 else 0
    
    gross_profit = wins['trade_return'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['trade_return'].sum()) if len(losses) > 0 else 0.001
    metrics['profit_factor'] = gross_profit / gross_loss
    metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
    
    results_df = results_df.copy()
    results_df['cum_return'] = results_df['trade_return'].cumsum()
    peak = results_df['cum_return'].cummax()
    drawdown = results_df['cum_return'] - peak
    metrics['max_drawdown'] = drawdown.min()
    
    if results_df['trade_return'].std() > 0:
        daily_sharpe = results_df['trade_return'].mean() / results_df['trade_return'].std()
        metrics['sharpe'] = daily_sharpe * np.sqrt(252)
    else:
        metrics['sharpe'] = 0
    
    return metrics

def calculate_baseline_metrics(df_horizon):
    """Calculate metrics for ALL forecasts in the dataset as-is (the baseline)"""
    # Filter to test period
    test_data = df_horizon[df_horizon['time'] >= TEST_START].copy()
    
    # Calculate for each forecast
    test_data['pred_dir'] = np.sign(test_data['close_predict'] - test_data['target_var_price'])
    test_data['actual_dir'] = np.sign(test_data['yn_actual'])
    
    # Filter valid signals
    valid = test_data[(test_data['pred_dir'] != 0) & (test_data['actual_dir'] != 0)].copy()
    
    if len(valid) == 0:
        return None
    
    valid['correct'] = (valid['pred_dir'] == valid['actual_dir']).astype(int)
    valid['trade_return'] = valid['pred_dir'] * valid['yn_actual']
    
    # Aggregate by date (average across all models for that day)
    daily = valid.groupby('time').agg({
        'correct': 'mean',  # Average correctness across models
        'trade_return': 'mean',  # Average return across models
        'pred_dir': 'first',
        'actual_dir': 'first'
    }).reset_index()
    
    return calculate_metrics(daily, "Baseline (All Forecasts)")

def run_ewma_ensemble(df_horizon, top_models, lookback, alpha):
    """Run EWMA ensemble"""
    df_h = df_horizon[df_horizon['symbol'].isin(top_models)].copy()
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
        
        results.append({
            'date': date, 'pred': ewma_pred, 'actual': actual_dir,
            'correct': ewma_correct, 'actual_return': actual_return,
            'trade_return': ewma_pred * actual_return if actual_return else 0
        })
        
        for model in preds.keys():
            was_correct = 1 if preds[model] == actual_dir else 0
            model_recent_correct[model].append(was_correct)
    
    return pd.DataFrame(results)

# ============= MAIN COMPARISON =============
print("\n[2] Comparing BASELINE vs EWMA for each horizon...")
print("=" * 80)

all_results = []

for horizon in sorted(optimal_params.keys()):
    params = optimal_params[horizon]
    lookback = params['lookback']
    alpha = params['alpha']
    
    print(f"\n{'='*80}")
    print(f"  HORIZON: {horizon} DAYS (L={lookback}, α={alpha})")
    print(f"{'='*80}")
    
    df_h = df[df['n_predict'] == horizon].copy()
    
    # Calculate BASELINE (raw forecasts from dataset)
    baseline_metrics = calculate_baseline_metrics(df_h)
    
    # Calculate EWMA ensemble
    top_models = get_top_models(df_h, TOP_N)
    ewma_results = run_ewma_ensemble(df_h, top_models, lookback, alpha)
    ewma_test = ewma_results[ewma_results['date'] >= TEST_START]
    ewma_metrics = calculate_metrics(ewma_test, "EWMA Ensemble")
    
    if not baseline_metrics or not ewma_metrics:
        print("  Insufficient data, skipping...")
        continue
    
    # Print comparison
    print(f"\n  {'Metric':<25} {'BASELINE':>18} {'EWMA':>18} {'Improvement':>15}")
    print(f"  {'-'*78}")
    
    metrics_to_compare = [
        ('Signals', 'n_signals', '{:.0f}', ''),
        ('Directional Acc', 'da', '{:.1f}', '%'),
        ('Win Rate', 'win_rate', '{:.1f}', '%'),
        ('Total Return', 'total_return', '{:.2f}', '%'),
        ('Avg Return/Trade', 'avg_return', '{:.4f}', '%'),
        ('Profit Factor', 'profit_factor', '{:.2f}', ''),
        ('Max Drawdown', 'max_drawdown', '{:.2f}', '%'),
        ('Sharpe Ratio', 'sharpe', '{:.2f}', ''),
    ]
    
    for label, key, fmt, suffix in metrics_to_compare:
        b_val = baseline_metrics[key]
        e_val = ewma_metrics[key]
        diff = e_val - b_val
        
        b_str = fmt.format(b_val) + suffix
        e_str = fmt.format(e_val) + suffix
        
        if key in ['max_drawdown']:  # Lower is better
            imp_str = f"{diff:+.2f}{suffix}" if diff < 0 else f"{diff:+.2f}{suffix}"
        else:  # Higher is better
            imp_str = f"{diff:+.2f}{suffix}" if diff != 0 else "0"
        
        print(f"  {label:<25} {b_str:>18} {e_str:>18} {imp_str:>15}")
    
    print(f"  {'-'*78}")
    
    # Store results
    all_results.append({
        'horizon': horizon,
        'baseline_da': baseline_metrics['da'],
        'ewma_da': ewma_metrics['da'],
        'da_improvement': ewma_metrics['da'] - baseline_metrics['da'],
        'baseline_return': baseline_metrics['total_return'],
        'ewma_return': ewma_metrics['total_return'],
        'return_improvement': ewma_metrics['total_return'] - baseline_metrics['total_return'],
        'baseline_pf': baseline_metrics['profit_factor'],
        'ewma_pf': ewma_metrics['profit_factor'],
        'baseline_sharpe': baseline_metrics['sharpe'],
        'ewma_sharpe': ewma_metrics['sharpe'],
    })

# ============= SUMMARY =============
print("\n")
print("=" * 80)
print("SUMMARY: BASELINE vs EWMA ENSEMBLE (Test Period 2025+)")
print("=" * 80)

summary_df = pd.DataFrame(all_results)

print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
print("│                    DIRECTIONAL ACCURACY COMPARISON                          │")
print("├──────────┬─────────────────┬─────────────────┬─────────────────────────────┤")
print("│ Horizon  │     BASELINE    │      EWMA       │        IMPROVEMENT          │")
print("├──────────┼─────────────────┼─────────────────┼─────────────────────────────┤")
for _, row in summary_df.iterrows():
    print(f"│  {row['horizon']:>4}d   │      {row['baseline_da']:>6.1f}%     │      {row['ewma_da']:>6.1f}%     │         +{row['da_improvement']:>5.1f} pp            │")
print("└──────────┴─────────────────┴─────────────────┴─────────────────────────────┘")

print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
print("│                     CUMULATIVE RETURN COMPARISON                            │")
print("├──────────┬─────────────────┬─────────────────┬─────────────────────────────┤")
print("│ Horizon  │     BASELINE    │      EWMA       │        IMPROVEMENT          │")
print("├──────────┼─────────────────┼─────────────────┼─────────────────────────────┤")
for _, row in summary_df.iterrows():
    b_sign = "+" if row['baseline_return'] >= 0 else ""
    e_sign = "+" if row['ewma_return'] >= 0 else ""
    i_sign = "+" if row['return_improvement'] >= 0 else ""
    print(f"│  {row['horizon']:>4}d   │     {b_sign}{row['baseline_return']:>6.2f}%     │     {e_sign}{row['ewma_return']:>6.2f}%     │        {i_sign}{row['return_improvement']:>6.2f}%             │")
print("└──────────┴─────────────────┴─────────────────┴─────────────────────────────┘")

# Average improvements
avg_da_imp = summary_df['da_improvement'].mean()
avg_ret_imp = summary_df['return_improvement'].mean()

print(f"\n  📊 AVERAGE IMPROVEMENTS ACROSS ALL HORIZONS:")
print(f"     • Directional Accuracy: +{avg_da_imp:.1f} percentage points")
print(f"     • Cumulative Return: +{avg_ret_imp:.2f}%")

print("\n" + "=" * 80)
summary_df.to_csv('data/21_USA_Beef_Tallow/baseline_vs_ewma.csv', index=False)
print("Results saved to: data/21_USA_Beef_Tallow/baseline_vs_ewma.csv")
print("=" * 80)

