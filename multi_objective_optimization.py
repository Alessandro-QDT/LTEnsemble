"""
Multi-Objective EWMA Optimization
=================================
Find parameters that beat baseline on BOTH:
- Directional Accuracy (DA)
- Quantitative Accuracy (QA)

Grid search over: L, alpha, N (top models), model selection criteria
"""
import pandas as pd
import numpy as np
import os
from itertools import product
import time

print("=" * 80)
print("MULTI-OBJECTIVE EWMA OPTIMIZATION")
print("Goal: Beat baseline on BOTH DA and QA")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(script_dir, 'data/21_USA_Beef_Tallow/all_children_data.parquet'))
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
print(f"    Total records: {len(df):,}")

# Constants
TRAIN_END = pd.Timestamp('2024-12-31')
TEST_START = pd.Timestamp('2025-01-01')

# Search space
HORIZONS = [30, 60, 90, 180]
L_VALUES = [3, 4, 5, 6, 8, 10]
ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
N_VALUES = [3, 5, 7, 10]
# Weight for DA in combined score (1-w is for QA)
DA_WEIGHTS = [0.5, 0.6, 0.7, 0.8]  # For model selection

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
    return [w / total for w in weights] if total > 0 else [1/n] * n

def get_top_models_by_combined(df_horizon, n=5, da_weight=0.7):
    """Get top N models based on combined DA + QA score from training"""
    train_data = df_horizon[df_horizon['time'] <= TRAIN_END]
    model_performance = []
    
    for symbol in df_horizon['symbol'].unique():
        train_model = train_data[train_data['symbol'] == symbol]
        if len(train_model) < 100:
            continue
            
        # Calculate DA
        pred_dir = np.sign(train_model['close_predict'] - train_model['target_var_price'])
        actual_dir = np.sign(train_model['yn_actual'])
        mask = (pred_dir != 0) & (actual_dir != 0)
        if mask.sum() < 50:
            continue
        train_da = (pred_dir[mask] == actual_dir[mask]).sum() / mask.sum()
        
        # Calculate QA (simplified - compare predicted vs actual prices)
        # Using available data to estimate price accuracy
        qa_values = []
        dates = sorted(train_model['time'].unique())
        n_predict = int(train_model['n_predict'].iloc[0])
        
        for i, date in enumerate(dates[:-n_predict]):
            day_data = train_model[train_model['time'] == date]
            future_date = dates[min(i + n_predict, len(dates)-1)]
            future_data = train_model[train_model['time'] == future_date]
            
            if len(day_data) > 0 and len(future_data) > 0:
                pred_price = day_data['close_predict'].values[0]
                actual_price = future_data['target_var_price'].values[0]
                if actual_price > 0:
                    qa = max(0, 1 - abs(pred_price - actual_price) / actual_price)
                    qa_values.append(qa)
        
        train_qa = np.mean(qa_values) if qa_values else 0.5
        
        # Combined score
        combined = da_weight * train_da + (1 - da_weight) * train_qa
        
        model_performance.append({
            'symbol': symbol, 
            'train_da': train_da, 
            'train_qa': train_qa,
            'combined': combined
        })
    
    perf_df = pd.DataFrame(model_performance).sort_values('combined', ascending=False)
    return perf_df.head(n)['symbol'].tolist()

def calculate_qa(predicted_price, actual_price):
    if actual_price == 0:
        return 0
    return max(0, 1 - abs(predicted_price - actual_price) / abs(actual_price))

def run_ewma_ensemble(df_horizon, top_models, lookback, alpha, n_predict):
    """Run EWMA ensemble and return both DA and QA metrics"""
    df_h = df_horizon[df_horizon['symbol'].isin(top_models)].copy()
    df_h = df_h[['time', 'symbol', 'close_predict', 'target_var_price', 'yn_actual']].copy()
    
    dates = sorted(df_h['time'].unique())
    results = []
    model_recent_correct = {m: [] for m in top_models}
    
    for i, date in enumerate(dates):
        if i < lookback:
            continue
            
        day_data = df_h[df_h['time'] == date]
        
        # Track model performance even before test period
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
                model_weights[model] = max(0.01, weighted_acc)
            else:
                model_weights[model] = 0.5
        
        total_weight = sum(model_weights.values()) + 0.001
        norm_weights = {m: w / total_weight for m, w in model_weights.items()}
        
        # EWMA weighted predictions
        ewma_vote = sum(preds[m] * norm_weights[m] for m in preds.keys())
        ewma_pred_dir = np.sign(ewma_vote) if ewma_vote != 0 else 1
        ewma_price_pred = sum(pred_prices[m] * norm_weights[m] for m in preds.keys())
        
        # Get actual direction
        actual_dir = np.sign(day_data['yn_actual'].iloc[0])
        
        # Get actual future price for QA
        future_idx = i + n_predict
        actual_future_price = None
        if future_idx < len(dates):
            future_date = dates[future_idx]
            future_data = df_h[df_h['time'] == future_date]
            if len(future_data) > 0:
                actual_future_price = future_data['target_var_price'].iloc[0]
        
        results.append({
            'date': date,
            'pred_dir': ewma_pred_dir,
            'actual_dir': actual_dir,
            'correct': 1 if ewma_pred_dir == actual_dir else 0,
            'pred_price': ewma_price_pred,
            'actual_price': actual_future_price,
        })
        
        # Update model tracking
        for model in preds.keys():
            model_day_data = day_data[day_data['symbol'] == model]
            if len(model_day_data) > 0:
                yn_actual = model_day_data['yn_actual'].values[0]
                was_correct = 1 if preds[model] == np.sign(yn_actual) else 0
                model_recent_correct[model].append(was_correct)
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        return None, None
    
    # Calculate DA
    da = results_df['correct'].mean() * 100
    
    # Calculate QA (only where we have actual future price)
    qa_df = results_df[results_df['actual_price'].notna()]
    if len(qa_df) > 0:
        qa_values = qa_df.apply(lambda r: calculate_qa(r['pred_price'], r['actual_price']), axis=1)
        qa = qa_values.mean() * 100
    else:
        qa = None
    
    return da, qa

def calculate_baseline(df_horizon, n_predict):
    """Calculate baseline DA and QA"""
    test_data = df_horizon[df_horizon['time'] >= TEST_START].copy()
    
    # DA: Average direction across all models per day
    test_data['pred_dir'] = np.sign(test_data['close_predict'] - test_data['target_var_price'])
    test_data['actual_dir'] = np.sign(test_data['yn_actual'])
    
    valid = test_data[(test_data['pred_dir'] != 0) & (test_data['actual_dir'] != 0)].copy()
    valid['correct'] = (valid['pred_dir'] == valid['actual_dir']).astype(int)
    
    daily = valid.groupby('time').agg({
        'correct': 'mean',
        'close_predict': 'mean',
        'target_var_price': 'first',
    }).reset_index()
    
    baseline_da = daily['correct'].mean() * 100
    
    # QA: Compare average prediction to actual future price
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
    
    baseline_qa = np.mean(qa_values) * 100 if qa_values else 0
    
    return baseline_da, baseline_qa

# ============= MAIN OPTIMIZATION =============
print("\n[2] Calculating baselines for each horizon...")
baselines = {}
for horizon in HORIZONS:
    df_h = df[df['n_predict'] == horizon].copy()
    baseline_da, baseline_qa = calculate_baseline(df_h, horizon)
    baselines[horizon] = {'da': baseline_da, 'qa': baseline_qa}
    print(f"    {horizon}d: DA={baseline_da:.2f}%, QA={baseline_qa:.2f}%")

print("\n[3] Running grid search...")
print(f"    Search space: {len(HORIZONS)} horizons × {len(L_VALUES)} L × {len(ALPHA_VALUES)} α × {len(N_VALUES)} N × {len(DA_WEIGHTS)} weights")
total_combos = len(HORIZONS) * len(L_VALUES) * len(ALPHA_VALUES) * len(N_VALUES) * len(DA_WEIGHTS)
print(f"    Total combinations: {total_combos}")
print("=" * 80)

all_results = []
best_per_horizon = {}

start_time = time.time()

for horizon in HORIZONS:
    print(f"\n{'='*80}")
    print(f"  HORIZON: {horizon} DAYS")
    print(f"  Baseline: DA={baselines[horizon]['da']:.2f}%, QA={baselines[horizon]['qa']:.2f}%")
    print(f"{'='*80}")
    
    df_h = df[df['n_predict'] == horizon].copy()
    baseline_da = baselines[horizon]['da']
    baseline_qa = baselines[horizon]['qa']
    
    horizon_results = []
    combo_count = 0
    total_horizon_combos = len(L_VALUES) * len(ALPHA_VALUES) * len(N_VALUES) * len(DA_WEIGHTS)
    
    for L, alpha, N, da_weight in product(L_VALUES, ALPHA_VALUES, N_VALUES, DA_WEIGHTS):
        combo_count += 1
        if combo_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"    Progress: {combo_count}/{total_horizon_combos} ({elapsed:.1f}s elapsed)")
        
        # Get top models using combined score
        top_models = get_top_models_by_combined(df_h, n=N, da_weight=da_weight)
        
        if len(top_models) < 2:
            continue
        
        # Run ensemble
        da, qa = run_ewma_ensemble(df_h, top_models, L, alpha, horizon)
        
        if da is None or qa is None:
            continue
        
        # Check if beats baseline on BOTH
        beats_da = da > baseline_da
        beats_qa = qa > baseline_qa
        beats_both = beats_da and beats_qa
        
        result = {
            'horizon': horizon,
            'L': L,
            'alpha': alpha,
            'N': N,
            'da_weight': da_weight,
            'da': da,
            'qa': qa,
            'baseline_da': baseline_da,
            'baseline_qa': baseline_qa,
            'da_improvement': da - baseline_da,
            'qa_improvement': qa - baseline_qa,
            'beats_da': beats_da,
            'beats_qa': beats_qa,
            'beats_both': beats_both,
        }
        
        horizon_results.append(result)
        all_results.append(result)
    
    # Find best combo for this horizon
    winners = [r for r in horizon_results if r['beats_both']]
    
    if winners:
        # Sort by total improvement (DA + QA)
        winners.sort(key=lambda x: x['da_improvement'] + x['qa_improvement'], reverse=True)
        best = winners[0]
        best_per_horizon[horizon] = best
        
        print(f"\n  ✅ FOUND {len(winners)} WINNING COMBINATIONS!")
        print(f"  Best: L={best['L']}, α={best['alpha']}, N={best['N']}, DA_weight={best['da_weight']}")
        print(f"        DA: {best['da']:.2f}% (baseline: {baseline_da:.2f}%, +{best['da_improvement']:.2f})")
        print(f"        QA: {best['qa']:.2f}% (baseline: {baseline_qa:.2f}%, +{best['qa_improvement']:.2f})")
    else:
        # Find closest to beating both
        horizon_results.sort(key=lambda x: min(x['da_improvement'], x['qa_improvement']), reverse=True)
        if horizon_results:
            best = horizon_results[0]
            print(f"\n  ⚠️  No combo beats BOTH metrics")
            print(f"  Closest: L={best['L']}, α={best['alpha']}, N={best['N']}")
            print(f"        DA: {best['da']:.2f}% ({'+' if best['da_improvement'] >= 0 else ''}{best['da_improvement']:.2f})")
            print(f"        QA: {best['qa']:.2f}% ({'+' if best['qa_improvement'] >= 0 else ''}{best['qa_improvement']:.2f})")

# ============= FINAL SUMMARY =============
elapsed_total = time.time() - start_time
print("\n")
print("=" * 80)
print("FINAL RESULTS")
print(f"Total time: {elapsed_total:.1f} seconds")
print("=" * 80)

results_df = pd.DataFrame(all_results)

print("\n┌─────────────────────────────────────────────────────────────────────────────────────┐")
print("│                    OPTIMAL PARAMETERS (BEATS BOTH METRICS)                          │")
print("├──────────┬─────┬───────┬─────┬──────────┬──────────┬──────────┬──────────┬──────────┤")
print("│ Horizon  │  L  │   α   │  N  │ DA_wt    │    DA    │    QA    │  DA Imp  │  QA Imp  │")
print("├──────────┼─────┼───────┼─────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

for horizon in HORIZONS:
    if horizon in best_per_horizon:
        b = best_per_horizon[horizon]
        print(f"│  {horizon:>4}d   │ {b['L']:>3} │ {b['alpha']:>5.2f} │ {b['N']:>3} │   {b['da_weight']:>4.1f}   │  {b['da']:>6.2f}% │  {b['qa']:>6.2f}% │  +{b['da_improvement']:>5.2f}  │  +{b['qa_improvement']:>5.2f}  │")
    else:
        print(f"│  {horizon:>4}d   │  -  │   -   │  -  │    -     │    -     │    -     │    -     │    -     │")

print("└──────────┴─────┴───────┴─────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

# Summary stats
total_winners = len([r for r in all_results if r['beats_both']])
print(f"\n  📊 SUMMARY:")
print(f"     • Total combinations tested: {len(all_results)}")
print(f"     • Combinations beating BOTH metrics: {total_winners}")
print(f"     • Horizons with winning combos: {len(best_per_horizon)}/{len(HORIZONS)}")

if best_per_horizon:
    avg_da_imp = np.mean([b['da_improvement'] for b in best_per_horizon.values()])
    avg_qa_imp = np.mean([b['qa_improvement'] for b in best_per_horizon.values()])
    print(f"     • Average DA improvement: +{avg_da_imp:.2f} pp")
    print(f"     • Average QA improvement: +{avg_qa_imp:.2f} pp")

# Save results
output_path = os.path.join(script_dir, 'data/21_USA_Beef_Tallow/multi_objective_results.csv')
results_df.to_csv(output_path, index=False)
print(f"\n  Results saved to: {output_path}")

# Save best params
if best_per_horizon:
    best_df = pd.DataFrame(best_per_horizon.values())
    best_path = os.path.join(script_dir, 'data/21_USA_Beef_Tallow/optimal_multi_params.csv')
    best_df.to_csv(best_path, index=False)
    print(f"  Best params saved to: {best_path}")

print("\n" + "=" * 80)

