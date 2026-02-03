"""
Generate Live EWMA Ensemble Forecasts for USA Beef Tallow
=========================================================
Uses the optimized EWMA parameters to generate current forecasts
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("LIVE EWMA ENSEMBLE FORECASTS - USA Beef Tallow")
print("=" * 70)

# Load data
print("\n[1] Loading data...")
df = pd.read_parquet('data/21_USA_Beef_Tallow/all_children_data.parquet')
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
print(f"    Total records: {len(df):,}")
print(f"    Date range: {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}")

# Get latest date
latest_date = df['time'].max()
print(f"    Latest data: {latest_date.strftime('%Y-%m-%d')}")

# Optimal EWMA parameters (from grid search)
optimal_params = {
    30: {'lookback': 4, 'alpha': 0.2},
    60: {'lookback': 6, 'alpha': 0.3},
    90: {'lookback': 6, 'alpha': 0.5},
    180: {'lookback': 6, 'alpha': 0.5},
    270: {'lookback': 5, 'alpha': 0.5},
    360: {'lookback': 3, 'alpha': 0.2},
    450: {'lookback': 10, 'alpha': 0.5},
}

TOP_N = 5

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
    """Get top N models by historical DA"""
    model_performance = []
    for symbol in df_horizon['symbol'].unique():
        model_data = df_horizon[df_horizon['symbol'] == symbol]
        pred_dir = np.sign(model_data['close_predict'] - model_data['target_var_price'])
        actual_dir = np.sign(model_data['yn_actual'])
        mask = (pred_dir != 0) & (actual_dir != 0)
        if mask.sum() >= 50:
            correct = (pred_dir[mask] == actual_dir[mask]).sum()
            da = correct / mask.sum()
            model_performance.append({'symbol': symbol, 'da': da, 'n': mask.sum()})
    
    perf_df = pd.DataFrame(model_performance).sort_values('da', ascending=False)
    return perf_df.head(n)['symbol'].tolist(), perf_df.head(n)

def get_live_forecast(df_horizon, top_models, lookback, alpha):
    """Generate live EWMA ensemble forecast"""
    df_h = df_horizon[df_horizon['symbol'].isin(top_models)].copy()
    df_h = df_h[['time', 'symbol', 'close_predict', 'target_var_price', 'yn_actual']].copy()
    
    dates = sorted(df_h['time'].unique())
    
    # Get recent data for EWMA calculation
    recent_dates = dates[-(lookback + 1):]
    
    # Track recent correctness for each model
    model_recent_correct = {m: [] for m in top_models}
    
    for date in recent_dates[:-1]:  # Exclude latest (that's what we're forecasting)
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
    
    # Get latest day's predictions
    latest_date = dates[-1]
    latest_data = df_h[df_h['time'] == latest_date]
    
    preds = {}
    pred_prices = {}
    current_price = None
    
    for model in top_models:
        model_day = latest_data[latest_data['symbol'] == model]
        if len(model_day) == 0:
            continue
        
        close_pred = float(model_day['close_predict'].values[0])
        target_price = float(model_day['target_var_price'].values[0])
        
        pred_dir = np.sign(close_pred - target_price)
        if pred_dir != 0:
            preds[model] = pred_dir
            pred_prices[model] = close_pred
            current_price = target_price
    
    if len(preds) < 2:
        return None
    
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
    
    # Normalize weights
    total_weight = sum(model_weights.values()) + 0.001
    norm_weights = {m: w / total_weight for m, w in model_weights.items()}
    
    # Calculate ensemble prediction
    ewma_vote = sum(preds[m] * norm_weights[m] for m in preds.keys())
    ewma_pred_dir = np.sign(ewma_vote)
    
    # Calculate weighted average predicted price
    weighted_price = sum(pred_prices[m] * norm_weights[m] for m in preds.keys())
    
    # Calculate confidence (how strong is the consensus)
    confidence = abs(ewma_vote)
    
    return {
        'current_price': current_price,
        'predicted_price': weighted_price,
        'direction': 'UP' if ewma_pred_dir > 0 else 'DOWN' if ewma_pred_dir < 0 else 'NEUTRAL',
        'confidence': confidence,
        'pct_change': ((weighted_price - current_price) / current_price) * 100,
        'models_used': len(preds),
        'weights': norm_weights
    }

# Generate forecasts
print("\n[2] Generating EWMA Ensemble Forecasts...")
print("=" * 70)

forecasts = {}
horizons = sorted(optimal_params.keys())

for horizon in horizons:
    params = optimal_params[horizon]
    lookback = params['lookback']
    alpha = params['alpha']
    
    df_h = df[df['n_predict'] == horizon].copy()
    
    if len(df_h) == 0:
        continue
    
    top_models, top_df = get_top_models(df_h, TOP_N)
    
    if len(top_models) < 2:
        continue
    
    forecast = get_live_forecast(df_h, top_models, lookback, alpha)
    
    if forecast:
        forecasts[horizon] = forecast
        forecast['top_models'] = top_models

# Display results
print("\n" + "=" * 70)
print("📊 LIVE EWMA ENSEMBLE FORECASTS")
print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"   Based on data through: {latest_date.strftime('%Y-%m-%d')}")
print("=" * 70)

current_price = None
for h in sorted(forecasts.keys()):
    f = forecasts[h]
    current_price = f['current_price']
    
    direction_emoji = "🟢 UP" if f['direction'] == 'UP' else "🔴 DOWN" if f['direction'] == 'DOWN' else "⚪ NEUTRAL"
    conf_str = "HIGH" if f['confidence'] > 0.7 else "MEDIUM" if f['confidence'] > 0.4 else "LOW"
    
    target_date = latest_date + timedelta(days=h)
    
    print(f"\n  📅 {h}-DAY HORIZON (Target: {target_date.strftime('%Y-%m-%d')})")
    print(f"     Direction: {direction_emoji}")
    print(f"     Current Price:   ${f['current_price']:.2f}")
    print(f"     Predicted Price: ${f['predicted_price']:.2f} ({f['pct_change']:+.2f}%)")
    print(f"     Confidence: {conf_str} ({f['confidence']:.2f})")

print("\n" + "=" * 70)
print(f"📍 CURRENT PRICE: ${current_price:.2f}")
print("=" * 70)

# Summary table
print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│                    FORECAST SUMMARY TABLE                           │")
print("├──────────┬─────────────┬───────────────┬───────────┬───────────────┤")
print("│ Horizon  │  Direction  │ Predicted ($) │  Change   │   Target Date │")
print("├──────────┼─────────────┼───────────────┼───────────┼───────────────┤")

for h in sorted(forecasts.keys()):
    f = forecasts[h]
    dir_str = f"{'🟢 UP' if f['direction'] == 'UP' else '🔴 DOWN':^11}"
    target_date = latest_date + timedelta(days=h)
    print(f"│  {h:>4}d   │ {dir_str} │    ${f['predicted_price']:>7.2f}  │ {f['pct_change']:>+7.2f}% │  {target_date.strftime('%Y-%m-%d')} │")

print("└──────────┴─────────────┴───────────────┴───────────┴───────────────┘")

# Save to JSON
import json
output = {
    'generated_at': datetime.now().isoformat(),
    'data_through': latest_date.strftime('%Y-%m-%d'),
    'current_price': current_price,
    'forecasts': {}
}

for h, f in forecasts.items():
    output['forecasts'][str(h)] = {
        'horizon_days': h,
        'target_date': (latest_date + timedelta(days=h)).strftime('%Y-%m-%d'),
        'direction': f['direction'],
        'current_price': f['current_price'],
        'predicted_price': round(f['predicted_price'], 2),
        'pct_change': round(f['pct_change'], 2),
        'confidence': round(f['confidence'], 2)
    }

with open('data/21_USA_Beef_Tallow/live_ewma_forecast.json', 'w') as fp:
    json.dump(output, fp, indent=2)

print(f"\n✅ Forecast saved to: data/21_USA_Beef_Tallow/live_ewma_forecast.json")

