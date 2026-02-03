"""
Generate Snake Chart for EWMA Ensemble
Using the same drift matrix logic as the multi-asset dashboard
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

print("=" * 70)
print("GENERATING EWMA ENSEMBLE SNAKE CHART")
print("=" * 70)

# Load data
df = pd.read_parquet('data/21_USA_Beef_Tallow/all_children_data.parquet')
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

TRAIN_END = pd.Timestamp('2024-12-31')
TEST_START = pd.Timestamp('2025-01-01')
TOP_N = 5

# Horizons we have
HORIZONS = [30, 60, 90, 180]

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

def generate_ewma_forecasts(df_horizon, horizon, top_models, lookback, alpha):
    """Generate EWMA ensemble forecast prices for each date"""
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
        prices = {}
        current_price = None
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
                prices[model] = close_pred
                current_price = target_price
                actual_dir = act_dir
        
        if len(preds) < 2 or current_price is None:
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
        
        # EWMA weighted forecast price
        ewma_forecast = sum(prices[m] * norm_weights[m] for m in prices.keys())
        
        # Baseline forecast (simple average)
        baseline_forecast = np.mean(list(prices.values()))
        
        results.append({
            'date': date,
            'current_price': current_price,
            'ewma_forecast': ewma_forecast,
            'baseline_forecast': baseline_forecast
        })
        
        # Update tracking
        for model in preds.keys():
            was_correct = 1 if preds[model] == actual_dir else 0
            model_recent_correct[model].append(was_correct)
    
    return pd.DataFrame(results)

# Generate forecasts for each horizon
print("\n[1] Generating EWMA forecasts for each horizon...")
horizon_forecasts = {}

for h in HORIZONS:
    print(f"  Processing {h}d horizon...")
    df_h = df[df['n_predict'] == h].copy()
    top_models = get_top_models(df_h, TOP_N)
    params = optimal_params[h]
    
    forecasts = generate_ewma_forecasts(df_h, h, top_models, params['lookback'], params['alpha'])
    horizon_forecasts[h] = forecasts.set_index('date')
    print(f"    {len(forecasts)} forecasts generated")

# Align all horizons to common dates
print("\n[2] Aligning horizons to common dates...")
common_dates = None
for h, fc in horizon_forecasts.items():
    if common_dates is None:
        common_dates = set(fc.index)
    else:
        common_dates = common_dates.intersection(set(fc.index))

common_dates = sorted(list(common_dates))
print(f"  Common dates: {len(common_dates)}")

# Build forecast matrix
print("\n[3] Building forecast matrix...")
forecast_matrix = pd.DataFrame(index=common_dates)
baseline_matrix = pd.DataFrame(index=common_dates)

for h in HORIZONS:
    forecast_matrix[h] = horizon_forecasts[h].loc[common_dates, 'ewma_forecast']
    baseline_matrix[h] = horizon_forecasts[h].loc[common_dates, 'baseline_forecast']

# Get current prices
current_prices = horizon_forecasts[30].loc[common_dates, 'current_price']

# Calculate drift matrix signals
print("\n[4] Calculating drift matrix signals...")

def calc_drift_signals(matrix):
    """Calculate net_prob from drift matrix"""
    results = []
    
    for date in matrix.index:
        row = matrix.loc[date]
        
        slopes = []
        for i_idx, i in enumerate(HORIZONS):
            for j in HORIZONS[i_idx + 1:]:
                drift = row[j] - row[i]  # Longer horizon - shorter horizon
                slopes.append(drift)
        
        slopes = np.array(slopes)
        total_pairs = len(slopes)
        bullish_pairs = (slopes > 0).sum()
        bearish_pairs = (slopes < 0).sum()
        
        net_prob = (bullish_pairs - bearish_pairs) / total_pairs
        mean_drift = slopes.mean()
        
        results.append({
            'date': date,
            'bullish_prob': bullish_pairs / total_pairs,
            'bearish_prob': bearish_pairs / total_pairs,
            'net_prob': net_prob,
            'mean_drift': mean_drift
        })
    
    return pd.DataFrame(results).set_index('date')

ewma_signals = calc_drift_signals(forecast_matrix)
baseline_signals = calc_drift_signals(baseline_matrix)

print(f"  EWMA signals: {len(ewma_signals)}")
print(f"  Baseline signals: {len(baseline_signals)}")

# Save signals for analysis
ewma_signals.to_csv('data/21_USA_Beef_Tallow/ewma_snake_signals.csv')
baseline_signals.to_csv('data/21_USA_Beef_Tallow/baseline_snake_signals.csv')

# Generate Snake Chart
print("\n[5] Generating snake chart...")

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("EWMA Ensemble Snake (Color=Direction, Thickness=Consensus)", 
                    "Baseline Snake (For Comparison)"),
    vertical_spacing=0.12,
    row_heights=[0.5, 0.5]
)

def add_snake(fig, dates, prices, probs, row, name_prefix):
    """Add snake visualization to figure"""
    
    # Background price line
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        mode='lines', name=f'{name_prefix} Price',
        line=dict(color='rgba(255,255,255,0.3)', width=1),
        showlegend=False
    ), row=row, col=1)
    
    # Snake segments
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i+1]
        p0, p1 = prices[i], prices[i+1]
        score = probs[i]
        
        width = 2 + (abs(score) * 12)  # 2 to 14
        
        if score > 0.25:
            c_str = f'rgba(0, 255, 0, {0.5 + abs(score)*0.5})'  # Green
        elif score < -0.25:
            c_str = f'rgba(255, 50, 50, {0.5 + abs(score)*0.5})'  # Red
        else:
            c_str = 'rgba(100, 100, 100, 0.4)'  # Gray
        
        fig.add_trace(go.Scatter(
            x=[d0, d1], y=[p0, p1],
            mode='lines',
            line=dict(color=c_str, width=width),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=1)

# EWMA Snake
dates = list(ewma_signals.index)
prices = current_prices.loc[dates].values
ewma_probs = ewma_signals['net_prob'].values
add_snake(fig, dates, prices, ewma_probs, 1, "EWMA")

# Baseline Snake
baseline_probs = baseline_signals['net_prob'].values
add_snake(fig, dates, prices, baseline_probs, 2, "Baseline")

# Add legend/color guide
fig.add_annotation(
    x=0.5, y=1.08,
    xref="paper", yref="paper",
    text="🟢 Green = Bullish (>0.25) | 🔴 Red = Bearish (<-0.25) | ⬜ Gray = Neutral | Thickness = Consensus Strength",
    showarrow=False,
    font=dict(size=12, color='white')
)

fig.update_layout(
    height=900,
    title_text="EWMA Ensemble Snake Chart - USA Beef Tallow (Project 21)",
    template="plotly_dark",
    paper_bgcolor='#111111',
    plot_bgcolor='#111111',
    showlegend=False
)

fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', tickprefix='$')

# Save
fig.write_html('ewma_snake_chart.html')
print("\n✓ Saved to ewma_snake_chart.html")

# Print signal stats
print("\n" + "=" * 70)
print("SNAKE SIGNAL COMPARISON")
print("=" * 70)

ewma_bullish = (ewma_signals['net_prob'] > 0.25).sum()
ewma_bearish = (ewma_signals['net_prob'] < -0.25).sum()
ewma_neutral = len(ewma_signals) - ewma_bullish - ewma_bearish

base_bullish = (baseline_signals['net_prob'] > 0.25).sum()
base_bearish = (baseline_signals['net_prob'] < -0.25).sum()
base_neutral = len(baseline_signals) - base_bullish - base_bearish

print(f"\nEWMA Ensemble:")
print(f"  🟢 Bullish signals: {ewma_bullish} ({ewma_bullish/len(ewma_signals)*100:.1f}%)")
print(f"  🔴 Bearish signals: {ewma_bearish} ({ewma_bearish/len(ewma_signals)*100:.1f}%)")
print(f"  ⬜ Neutral signals: {ewma_neutral} ({ewma_neutral/len(ewma_signals)*100:.1f}%)")

print(f"\nBaseline:")
print(f"  🟢 Bullish signals: {base_bullish} ({base_bullish/len(baseline_signals)*100:.1f}%)")
print(f"  🔴 Bearish signals: {base_bearish} ({base_bearish/len(baseline_signals)*100:.1f}%)")
print(f"  ⬜ Neutral signals: {base_neutral} ({base_neutral/len(baseline_signals)*100:.1f}%)")

# Save JSON for dashboard integration
snake_data = {
    'dates': [d.strftime('%Y-%m-%d') for d in dates],
    'prices': prices.tolist(),
    'ewma_net_prob': ewma_probs.tolist(),
    'baseline_net_prob': baseline_probs.tolist()
}

with open('data/21_USA_Beef_Tallow/snake_data.json', 'w') as f:
    json.dump(snake_data, f)
print("\n✓ Snake data saved to data/21_USA_Beef_Tallow/snake_data.json")

