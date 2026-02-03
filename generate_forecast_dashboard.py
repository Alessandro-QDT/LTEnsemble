"""
Generate Beef Tallow Dashboard with Live EWMA Forecasts
=======================================================
Shows historical prices + live forecast lines for each horizon
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("=" * 70)
print("GENERATING BEEF TALLOW FORECAST DASHBOARD")
print("=" * 70)

# Load data
print("\n[1] Loading data...")
df = pd.read_parquet('data/21_USA_Beef_Tallow/all_children_data.parquet')
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

# Get unique prices (from target_var_price which is the current price at each date)
prices = df.groupby('time')['target_var_price'].first().reset_index()
prices.columns = ['date', 'price']
prices = prices.sort_values('date')

latest_date = prices['date'].max()
current_price = prices['price'].iloc[-1]

print(f"    Price data: {len(prices)} days")
print(f"    Latest date: {latest_date.strftime('%Y-%m-%d')}")
print(f"    Current price: ${current_price:.2f}")

# Optimal EWMA parameters
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
    return [w / total for w in weights] if total > 0 else [1/n] * n

def get_top_models(df_horizon, n=5):
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
    return perf_df.head(n)['symbol'].tolist()

def get_live_forecast(df_horizon, top_models, lookback, alpha):
    df_h = df_horizon[df_horizon['symbol'].isin(top_models)].copy()
    df_h = df_h[['time', 'symbol', 'close_predict', 'target_var_price', 'yn_actual']].copy()
    
    dates = sorted(df_h['time'].unique())
    if len(dates) < lookback + 1:
        return None
    
    recent_dates = dates[-(lookback + 1):]
    model_recent_correct = {m: [] for m in top_models}
    
    for date in recent_dates[:-1]:
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
    
    if len(preds) < 2 or current_price is None:
        return None
    
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
    ewma_pred_dir = np.sign(ewma_vote)
    
    weighted_price = sum(pred_prices[m] * norm_weights[m] for m in preds.keys())
    confidence = abs(ewma_vote)
    
    return {
        'current_price': current_price,
        'predicted_price': weighted_price,
        'direction': 'UP' if ewma_pred_dir > 0 else 'DOWN' if ewma_pred_dir < 0 else 'NEUTRAL',
        'confidence': confidence,
        'pct_change': ((weighted_price - current_price) / current_price) * 100
    }

# Generate forecasts
print("\n[2] Generating EWMA Ensemble Forecasts...")
forecasts = {}

for horizon in sorted(optimal_params.keys()):
    params = optimal_params[horizon]
    lookback = params['lookback']
    alpha = params['alpha']
    
    df_h = df[df['n_predict'] == horizon].copy()
    
    if len(df_h) == 0:
        continue
    
    top_models = get_top_models(df_h, TOP_N)
    
    if len(top_models) < 2:
        continue
    
    forecast = get_live_forecast(df_h, top_models, lookback, alpha)
    
    if forecast and forecast['predicted_price'] > 0:  # Filter out invalid forecasts
        forecasts[horizon] = forecast
        print(f"    {horizon}d: {forecast['direction']} → ${forecast['predicted_price']:.2f} ({forecast['pct_change']:+.2f}%)")

# Create dashboard
print("\n[3] Creating dashboard...")

# Filter historical prices to last 2 years
recent_prices = prices[prices['date'] >= '2024-01-01'].copy()

fig = make_subplots(
    rows=1, cols=1,
    subplot_titles=["USA Beef Tallow - Historical Prices + EWMA Forecasts"]
)

# Add historical prices
fig.add_trace(go.Scatter(
    x=recent_prices['date'],
    y=recent_prices['price'],
    mode='lines',
    name='Historical Price',
    line=dict(color='#00d4ff', width=2)
))

# Add forecast lines
colors = {
    30: '#10b981',   # Green
    60: '#22c55e',   # Light green
    90: '#84cc16',   # Lime
    180: '#eab308',  # Yellow
    270: '#f97316',  # Orange
    360: '#ef4444',  # Red
    450: '#ec4899',  # Pink
}

for horizon, forecast in sorted(forecasts.items()):
    target_date = latest_date + timedelta(days=horizon)
    
    # Draw line from current price to forecast (convert dates to strings)
    fig.add_trace(go.Scatter(
        x=[latest_date.strftime('%Y-%m-%d'), target_date.strftime('%Y-%m-%d')],
        y=[current_price, forecast['predicted_price']],
        mode='lines+markers',
        name=f"{horizon}d: ${forecast['predicted_price']:.2f} ({forecast['pct_change']:+.1f}%)",
        line=dict(color=colors.get(horizon, '#888'), width=3, dash='dot'),
        marker=dict(size=[8, 12], symbol=['circle', 'diamond'])
    ))
    
    # Add annotation for the forecast point
    fig.add_annotation(
        x=target_date.strftime('%Y-%m-%d'),
        y=forecast['predicted_price'],
        text=f"{horizon}d<br>${forecast['predicted_price']:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=colors.get(horizon, '#888'),
        font=dict(size=10, color='white'),
        bgcolor=colors.get(horizon, '#888'),
        bordercolor=colors.get(horizon, '#888'),
        borderwidth=1,
        borderpad=3,
        ax=20,
        ay=-30
    )

# Add vertical line for "today" using shape instead of vline
fig.add_shape(
    type="line",
    x0=latest_date.strftime('%Y-%m-%d'),
    x1=latest_date.strftime('%Y-%m-%d'),
    y0=0,
    y1=1,
    yref="paper",
    line=dict(color="white", width=2, dash="dash")
)

# Add "Today" annotation
fig.add_annotation(
    x=latest_date.strftime('%Y-%m-%d'),
    y=1.02,
    yref="paper",
    text=f"Today: ${current_price:.2f}",
    showarrow=False,
    font=dict(color="white", size=12),
    bgcolor="rgba(0,0,0,0.7)",
    borderpad=4
)

# Update layout
fig.update_layout(
    title=dict(
        text=f"<b>USA Beef Tallow</b> - EWMA Ensemble Forecasts<br><sup>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Current: ${current_price:.2f}</sup>",
        font=dict(size=20, color='white'),
        x=0.5
    ),
    template="plotly_dark",
    paper_bgcolor='#0a0a0f',
    plot_bgcolor='#0a0a0f',
    height=700,
    xaxis_title="Date",
    yaxis_title="Price ($/unit)",
    hovermode="x unified",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(0,0,0,0.5)'
    )
)

fig.update_xaxes(
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color='#888'),
    range=[recent_prices['date'].min().strftime('%Y-%m-%d'), (latest_date + timedelta(days=500)).strftime('%Y-%m-%d')]
)

fig.update_yaxes(
    gridcolor='rgba(255,255,255,0.1)',
    tickprefix='$',
    tickfont=dict(color='#888')
)

# Save dashboard
output_path = 'ewma_forecast_dashboard.html'
fig.write_html(output_path)
print(f"\n✅ Dashboard saved to: {output_path}")

# Also create a summary JSON
summary = {
    'generated_at': datetime.now().isoformat(),
    'current_price': current_price,
    'current_date': latest_date.strftime('%Y-%m-%d'),
    'forecasts': []
}

for horizon, forecast in sorted(forecasts.items()):
    summary['forecasts'].append({
        'horizon': horizon,
        'target_date': (latest_date + timedelta(days=horizon)).strftime('%Y-%m-%d'),
        'direction': forecast['direction'],
        'predicted_price': round(forecast['predicted_price'], 2),
        'pct_change': round(forecast['pct_change'], 2),
        'confidence': round(forecast['confidence'], 2)
    })

with open('data/21_USA_Beef_Tallow/live_ewma_forecast.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Forecast JSON saved to: data/21_USA_Beef_Tallow/live_ewma_forecast.json")

# Print summary
print("\n" + "=" * 70)
print("📊 FORECAST SUMMARY")
print("=" * 70)
print(f"Current Price: ${current_price:.2f}")
print(f"As of: {latest_date.strftime('%Y-%m-%d')}")
print()
for horizon, forecast in sorted(forecasts.items()):
    emoji = "🟢" if forecast['direction'] == 'UP' else "🔴"
    print(f"  {emoji} {horizon:>3}d → ${forecast['predicted_price']:.2f} ({forecast['pct_change']:+.2f}%) by {(latest_date + timedelta(days=horizon)).strftime('%Y-%m-%d')}")

