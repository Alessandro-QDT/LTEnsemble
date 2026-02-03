"""
Update ewma_dashboard_v4.html with Live Forecasts
=================================================
Adds a new chart section showing historical prices + forecast line
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import re

# Import project configuration
from config import PARQUET_PATH, LIVE_FORECAST, PROJECT_ID, PROJECT_NAME

print("=" * 70)
print("UPDATING DASHBOARD WITH LIVE FORECASTS")
print("=" * 70)
print(f"Project: {PROJECT_ID} - {PROJECT_NAME}")

# Load data
print("\n[1] Loading data...")
if not os.path.exists(PARQUET_PATH):
    print(f"\n❌ ERROR: Data file not found: {PARQUET_PATH}")
    exit(1)

df = pd.read_parquet(PARQUET_PATH)
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

# Get price history
prices = df.groupby('time')['target_var_price'].first().reset_index()
prices.columns = ['date', 'price']
prices = prices.sort_values('date')
prices = prices[prices['date'] >= '2024-01-01']  # Last 2 years

latest_date = prices['date'].max()
current_price = prices['price'].iloc[-1]

print(f"    Latest date: {latest_date.strftime('%Y-%m-%d')}")
print(f"    Current price: ${current_price:.2f}")

# Load live forecasts
if not os.path.exists(LIVE_FORECAST):
    print(f"\n❌ ERROR: Forecast file not found: {LIVE_FORECAST}")
    exit(1)

with open(LIVE_FORECAST, 'r') as f:
    forecast_data = json.load(f)

forecasts = forecast_data['forecasts']
print(f"    Loaded {len(forecasts)} forecasts")

# Prepare data for chart
historical_dates = prices['date'].dt.strftime('%Y-%m-%d').tolist()
historical_prices = prices['price'].tolist()

# Forecast data: from latest date extending to each horizon
forecast_points = []
for fc in forecasts:
    forecast_points.append({
        'horizon': fc['horizon'],
        'date': fc['target_date'],
        'price': fc['predicted_price'],
        'change': fc['pct_change'],
        'direction': fc['direction']
    })

# Sort by horizon
forecast_points = sorted(forecast_points, key=lambda x: x['horizon'])

print("\n[2] Creating forecast section HTML...")

# Build the forecast chart data as JSON
forecast_chart_data = {
    'historical': {
        'dates': historical_dates,
        'prices': historical_prices
    },
    'current': {
        'date': latest_date.strftime('%Y-%m-%d'),
        'price': current_price
    },
    'forecasts': forecast_points
}

# Read the current dashboard
with open('ewma_dashboard_v4.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Create the new forecast section HTML
forecast_section = '''
        <!-- Live Forecast Chart -->
        <div class="chart-section">
            <div class="section-header">
                <span class="section-title">🔮 Live EWMA Forecasts</span>
                <span style="color: #888; font-size: 0.9rem;">Current: $''' + f'{current_price:.2f}' + ''' | As of: ''' + latest_date.strftime('%Y-%m-%d') + '''</span>
            </div>
            <div id="forecastChart" class="chart-container"></div>
            <div class="chart-hint">
                📊 Dotted line shows EWMA ensemble forecasts • Diamond markers indicate forecast targets
            </div>
            
            <!-- Forecast Summary Cards -->
            <div class="metrics-row" style="margin-top: 20px;">
'''

# Add forecast cards
for fc in forecast_points:
    color = '#10b981' if fc['direction'] == 'UP' else '#ef4444'
    arrow = '↑' if fc['direction'] == 'UP' else '↓'
    forecast_section += f'''
                <div class="metric-card">
                    <div class="metric-label">{fc['horizon']}-Day Forecast</div>
                    <div class="metric-value" style="color: {color};">{arrow} ${fc['price']:.2f}</div>
                    <div style="font-size: 0.85rem; color: {color};">{fc['change']:+.2f}%</div>
                    <div style="font-size: 0.75rem; color: #888;">by {fc['date']}</div>
                </div>
'''

forecast_section += '''
            </div>
        </div>
'''

# Find where to insert (after the Equity Curve section, before the Comparison Table)
insert_marker = '<!-- Comparison Table -->'
html_content = html_content.replace(insert_marker, forecast_section + '\n        ' + insert_marker)

# Add the forecast chart initialization JavaScript
forecast_js = '''
        // Live Forecast Data
        const forecastData = ''' + json.dumps(forecast_chart_data) + ''';
        
        function initForecastChart() {
            const hist = forecastData.historical;
            const forecasts = forecastData.forecasts;
            const currentDate = forecastData.current.date;
            const currentPrice = forecastData.current.price;
            
            // Create forecast line: current price → each forecast point
            const forecastDates = [currentDate];
            const forecastPrices = [currentPrice];
            
            forecasts.forEach(f => {
                forecastDates.push(f.date);
                forecastPrices.push(f.price);
            });
            
            const traces = [
                // Historical price line
                {
                    x: hist.dates,
                    y: hist.prices,
                    name: 'Historical Price',
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#00d4ff', width: 2 },
                    hovertemplate: '%{x}<br>$%{y:.2f}<extra>Historical</extra>'
                },
                // Forecast line (dotted)
                {
                    x: forecastDates,
                    y: forecastPrices,
                    name: 'EWMA Forecast',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#10b981', width: 3, dash: 'dot' },
                    marker: { 
                        size: forecastDates.map((d, i) => i === 0 ? 10 : 14),
                        symbol: forecastDates.map((d, i) => i === 0 ? 'circle' : 'diamond'),
                        color: '#10b981'
                    },
                    hovertemplate: '%{x}<br>$%{y:.2f}<extra>Forecast</extra>'
                }
            ];
            
            // Add annotations for each forecast point
            const annotations = forecasts.map(f => ({
                x: f.date,
                y: f.price,
                text: f.horizon + 'd<br>$' + f.price.toFixed(2) + '<br>' + (f.change >= 0 ? '+' : '') + f.change.toFixed(1) + '%',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 1,
                arrowcolor: f.direction === 'UP' ? '#10b981' : '#ef4444',
                font: { size: 10, color: 'white' },
                bgcolor: f.direction === 'UP' ? 'rgba(16, 185, 129, 0.9)' : 'rgba(239, 68, 68, 0.9)',
                bordercolor: f.direction === 'UP' ? '#10b981' : '#ef4444',
                borderwidth: 1,
                borderpad: 4,
                ax: 0,
                ay: -40
            }));
            
            const layout = {
                ...darkLayout,
                annotations: annotations,
                xaxis: {
                    ...darkLayout.xaxis,
                    range: [hist.dates[Math.floor(hist.dates.length * 0.7)], forecasts[forecasts.length - 1].date]
                },
                shapes: [{
                    type: 'line',
                    x0: currentDate,
                    x1: currentDate,
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: { color: 'rgba(255,255,255,0.5)', width: 2, dash: 'dash' }
                }]
            };
            
            Plotly.newPlot('forecastChart', traces, layout, config);
        }
'''

# Find where to add the forecast JS (after updateEquityChart function)
js_insert_marker = 'updatePriceChart();'
html_content = html_content.replace(js_insert_marker, forecast_js + '\n        ' + js_insert_marker)

# Add call to init forecast chart
init_marker = 'updateEquityChart();'
html_content = html_content.replace(init_marker, init_marker + '\n        initForecastChart();')

# Save updated dashboard
output_path = 'ewma_dashboard_v5.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✅ Updated dashboard saved to: {output_path}")
print(f"\n📊 Forecasts added:")
for fc in forecast_points:
    emoji = "🟢" if fc['direction'] == 'UP' else "🔴"
    print(f"   {emoji} {fc['horizon']:>3}d → ${fc['price']:.2f} ({fc['change']:+.2f}%) by {fc['date']}")

