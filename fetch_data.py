"""
Fetch Children Model Data from Local QML API
=============================================
This script fetches model forecast data from your localhost QML API
and saves it in the format expected by the LTEnsemble pipeline.

Usage:
    python fetch_data.py
    
The script uses settings from config.py:
    - PROJECT_ID, PROJECT_NAME
    - API_BASE_URL, API_ENDPOINT

Authentication:
    Set the QML_API_KEY environment variable:
    - Windows: set QML_API_KEY=your-api-key
    - Linux/Mac: export QML_API_KEY=your-api-key
    
Output:
    data/{PROJECT_ID}_{PROJECT_NAME}/all_children_data.parquet
"""
import os
import sys
import argparse
import requests
import pandas as pd

# Import configuration from config.py
from config import (
    PROJECT_ID, PROJECT_NAME, DATA_DIR, PARQUET_PATH,
    API_BASE_URL, API_ENDPOINT
)

# Get API key from environment variable
QML_API_KEY = os.getenv('QML_API_KEY', '')


def fetch_qml_models(project_id: str, api_key: str = None) -> pd.DataFrame:
    """
    Fetch QML model data from the API.
    
    Args:
        project_id: The project ID to fetch
        api_key: API key for authentication
        
    Returns:
        DataFrame with model forecasts
    """
    url = f"{API_BASE_URL}{API_ENDPOINT}/{project_id}"
    
    headers = {}
    if api_key:
        # QML API uses 'qml-api-key' header
        headers['qml-api-key'] = api_key
    
    print(f"  Fetching from: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=300)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        
        return df
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Could not connect to {API_BASE_URL}")
        print("   Make sure your QML server is running on localhost:5001")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print(f"\n❌ ERROR: Authentication required")
            print("   Use --api_token to provide your API token")
            print(f"   Get your token from: {API_BASE_URL}/user/get_api_token")
        elif e.response.status_code == 404:
            print(f"\n❌ ERROR: Project {project_id} not found")
        else:
            print(f"\n❌ ERROR: HTTP {e.response.status_code}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has the required columns.
    """
    required_columns = ['symbol', 'time', 'n_predict', 'target_var_price', 'close_predict', 'yn_actual']
    
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        print(f"\n⚠️  WARNING: Missing columns: {missing}")
        print(f"   Available columns: {list(df.columns)}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Fetch QML model data for LTEnsemble pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Settings from config.py:
    PROJECT_ID:   {PROJECT_ID}
    PROJECT_NAME: {PROJECT_NAME}
    API_BASE_URL: {API_BASE_URL}

Authentication:
    Set environment variable QML_API_KEY before running:
    - Windows:    set QML_API_KEY=your-api-key
    - Linux/Mac:  export QML_API_KEY=your-api-key
    
Examples:
    python fetch_data.py
        """
    )
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = QML_API_KEY
    
    print("=" * 70)
    print("FETCH QML MODEL DATA FOR LTENSEMBLE")
    print("=" * 70)
    print(f"\n  Project ID:   {PROJECT_ID}")
    print(f"  Project Name: {PROJECT_NAME}")
    print(f"  API URL:      {API_BASE_URL}")
    print(f"  API Key:      {'✓ Found in QML_API_KEY env var' if api_key else '✗ Not set (QML_API_KEY)'}")
    
    if not api_key:
        print("\n⚠️  WARNING: QML_API_KEY environment variable not set")
        print("   Set it with: set QML_API_KEY=your-api-key  (Windows)")
        print("   Or:          export QML_API_KEY=your-api-key  (Linux/Mac)")
    
    # Create output directory
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\n  Output dir:   {DATA_DIR}/")
    
    # Fetch data
    print("\n[1] Fetching data from API...")
    df = fetch_qml_models(PROJECT_ID, api_key)
    
    print(f"    ✓ Received {len(df):,} records")
    print(f"    ✓ Columns: {list(df.columns)}")
    
    # Validate
    print("\n[2] Validating data...")
    if validate_data(df):
        print("    ✓ All required columns present")
    else:
        print("    ⚠️  Some columns missing - pipeline may not work correctly")
    
    # Convert time column
    print("\n[3] Processing data...")
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        print(f"    ✓ Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Show horizons
    if 'n_predict' in df.columns:
        horizons = sorted(df['n_predict'].unique())
        print(f"    ✓ Horizons: {[int(h) for h in horizons]}")
    
    # Show model count
    if 'symbol' in df.columns:
        n_models = df['symbol'].nunique()
        print(f"    ✓ Unique models: {n_models:,}")
    
    # Save to parquet
    print("\n[4] Saving data...")
    df.to_parquet(PARQUET_PATH, index=False)
    
    file_size = os.path.getsize(PARQUET_PATH) / (1024 * 1024)
    print(f"    ✓ Saved to: {PARQUET_PATH}")
    print(f"    ✓ File size: {file_size:.1f} MB")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DATA FETCH COMPLETE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. python grid_search_ewma.py")
    print(f"  2. python baseline_vs_ewma.py")
    print(f"  3. python generate_equity_data.py")
    print(f"  4. python generate_dashboard_plotly.py")
    print()


if __name__ == '__main__':
    main()
