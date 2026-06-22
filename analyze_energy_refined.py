#!/usr/bin/env python3
"""
Refined analysis of energy consumption data to detect the effect of a new device (~10-20W).
Uses Gesamtverbrauch directly and applies a smart nighttime window.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

# --- Configuration ---
DATA_DIR = "/workspace/HendrikLambrecht__hello_vibe/energy_data"
NIGHTTIME_START = 0  # Start at 00:00
CONSUMPTION_LOWER = 80  # Lower bound for "normal" consumption (W)
CONSUMPTION_UPPER = 130  # Upper bound for "normal" consumption (W)
INSTALLATION_DATE = datetime(2026, 5, 13)

# --- Helper Functions ---
def parse_time(time_str):
    """Parse time string like '="01:00"' to hour and minute."""
    time_str = time_str.strip('"=').strip()
    try:
        hour, minute = map(int, time_str.split(':'))
        return hour, minute
    except:
        return None, None

def get_date_from_filename(filename):
    """Extract date from filename like Energiebilanz_2026_05_12.csv."""
    basename = os.path.basename(filename)
    date_str = basename.replace("Energiebilanz_", "").replace(".csv", "")
    try:
        return datetime.strptime(date_str, "%Y_%m_%d")
    except:
        return None

def process_file(filepath):
    """Process a single CSV file and return smart nighttime data."""
    date = get_date_from_filename(filepath)
    if date is None:
        print(f"Skipping {filepath} (invalid date)")
        return None
    
    try:
        df = pd.read_csv(filepath, delimiter=';', skipinitialspace=True, encoding='utf-8-sig')
    except:
        print(f"Failed to read {filepath}")
        return None
    
    # Extract columns
    time_col = df.iloc[:, 0]
    gesamtverbrauch_col = df.iloc[:, 4]  # Gesamtverbrauch / Mittelwerte [W]
    
    # Parse all intervals and find the dynamic nighttime end
    intervals = []
    nighttime_end_hour = None
    
    for idx, time_str in enumerate(time_col):
        hour, minute = parse_time(time_str)
        if hour is None:
            continue
        
        gesamtverbrauch = gesamtverbrauch_col.iloc[idx]
        if pd.isna(gesamtverbrauch):
            continue
        
        # Store all intervals for later filtering
        intervals.append({
            'date': date,
            'hour': hour,
            'minute': minute,
            'Gesamtverbrauch': gesamtverbrauch
        })
        
        # Determine nighttime end: first interval where Gesamtverbrauch > 130W
        if hour >= NIGHTTIME_START and gesamtverbrauch > CONSUMPTION_UPPER:
            if nighttime_end_hour is None:
                nighttime_end_hour = hour
    
    # If no nighttime end found (all intervals <= 130W), set to 5:00 AM
    if nighttime_end_hour is None:
        nighttime_end_hour = 5
    
    # Filter intervals for smart nighttime window and normal consumption
    nighttime_data = []
    for interval in intervals:
        hour = interval['hour']
        gesamtverbrauch = interval['Gesamtverbrauch']
        
        # Check if within nighttime window and normal consumption range
        if (NIGHTTIME_START <= hour < nighttime_end_hour and 
            CONSUMPTION_LOWER <= gesamtverbrauch <= CONSUMPTION_UPPER):
            nighttime_data.append(interval)
    
    if not nighttime_data:
        print(f"No smart nighttime data for {date.strftime('%Y-%m-%d')}")
    
    return pd.DataFrame(nighttime_data)

# --- Main Analysis ---
def main():
    # Step 1: Load all CSV files
    csv_files = glob.glob(os.path.join(DATA_DIR, "Energiebilanz_*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    # Step 2: Process all files and collect smart nighttime data
    all_nighttime_data = []
    for filepath in csv_files:
        df = process_file(filepath)
        if df is not None and not df.empty:
            all_nighttime_data.append(df)
    
    if not all_nighttime_data:
        print("No smart nighttime data found!")
        return
    
    nighttime_df = pd.concat(all_nighttime_data, ignore_index=True)
    print(f"Total smart nighttime intervals: {len(nighttime_df)}")
    
    # Step 3: Segment into pre- and post-installation
    pre_installation = nighttime_df[nighttime_df['date'] < INSTALLATION_DATE]
    post_installation = nighttime_df[nighttime_df['date'] >= INSTALLATION_DATE]
    
    print(f"\nPre-installation (before {INSTALLATION_DATE.strftime('%Y-%m-%d')}):")
    print(f"  - Days: {pre_installation['date'].nunique()}")
    print(f"  - Intervals: {len(pre_installation)}")
    
    print(f"\nPost-installation (on/after {INSTALLATION_DATE.strftime('%Y-%m-%d')}):")
    print(f"  - Days: {post_installation['date'].nunique()}")
    print(f"  - Intervals: {len(post_installation)}")
    
    # Step 4: Compute statistics
    pre_avg = pre_installation['Gesamtverbrauch'].mean()
    pre_std = pre_installation['Gesamtverbrauch'].std()
    post_avg = post_installation['Gesamtverbrauch'].mean()
    post_std = post_installation['Gesamtverbrauch'].std()
    delta = post_avg - pre_avg
    
    print(f"\n--- Results ---")
    print(f"Pre-installation avg Gesamtverbrauch: {pre_avg:.2f} W (±{pre_std:.2f} W)")
    print(f"Post-installation avg Gesamtverbrauch: {post_avg:.2f} W (±{post_std:.2f} W)")
    print(f"Δ (Post - Pre): {delta:.2f} W")
    
    # Step 5: Statistical significance (t-test)
    t_stat, p_value = stats.ttest_ind(
        post_installation['Gesamtverbrauch'],
        pre_installation['Gesamtverbrauch'],
        equal_var=False
    )
    print(f"\nStatistical test (Welch's t-test):")
    print(f"  - t-statistic: {t_stat:.3f}")
    print(f"  - p-value: {p_value:.4f}")
    print(f"  - Significant (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}")
    
    # Step 6: Plot the data
    plt.figure(figsize=(12, 6))
    
    # Group by date and compute daily average
    daily_pre = pre_installation.groupby('date')['Gesamtverbrauch'].mean()
    daily_post = post_installation.groupby('date')['Gesamtverbrauch'].mean()
    
    plt.scatter(daily_pre.index, daily_pre.values, color='blue', label='Pre-installation (Daily Avg)')
    plt.scatter(daily_post.index, daily_post.values, color='red', label='Post-installation (Daily Avg)')
    
    # Add horizontal lines for overall averages
    plt.axhline(y=pre_avg, color='blue', linestyle='--', label=f'Pre-avg: {pre_avg:.1f} W')
    plt.axhline(y=post_avg, color='red', linestyle='--', label=f'Post-avg: {post_avg:.1f} W')
    
    plt.title('Smart Nighttime Gesamtverbrauch (00:00 - Dynamic End, 80W ≤ Consumption ≤ 130W)')
    plt.xlabel('Date')
    plt.ylabel('Gesamtverbrauch [W]')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(DATA_DIR, "smart_nighttime_consumption.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    # Step 7: Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['Pre-installation Avg', 'Post-installation Avg', 'Δ (Post - Pre)', 'p-value'],
        'Value': [f"{pre_avg:.2f} W", f"{post_avg:.2f} W", f"{delta:.2f} W", f"{p_value:.4f}"]
    })
    results_path = os.path.join(DATA_DIR, "refined_analysis_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
