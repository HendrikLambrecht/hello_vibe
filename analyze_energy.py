#!/usr/bin/env python3
"""
Analysis of energy consumption data to detect the effect of a new device (~10-20W).
Focuses on nighttime intervals (1:00-5:00 AM) where Netzbezug = 0.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "/workspace/HendrikLambrecht__hello_vibe/energy_data"
NIGHTTIME_START = 1  # 1:00 AM
NIGHTTIME_END = 5    # 5:00 AM
INSTALLATION_DATE = datetime(2026, 5, 13)

# --- Helper Functions ---
def parse_time(time_str):
    """Parse time string like '="01:00"' to hour and minute."""
    # Remove quotes and = signs
    time_str = time_str.strip('"=').strip()
    try:
        hour, minute = map(int, time_str.split(':'))
        return hour, minute
    except:
        return None, None

def is_nighttime(hour, minute):
    """Check if the time is between 1:00 and 5:00 AM."""
    return NIGHTTIME_START <= hour < NIGHTTIME_END

def get_date_from_filename(filename):
    """Extract date from filename like Energiebilanz_2026_05_12.csv."""
    basename = os.path.basename(filename)
    # Remove prefix and extension
    date_str = basename.replace("Energiebilanz_", "").replace(".csv", "")
    try:
        return datetime.strptime(date_str, "%Y_%m_%d")
    except:
        return None

def process_file(filepath):
    """Process a single CSV file and return nighttime data with Netzbezug=0."""
    date = get_date_from_filename(filepath)
    if date is None:
        print(f"Skipping {filepath} (invalid date)")
        return None
    
    # Read CSV (skip BOM and handle semicolon delimiter)
    try:
        df = pd.read_csv(filepath, delimiter=';', skipinitialspace=True, encoding='utf-8-sig')
    except:
        print(f"Failed to read {filepath}")
        return None
    
    # Parse time column (first column)
    time_col = df.iloc[:, 0]
    netzbezug_col = df.iloc[:, 3]  # Netzbezug / Mittelwerte [W]
    gesamtverbrauch_col = df.iloc[:, 4]  # Gesamtverbrauch / Mittelwerte [W]
    
    nighttime_data = []
    for idx, time_str in enumerate(time_col):
        hour, minute = parse_time(time_str)
        if hour is None:
            continue
        if is_nighttime(hour, minute):
            netzbezug = netzbezug_col.iloc[idx]
            gesamtverbrauch = gesamtverbrauch_col.iloc[idx]
            if pd.notna(netzbezug) and pd.notna(gesamtverbrauch):
                if netzbezug == 0:
                    nighttime_data.append({
                        'date': date,
                        'hour': hour,
                        'minute': minute,
                        'Netzbezug': netzbezug,
                        'Gesamtverbrauch': gesamtverbrauch
                    })
    
    if not nighttime_data:
        print(f"No nighttime data with Netzbezug=0 for {date.strftime('%Y-%m-%d')}")
    
    return pd.DataFrame(nighttime_data)

# --- Main Analysis ---
def main():
    # Step 1: Load all CSV files
    csv_files = glob.glob(os.path.join(DATA_DIR, "Energiebilanz_*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    # Step 2: Process all files and collect nighttime data
    all_nighttime_data = []
    for filepath in csv_files:
        df = process_file(filepath)
        if df is not None and not df.empty:
            all_nighttime_data.append(df)
    
    if not all_nighttime_data:
        print("No nighttime data found!")
        return
    
    nighttime_df = pd.concat(all_nighttime_data, ignore_index=True)
    print(f"Total nighttime intervals with Netzbezug=0: {len(nighttime_df)}")
    
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
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(
        post_installation['Gesamtverbrauch'],
        pre_installation['Gesamtverbrauch'],
        equal_var=False  # Welch's t-test (unequal variances)
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
    
    plt.title('Nighttime Gesamtverbrauch (1:00-5:00 AM, Netzbezug=0)')
    plt.xlabel('Date')
    plt.ylabel('Gesamtverbrauch [W]')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(DATA_DIR, "nighttime_consumption.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    # Step 7: Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['Pre-installation Avg', 'Post-installation Avg', 'Δ (Post - Pre)', 'p-value'],
        'Value': [f"{pre_avg:.2f} W", f"{post_avg:.2f} W", f"{delta:.2f} W", f"{p_value:.4f}"]
    })
    results_path = os.path.join(DATA_DIR, "analysis_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
