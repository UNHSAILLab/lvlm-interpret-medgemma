#!/usr/bin/env python3
"""
Monitor the progress of the phrasing analysis
"""

import time
import json
from pathlib import Path
from datetime import datetime

def check_progress():
    results_dir = Path('question_phrasing_results')
    
    # Check if detailed results exist
    results_file = results_dir / 'detailed_results.json'
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            try:
                results = json.load(f)
                return len(results)
            except:
                return 0
    return 0

def main():
    print("Monitoring phrasing analysis progress...")
    print("-" * 50)
    
    last_count = 0
    start_time = time.time()
    
    while True:
        current_count = check_progress()
        
        if current_count > last_count:
            elapsed = time.time() - start_time
            elapsed_min = elapsed / 60
            rate = current_count / elapsed_min if elapsed_min > 0 else 0
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed: {current_count}/63 groups")
            print(f"  Rate: {rate:.1f} groups/min")
            
            if current_count > 0:
                eta_min = (63 - current_count) / rate if rate > 0 else 0
                print(f"  Estimated time remaining: {eta_min:.1f} minutes")
            
            last_count = current_count
            
            if current_count >= 63:
                print("\n✓ Analysis complete!")
                break
        
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    main()