# -*- coding: utf-8 -*-
"""Check preprocessing report status."""
import pandas as pd
from pathlib import Path

def main():
    report_path = Path("data/processed/preprocessing_report.csv")
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return
        
    df = pd.read_csv(report_path)
    print("Preprocessing report shape:", df.shape)
    print("\nStatus value counts:")
    print(df["status"].value_counts())
    
    errors = df[df["status"] != "success"]
    if not errors.empty:
        print(f"\nFound {len(errors)} non-successful cases:")
        print(errors[["case_id", "status", "error_message"]])
    else:
        print("\nAll cases preprocessed successfully (100% success)!")

if __name__ == "__main__":
    main()
