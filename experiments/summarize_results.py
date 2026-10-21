#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main():
    out = Path("outputs/results")
    cc = out / "cross_city_summary.csv"
    ab = out / "ablation_nyc.csv"

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    if cc.exists():
        df = pd.read_csv(cc)
        # Show best in-city (train=test) by R2 per model
        same = df[df["train"] == df["test"]].copy()
        if not same.empty:
            same = same.sort_values(["model", "R2"], ascending=[True, False])
            print("\nCross-city eval (in-city baselines):")
            print(
                same[["train", "test", "model", "R2", "MAE", "RMSE", "Spearman"]].to_string(
                    index=False
                )
            )
        else:
            print("\nCross-city eval: (no train==test rows found)")
    else:
        print("\nCross-city eval: missing outputs/results/cross_city_summary.csv")

    if ab.exists():
        df = pd.read_csv(ab).sort_values("r2", ascending=False)
        print("\nNYC ablation (top configs):")
        print(df.head(8).to_string(index=False))
        best = df.iloc[0]
        print(
            f"\nBest NYC config: {best['feature_set']} | {best['model']} | "
            f"log={best['log_target']} -> R2={best['r2']:.3f}, MAE={best['mae']:.0f}"
        )
    else:
        print("\nNYC ablation: missing outputs/results/ablation_nyc.csv")


if __name__ == "__main__":
    main()
