"""
รวม summary.csv ของทุก dataset/family ใน experiment เดียว เป็น Excel workbook
เดียว โดยแยกเป็น 1 sheet ต่อ 1 dataset (เหมือนที่ทำมือใน Google Sheets)

ใช้:
    python aggregate_results.py --experiment model_benchmark
    python aggregate_results.py --experiment model_benchmark --output /path/to/report.xlsx
"""
import argparse
from pathlib import Path

import pandas as pd

import config as c


def collect_summaries(experiment_dir: Path):
    """คืน dict: dataset_name -> DataFrame ที่รวม summary.csv ของทุก family แล้ว"""
    log_root = experiment_dir / 'log'
    per_dataset = {}

    for dataset_dir in sorted(log_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        frames = []
        for family_dir in sorted(dataset_dir.iterdir()):
            summary_path = family_dir / 'summary.csv'
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                df.insert(0, 'family', family_dir.name)
                frames.append(df)
        if frames:
            per_dataset[dataset_dir.name] = pd.concat(frames, ignore_index=True)

    return per_dataset


def main():
    parser = argparse.ArgumentParser(description='รวม summary.csv ทั้งหมดเป็น Excel workbook เดียว')
    parser.add_argument('--experiment', '-e', type=str, default=c.EXPERIMENT_NAME)
    parser.add_argument('--output', '-o', type=str, default=None,
                         help='path ไฟล์ .xlsx ปลายทาง (default: <output_dir>/<experiment>/all_results.xlsx)')
    args = parser.parse_args()

    experiment_dir = Path(c.OUTPUT_DIR) / args.experiment
    out_path = Path(args.output) if args.output else experiment_dir / 'all_results.xlsx'

    per_dataset = collect_summaries(experiment_dir)
    if not per_dataset:
        print(f"❌ ไม่เจอ summary.csv เลยใต้ {experiment_dir / 'log'}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for dataset_name, df in per_dataset.items():
            sheet_name = dataset_name[:31]  # Excel จำกัดชื่อ sheet 31 ตัวอักษร
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    total_rows = sum(len(df) for df in per_dataset.values())
    print(f"✅ เขียน {total_rows} แถว ครอบคลุม {len(per_dataset)} dataset sheet ไปที่ {out_path}")


if __name__ == '__main__':
    main()