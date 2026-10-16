"""
รวม summary.csv ของทุก dataset/family ใน experiment เดียว เป็น Excel workbook เดียว

โครงสร้าง workbook ที่ได้:
  - 1 sheet ต่อ 1 family (convnext, resnet, vit, swin, swin_v2, efficientnet_v2, yolo26)
    รวมทุก dataset ไว้ในตารางเดียว มีคอลัมน์ 'dataset' บอกที่มา
    -> ใช้เทียบ performance ของ family เดียวกันข้าม dataset ได้ในตารางเดียว
  - 1 sheet ต่อ 1 dataset (2class, 5class, ...)
    รวมทุก family ไว้ในตารางเดียว มีคอลัมน์ 'family' บอกที่มา
    -> ใช้เทียบ family ต่างๆ บน dataset เดียวกัน

ใช้:
    python aggregate_results.py --experiment model_benchmark
    python aggregate_results.py --experiment model_benchmark --output /path/to/report.xlsx
"""
import argparse
from pathlib import Path

import pandas as pd

import config as c


def collect_all_runs(experiment_dir: Path) -> pd.DataFrame:
    """อ่าน summary.csv ทุกไฟล์ใต้ log/<dataset>/<family>/summary.csv
    คืน DataFrame เดียวรวมทุกอย่าง พร้อมคอลัมน์ 'dataset' และ 'family'"""
    log_root = experiment_dir / 'log'
    frames = []

    for dataset_dir in sorted(log_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for family_dir in sorted(dataset_dir.iterdir()):
            summary_path = family_dir / 'summary.csv'
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                df.insert(0, 'dataset', dataset_dir.name)
                df.insert(0, 'family', family_dir.name)
                frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def split_by(df: pd.DataFrame, key: str, sort_by: list[str]) -> dict[str, pd.DataFrame]:
    """แยก DataFrame รวมออกเป็น dict ตามคอลัมน์ key เช่น 'family' หรือ 'dataset'"""
    result = {}
    for value in sorted(df[key].unique()):
        sub = df[df[key] == value].sort_values(sort_by).reset_index(drop=True)
        result[value] = sub
    return result


def main():
    parser = argparse.ArgumentParser(description='รวม summary.csv ทั้งหมดเป็น Excel workbook เดียว')
    parser.add_argument('--experiment', '-e', type=str, default=c.EXPERIMENT_NAME)
    parser.add_argument('--output', '-o', type=str, default=None,
                         help='path ไฟล์ .xlsx ปลายทาง (default: <output_dir>/<experiment>/all_results.xlsx)')
    args = parser.parse_args()

    experiment_dir = Path(c.OUTPUT_DIR) / args.experiment
    out_path = Path(args.output) if args.output else experiment_dir / 'all_results.xlsx'

    all_runs = collect_all_runs(experiment_dir)
    if all_runs.empty:
        print(f"❌ ไม่เจอ summary.csv เลยใต้ {experiment_dir / 'log'}")
        return

    per_family = split_by(all_runs, key='family', sort_by=['dataset', 'variant'])
    per_dataset = split_by(all_runs, key='dataset', sort_by=['family', 'variant'])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # sheet ต่อ family ก่อน (เทียบข้าม dataset)
        for family_name, df in per_family.items():
            sheet_name = family_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # ตามด้วย sheet ต่อ dataset (เทียบข้าม family) เหมือนเดิม
        for dataset_name, df in per_dataset.items():
            sheet_name = dataset_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ เขียน {len(all_runs)} แถว รวม -> {len(per_family)} family sheet, "
          f"{len(per_dataset)} dataset sheet ไปที่ {out_path}")


if __name__ == '__main__':
    main()