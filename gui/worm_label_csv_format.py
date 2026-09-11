import argparse
import shutil
import sys
from pathlib import Path
import hashlib
import pandas as pd


def backup_csv_file(csv_path: Path) -> Path | None:
    """Copy csv_path -> csv_path.bak ก่อนเขียนทับไฟล์เดิมทุกครั้ง"""
    if not csv_path.exists():
        return None
    backup_path = csv_path.with_suffix(csv_path.suffix + '.bak')
    try:
        shutil.copy2(csv_path, backup_path)  # copy2 = คง metadata/mtime ไว้ด้วย
        return backup_path
    except Exception as e:
        raise RuntimeError(f"ไม่สามารถสำรองไฟล์ '{csv_path}' ก่อนบันทึกได้: {e}")


def get_file_hash(filepath: Path, chunk_size=8192) -> str:
    """คำนวณ MD5 Hash ของไฟล์"""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}", file=sys.stderr)
        return ""


def convert_csv(csv_file: Path, mask_dir: Path, output_file: Path = None,
                backup: bool = True) -> None:
    df = pd.read_csv(csv_file)

    if 'Filename' not in df.columns:
        print(f"Error: 'Filename' column not found in {csv_file}", file=sys.stderr)
        sys.exit(1)

    if 'file_hash' in df.columns:
        print(f"Note: '{csv_file}' already has a 'file_hash' column. Nothing to convert.")
        return

    # หาไฟล์ .npy ทั้งหมดใน roi_mask
    mask_paths = sorted(mask_dir.glob('*.npy'))
    if not mask_paths:
        print(f"Warning: no .npy files found in '{mask_dir}'.", file=sys.stderr)

    print(f"Calculating MD5 hashes for {len(mask_paths)} files... This might take a moment.")
    
    # คำนวณ Hash แทน mtime
    hash_map = {p.name: get_file_hash(p) for p in mask_paths}

    df['file_hash'] = df['Filename'].map(hash_map)

    missing = df['file_hash'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} row(s) in '{csv_file}' have a Filename not found in "
              f"'{mask_dir}' (file_hash left blank for those rows).")

    # เรียงคอลัมน์ให้เป็น Class, Filename, file_hash (เผื่อมี column อื่นแทรกอยู่ ก็ต่อท้ายไป)
    ordered_cols = [c for c in ['Class', 'Filename', 'file_hash'] if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    target_path = output_file if output_file else csv_file

    if target_path == csv_file and backup:
        backup_path = backup_csv_file(csv_file)
        if backup_path:
            print(f"Backup written to '{backup_path}'.")

    df.to_csv(target_path, index=False, encoding='utf-8')
    print(f"Successfully converted {len(df)} row(s) -> '{target_path}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a 2-column (Class, Filename) CSV into 3-column "
                    "(Class, Filename, file_hash) by calculating MD5 hashes from roi_mask/*.npy"
    )
    parser.add_argument('csv_file', help="Path to the existing CSV file")
    parser.add_argument('mask_dir', help="Path to the roi_mask folder containing .npy files")
    parser.add_argument('-o', '--output', default=None,
                        help="Path to write the converted CSV to (default: overwrite csv_file, with a .bak backup)")
    parser.add_argument('-nb', '--no-backup', action='store_true',
                        help="Skip writing a .bak backup when overwriting csv_file (ignored if --output is set)")

    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    mask_path = Path(args.mask_dir)
    output_path = Path(args.output) if args.output else None

    if not csv_path.is_file() or csv_path.suffix.lower() != '.csv':
        print(f"Error: CSV file not found at '{csv_path}' or wrong file format.", file=sys.stderr)
        sys.exit(1)

    if not mask_path.is_dir():
        print(f"Error: '{mask_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    convert_csv(csv_path, mask_path, output_file=output_path, backup=not args.no_backup)


if __name__ == '__main__':
    main()