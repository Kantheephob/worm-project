import argparse
import pandas as pd
from pathlib import Path
import sys


def remove_row_from_file(csv_file: Path, text_file: Path, column: str = 'Filename',
                          dry_run: bool = False, backup: bool = True) -> None:
    df = pd.read_csv(csv_file)

    # check csv file
    if column not in df.columns:
        print(f"Error: '{column}' column not found in {csv_file}", file=sys.stderr)
        sys.exit(1)

    # read text file
    with open(text_file, mode='r', encoding='utf-8') as f:
        remove_list = [line.strip() for line in f if line.strip()]

    if not remove_list:
        print(f"Warning: '{text_file}' contains no filenames to remove. Nothing to do.")
        return

    remove_set = set(remove_list)
    requested = len(remove_set)

    mask = df[column].isin(remove_set)
    matched = int(mask.sum())

    if matched == 0:
        print(f"Warning: none of the {requested} filenames in '{text_file}' were found in '{csv_file}'.")
        return

    if matched < requested:
        missing = requested - matched
        print(f"Note: {missing} filename(s) from '{text_file}' were not found in '{csv_file}' (skipped).")

    if dry_run:
        print(f"[Dry run] Would remove {matched} row(s) from '{csv_file}'. No changes written.")
        return

    if backup:
        backup_path = csv_file.with_suffix(csv_file.suffix + '.bak') # .csv.bak
        df.to_csv(backup_path, index=False)
        print(f"Backup written to '{backup_path}'.")

    df = df[~mask]
    df.to_csv(csv_file, index=False)
    print(f"Successfully removed {matched} row(s) from '{csv_file}'.")


def main():
    parser = argparse.ArgumentParser(description="Remove rows from CSV based on a text file list.")

    parser.add_argument('csv_file', help="Path to CSV file")
    parser.add_argument('text_file', help="Path to TXT file containing filenames to remove")
    parser.add_argument('--column', '-c', default='Filename', # CSV column name ex -c 'Class'
                         help="Column to match filenames against (default: Filename)")
    parser.add_argument('--dry-run', '-dr', action='store_true', # store_true mean True if type frag False if not
                         help="Show how many rows would be removed without writing any changes")
    parser.add_argument('--no-backup', '-nb', action='store_true',
                         help="Skip writing a .bak backup of the CSV before overwriting it")

    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    text_path = Path(args.text_file)

    if not csv_path.is_file() or csv_path.suffix.lower() != '.csv':
        print(f"Error: CSV file not found at '{csv_path}' or wrong file format.", file=sys.stderr)
        sys.exit(1)

    if not text_path.is_file() or text_path.suffix.lower() != '.txt':
        print(f"Error: Text file not found at '{text_path}' or wrong file format.", file=sys.stderr)
        sys.exit(1)

    remove_row_from_file(csv_path, text_path, column=args.column,
                          dry_run=args.dry_run, backup=not args.no_backup)


if __name__ == '__main__':
    main()