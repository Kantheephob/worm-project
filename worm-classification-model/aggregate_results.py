import pandas as pd
from pathlib import Path
import argparse
import re
import config as c


def combine_results(root_folder):
    work_folder = root_folder / 'log'
    out_file = root_folder / ('all_' + root_folder.name + '_results.xlsx')
    
    if not work_folder.exists():
        print(f'Directory {root_folder} not found.')
        return
    
    found_file = False
    df_list = []
    
    for f in work_folder.rglob('summary.csv'):
        found_file = True
        print(f'Reading: {f}')
        
        try:
            df = pd.read_csv(f)
            df_list.append(df)
            
        except Exception as e:
            print(f'Error reading {f}: {e}')
    
    if not found_file or not df_list:
        print(f"No 'summary.csv file' found in {work_folder}")
        return
    
    # combine all data        
    df_master = pd.concat(df_list, ignore_index=True) 
    
    # helper function for handle sheet name limitation
    def clean_sheet_name(name):
        cleaned = ''.join(ch for ch in str(name) if ch.isalnum() or ch in ('_', ' ', '-'))  # isalnum คือ เช็คว่าเป็น alphabet or number
        return cleaned[:31].strip() or 'Sheet'  # excel sheet name ลองรับสูงสุด 31 ตัวอักษร
    
    # helper function for handle duplicate sheet name
    def get_unique_sheet_name(base_name, existing_sheets):
        sheet_name = clean_sheet_name(base_name)
        base = sheet_name
        counter = 1
        while sheet_name in existing_sheets:
            suffix = f'_{counter}'
            sheet_name = base[:31 - len(suffix)] + suffix
            counter += 1
        return sheet_name
    
    # write excel
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        # num class base
        if 'dataset_name' in df_master.columns:
            df_master['num_class_group'] = df_master['dataset_name'].astype(str).str.extract(r'(\d+)\s*class', flags=re.IGNORECASE)[0]
        
            for num_cls, group_df in df_master.groupby('num_class_group'):
                if pd.isna(num_cls):
                    continue
                sheet_name = get_unique_sheet_name(f"{num_cls}class", writer.sheets)
                group_df.drop(columns=['num_class_group'], errors='ignore').to_excel(writer, sheet_name=sheet_name, index=False)
                
            df_master = df_master.drop(columns=['num_class_group'], errors='ignore')
            
        # individual dataset base
        for ds_name, group_df in df_master.groupby('dataset_name'):
            sheet_name = clean_sheet_name(ds_name)
            
            # handle duplicate name
            base_name = sheet_name
            counter = 1
            while sheet_name in writer.sheets:
                suffix = f'_{counter}'
                sheet_name = base_name[:31 - len(suffix)] + suffix
                counter += 1
            
            group_df.drop(columns=['dataset_name'], errors='ignore').to_excel(writer, sheet_name=sheet_name, index=False)
            
        # model base
        for model_name, group_df in df_master.groupby('model'):
            sheet_name = clean_sheet_name(model_name)
            
            # handle duplicate name
            base_name = sheet_name
            counter = 1
            while sheet_name in writer.sheets:
                suffix = f'_{counter}'
                sheet_name = base_name[:31 - len(suffix)] + suffix
                counter += 1
            
            group_df.drop(columns=['model'], errors='ignore').to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f'\nSuccessfully saved Excel file to: {out_file}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, default=getattr(c, 'EXPERIMENT_NAME', None), help='Name of train experiment.')
    parser.add_argument('-d', '--dir', type=str, default=getattr(c, 'OUTPUT_DIR', None), help='Path to root directory that store experiment results.')
    args = parser.parse_args()
    
    root = Path(args.dir)
    experiment_name = args.experiment
        
    if not experiment_name:
        print('[Error] Experiment name is missing. Please provide it via argument or in config file.')
        return
    
    combine_results(root / experiment_name)
    
    
if __name__ == '__main__':
    main()