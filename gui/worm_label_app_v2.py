import shutil
import threading
import concurrent.futures
from pathlib import Path
import tkinter as tk
from tkinter import Tk, Canvas, Button, Frame, filedialog, Label, Radiobutton, IntVar, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import hashlib

# --- Configuration & Styling ---
IMG_SIZE = 450
BG_COLOR = "#2C3E50"       
FG_COLOR = "#ECF0F1"       
BUTTON_COLOR = "#3498DB"   
HIGHLIGHT_COLOR = "#E74C3C" 
FONT_MAIN = ("Tahoma", 12)
FONT_HEADER = ("Tahoma", 12, "bold")

CACHE_MAX_SIZE = 40

# Global variables
csv_save_path = None
txt_force_path = None
grouped_files = []
group_index = 0
choices = []
folder = ""
folder_name = ""

# --- Cache & threading state ---
image_cache = {}
cache_order = []
cache_lock = threading.Lock()
is_loading = False

def backup_csv_file(csv_path: Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    backup_path = csv_path.with_suffix(csv_path.suffix + '.bak')
    try:
        shutil.copy2(csv_path, backup_path)
        return backup_path
    except Exception as e:
        raise RuntimeError(f"ไม่สามารถสำรองไฟล์ '{csv_path}' ก่อนบันทึกได้: {e}")

def get_file_hash_task(filepath: Path) -> dict:
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(1048576), b''):
                hasher.update(chunk)
        return {
            'Filename': filepath.name, 
            'mask_path': filepath, 
            'current_file_hash': hasher.hexdigest()
        }
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return {
            'Filename': filepath.name, 
            'mask_path': filepath, 
            'current_file_hash': ""
        }

def get_file_hash(filepath: Path) -> str:
    """เอาไว้ใช้ตอน save (ใช้แค่ไฟล์เดียว ไม่ต้อง multi-thread)"""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(1048576), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        return ""

# --- Logic Functions ---
def load_folders():
    global is_loading
    if is_loading:
        return

    folder_selected = filedialog.askdirectory(
        title="Please select folder that contain 'roi_mask' and 'roi_src' subfolder"
    )
    if not folder_selected:
        return

    is_loading = True
    load_button.config(state='disabled')
    label_mask.config(text="⏳ กำลังสแกนไฟล์กรุณารอสักครู่...")
    label_src.config(text="")
    canvas_mask.delete("all")
    canvas_src.delete("all")
    root.update_idletasks()

    threading.Thread(target=_load_folders_worker, args=(folder_selected,), daemon=True).start()

def _load_folders_worker(folder_selected):
    try:
        folder_path = Path(folder_selected)
        f_name = folder_path.name
        csv_path = next(iter(sorted(folder_path.glob('*.csv'))), None)

        txt_path = next(iter(folder_path.glob('*.txt')), None)
        relabel = set()
        if txt_path and txt_path.exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            relabel.add(Path(line.strip()).stem)
            except Exception as e:
                print(f"Error reading txt file: {e}")
                
        warning_msg = None
        mask_dir = folder_path / 'roi_mask'
        if mask_dir.exists():
            all_mask_paths = sorted(mask_dir.glob('*.npy'))
        else:
            warning_msg = "ไม่พบโฟลเดอร์ 'roi_mask' ในโฟลเดอร์ที่เลือก"
            all_mask_paths = []

        mask_data = []
        if all_mask_paths:
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                mask_data = list(executor.map(get_file_hash_task, all_mask_paths))
            
        mask_df = pd.DataFrame(mask_data)

        pending_mask_paths = []
        if csv_path and csv_path.exists():
            label_df = pd.read_csv(csv_path)

            if 'file_hash' not in label_df.columns:
                temp_df = mask_df[['Filename', 'current_file_hash']].copy()
                temp_df.rename(columns={'current_file_hash': 'file_hash'}, inplace=True)
                label_df = label_df.merge(temp_df, on='Filename', how='left')
                backup_csv_file(csv_path)
                label_df.to_csv(csv_path, index=False, encoding='utf-8')

            if 'Filename' in label_df.columns and 'file_hash' in label_df.columns:
                merged_df = mask_df.merge(label_df, on='Filename', how='left')
                is_processed = merged_df['current_file_hash'] == merged_df['file_hash']
                
                if relabel:
                    in_relabel_list = merged_df['Filename'].apply(lambda x: Path(x).stem).isin(relabel)
                    is_processed = is_processed & ~in_relabel_list

                pending_mask_paths = merged_df.loc[~is_processed, 'mask_path'].tolist()
            else:
                pending_mask_paths = mask_df['mask_path'].tolist()
        else:
            pending_mask_paths = mask_df['mask_path'].tolist()

        new_grouped_files = [(mask, folder_path / 'roi_src' / mask.name) for mask in pending_mask_paths]

        result = {
            'folder': folder_selected,
            'folder_name': f_name,
            'csv_path': csv_path,
            'txt_path': txt_path,
            'grouped_files': new_grouped_files,
            'warning_msg': warning_msg,
        }
    except Exception as e:
        result = {'error': str(e)}

    root.after(0, lambda: _on_folders_loaded(result))

def _on_folders_loaded(result):
    global grouped_files, group_index, choices, csv_save_path, txt_force_path, folder, folder_name, is_loading

    is_loading = False
    load_button.config(state='normal')

    if 'error' in result:
        messagebox.showerror("Error", f"โหลดโฟลเดอร์ไม่สำเร็จ: {result['error']}")
        return

    if result['warning_msg']:
        messagebox.showwarning("Warning", result['warning_msg'])

    folder = result['folder']
    folder_name = result['folder_name']
    csv_save_path = result['csv_path']
    txt_force_path = result.get('txt_path')
    grouped_files = result['grouped_files']

    group_index = 0
    choices = [None] * len(grouped_files)
    clear_image_cache()

    if grouped_files:
        display_images()
    else:
        messagebox.showinfo("Info", "ไม่พบรูปภาพใหม่ หรือรูปทั้งหมดถูก Label ครบแล้ว!")
        label_mask.config(text="No Data / Finished")
        label_src.config(text="")
        canvas_mask.delete("all")
        canvas_src.delete("all")

def process_npy_to_image(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        img_array = np.load(str(path))
        if img_array.dtype == bool:
            img_array = img_array.astype(np.uint8) * 255
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        return Image.fromarray(img_array.astype(np.uint8))
    except Exception as e:
        print(f"Error processing {str(path)}: {e}")
        return None

def create_overlay_image(src_img, mask_path):
    mask_path_obj = Path(mask_path)
    if not mask_path_obj.exists() or src_img is None:
        return src_img
    try:
        mask_array = np.load(str(mask_path_obj))
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0]

        overlay_color = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        overlay_color[mask_array > 0] = [52, 152, 219, 130]

        src_rgba = src_img.convert("RGBA")
        overlay_img = Image.fromarray(overlay_color)

        combined = Image.alpha_composite(src_rgba, overlay_img).convert("RGB")
        return combined
    except Exception as e:
        print(f"Error creating overlay for {mask_path}: {e}")
        return src_img

def resize_and_pad(img):
    if img is None:
        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
        return ImageTk.PhotoImage(img)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return ImageTk.PhotoImage(img)

def clear_image_cache():
    with cache_lock:
        image_cache.clear()
        cache_order.clear()

def _build_images_for_index(idx):
    mask_path, src_path = grouped_files[idx]
    raw_src_img = process_npy_to_image(src_path)
    overlayed_img = create_overlay_image(raw_src_img, mask_path)
    display_mask = resize_and_pad(overlayed_img)
    display_src = resize_and_pad(raw_src_img)
    return display_mask, display_src

def get_display_images(idx):
    with cache_lock:
        cached = image_cache.get(idx)
    if cached is not None:
        return cached

    imgs = _build_images_for_index(idx)
    _store_in_cache(idx, imgs)
    return imgs

def _store_in_cache(idx, imgs):
    with cache_lock:
        if idx not in image_cache:
            cache_order.append(idx)
        image_cache[idx] = imgs
        while len(cache_order) > CACHE_MAX_SIZE:
            oldest = cache_order.pop(0)
            image_cache.pop(oldest, None)

def prefetch_index(idx):
    if idx < 0 or idx >= len(grouped_files):
        return
    with cache_lock:
        if idx in image_cache:
            return

    def worker():
        try:
            imgs = _build_images_for_index(idx)
            root.after(0, lambda: _store_in_cache(idx, imgs))
        except Exception as e:
            print(f"Prefetch error for index {idx}: {e}")

    threading.Thread(target=worker, daemon=True).start()

def display_images():
    global group_index, grouped_files, choice_var
    if group_index < len(grouped_files):
        mask_path, src_path = grouped_files[group_index]
        display_mask, display_src = get_display_images(group_index)
        Filename = Path(mask_path).name

        label_mask.config(text=f"Overlay Mask ({group_index + 1}/{len(grouped_files)})")
        label_src.config(text=f"Source: {Filename}")

        if display_mask:
            canvas_mask.create_image(0, 0, anchor=tk.NW, image=display_mask)
            canvas_mask.image = display_mask
        if display_src:
            canvas_src.create_image(0, 0, anchor=tk.NW, image=display_src)
            canvas_src.image = display_src

        if choices[group_index] is not None:
            choice_var.set(choices[group_index])
        else:
            choice_var.set(-1)

        prev_button.config(state='normal' if group_index > 0 else 'disabled', bg=BUTTON_COLOR)
        next_button.config(state='normal' if group_index < len(grouped_files) - 1 else 'disabled', bg=BUTTON_COLOR)
        root.title(f"Worm Stage Labeler - {Filename}")

        prefetch_index(group_index + 1)
        prefetch_index(group_index - 1)
    else:
        label_mask.config(text="No Data / Finished")
        label_src.config(text="")
        canvas_mask.delete("all")
        canvas_src.delete("all")
        prev_button.config(state='disabled')
        next_button.config(state='disabled')

def prev_images():
    global group_index
    if group_index > 0:
        group_index -= 1
        display_images()

def next_images():
    global group_index, choices
    current_val = choice_var.get()
    if current_val in [0, 1, 2, 3, 4, 5, 6]:
        choices[group_index] = current_val

    if group_index < len(grouped_files) - 1:
        group_index += 1
        display_images()

def save_to_csv():
    global grouped_files, choices, group_index, csv_save_path, folder_name, folder, txt_force_path
    current_val = choice_var.get()
    if current_val in [0, 1, 2, 3, 4, 5, 6]:
        choices[group_index] = current_val

    if not csv_save_path:
        csv_name = f"ground_truth_worms_{folder_name}.csv"
        csv_save_path = Path(folder) / csv_name
    else:
        csv_save_path = Path(csv_save_path)

    new_data = []
    count = 0
    for idx, (mask_path, _) in enumerate(grouped_files):
        choice = choices[idx]
        if choice is not None and choice != -1:
            mask_p = Path(mask_path)
            new_data.append({
                'Class': choice,
                'Filename': mask_p.name,
                'file_hash': get_file_hash(mask_p),
            })
            count += 1

    if count == 0:
        messagebox.showinfo('Info', 'ไม่มีข้อมูลใหม่ให้บันทึก')
        return

    new_df = pd.DataFrame(new_data)

    if csv_save_path.exists():
        existing_df = pd.read_csv(csv_save_path)
        if 'Filename' in existing_df.columns:
            existing_df = existing_df[~existing_df['Filename'].isin(new_df['Filename'])]
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.sort_values(by='Filename', inplace=True)

    try:
        backup_path = backup_csv_file(csv_save_path)
    except RuntimeError as e:
        messagebox.showerror("Backup Failed", f"{e}\n\nยกเลิกการบันทึกเพื่อความปลอดภัยของข้อมูลเดิม")
        return

    try:
        final_df.to_csv(csv_save_path, index=False, encoding='utf-8')
        
        if txt_force_path and Path(txt_force_path).exists():
            try:
                with open(txt_force_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                labeled_stems = {Path(item['Filename']).stem for item in new_data}
                remaining_lines = [line for line in lines if Path(line).stem not in labeled_stems]
                with open(txt_force_path, 'w', encoding='utf-8') as f:
                    for line in remaining_lines:
                        f.write(line + '\n')
            except Exception as txt_e:
                print(f"Warning: เกิดข้อผิดพลาดตอนอัปเดตไฟล์ txt: {txt_e}")

    except Exception as e:
        msg = f"บันทึกไฟล์ล้มเหลว: {e}"
        if backup_path:
            msg += f"\n\nไฟล์เดิมยังปลอดภัยอยู่ที่: {backup_path}"
        messagebox.showerror("Save Failed", msg)
        return

    messagebox.showinfo("Saved", f"บันทึกข้อมูลใหม่ {count} รายการ เรียบร้อยแล้ว")

def on_radio_click():
    global group_index, choices
    current_val = choice_var.get()
    if current_val in [0, 1, 2, 3, 4, 5, 6]:
        choices[group_index] = current_val

def on_key_press(event):
    keysym = event.keysym.lower()
    char = event.char.lower() if event.char else ""

    current_val = choice_var.get()
    if current_val not in [0, 1, 2, 3, 4, 5, 6]:
        current_val = -1

    if keysym == 'right':
        new_val = (current_val + 1) % 7 if current_val != -1 else 0
        choice_var.set(new_val)
        choices[group_index] = new_val

    elif keysym == 'left':
        new_val = (current_val - 1) % 7 if current_val != -1 else 6
        choice_var.set(new_val)
        choices[group_index] = new_val

    elif keysym == 'down':
        down_map = {0: 4, 1: 5, 2: 6, 3: 6, 4: 4, 5: 5, 6: 6, -1: 0}
        new_val = down_map.get(current_val, current_val)
        choice_var.set(new_val)
        choices[group_index] = new_val

    elif keysym == 'up':
        up_map = {4: 0, 5: 1, 6: 3, 0: 0, 1: 1, 2: 2, 3: 3, -1: 0}
        new_val = up_map.get(current_val, current_val)
        choice_var.set(new_val)
        choices[group_index] = new_val

    elif keysym == 'space' or char in ['d', 'ก']:
        next_images()
        
    elif keysym == 'backspace' or char in ['a', 'ฟ']:
        prev_images()
        
    elif char in ['1', '2', '3', '4', '5', '6', '7'] or keysym in ['1', '2', '3', '4', '5', '6', '7']:
        val = char if char in ['1', '2', '3', '4', '5', '6', '7'] else keysym
        choice = int(val) - 1
        choice_var.set(choice)
        choices[group_index] = choice

    elif char in ['s', 'ห'] or keysym == 's':
        save_to_csv()

    elif char in ['w', 'ไ'] or keysym == 'w':
        load_folders()

# --- GUI Setup ---
root = tk.Tk()
root.title("Worm Label Tool V3 (Multi-Thread Hash + Text Relabel)")
root.configure(bg=BG_COLOR)
root.bind("<Key>", on_key_press)

try:
    root.state('zoomed')
except Exception:
    root.attributes('-zoomed', True)

main_frame = Frame(root, bg=BG_COLOR)
main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

content_frame = Frame(main_frame, bg=BG_COLOR)
content_frame.pack(expand=True)

display_frame = Frame(content_frame, bg=BG_COLOR)
display_frame.pack(pady=30)

control_frame = Frame(content_frame, bg=BG_COLOR)
control_frame.pack(pady=10)

choice_frame = Frame(content_frame, bg=BG_COLOR)
choice_frame.pack(pady=10)

label_mask = Label(display_frame, text="Overlay Mask", width=40, fg=FG_COLOR, bg=BG_COLOR, font=FONT_HEADER)
label_mask.grid(row=0, column=0)
label_src = Label(display_frame, text="Source", width=40, fg=FG_COLOR, bg=BG_COLOR, font=FONT_HEADER)
label_src.grid(row=0, column=1)

canvas_mask = Canvas(display_frame, width=IMG_SIZE, height=IMG_SIZE, bg="black", highlightthickness=2, highlightbackground="#34495E")
canvas_mask.grid(row=1, column=0, padx=15, pady=5)
canvas_src = Canvas(display_frame, width=IMG_SIZE, height=IMG_SIZE, bg="black", highlightthickness=2, highlightbackground="#34495E")
canvas_src.grid(row=1, column=1, padx=15, pady=5)

load_button = Button(control_frame, text="📂 Load Folders (w)", command=load_folders, font=FONT_MAIN, bg=BUTTON_COLOR, fg=FG_COLOR)
load_button.grid(row=0, column=0, padx=10)
prev_button = Button(control_frame, text="◀ Previous (Backspace / a)", command=prev_images, state='disabled', font=FONT_MAIN, bg=BUTTON_COLOR, fg=FG_COLOR)
prev_button.grid(row=0, column=1, padx=10)
next_button = Button(control_frame, text="Next (Spacebar / d) ▶", command=next_images, state='disabled', font=FONT_MAIN, bg=BUTTON_COLOR, fg=FG_COLOR)
next_button.grid(row=0, column=2, padx=10)
save_button = Button(control_frame, text="💾 Save CSV (S)", command=save_to_csv, font=FONT_MAIN, bg="#27AE60", fg=FG_COLOR)
save_button.grid(row=0, column=3, padx=10)

choice_var = IntVar(value=-1)

radio_options = [
    ("หนอนเด็ก (กด 1)", 0),
    ("หนอนพอดี (กด 2)", 1),
    ("หนอนแก่ (กด 3)", 2),
    ("หนอนระบุรุ่นไม่ได้ (กด 4)", 3),
    ("ไม่ใช่หนอน (กด 5)", 4),
    ("รูปแหว่ง/พลาด (กด 6)", 5),
    ("หนอนไม่มีคุณภาพ (กด 7)", 6)
]

row1_frame = Frame(choice_frame, bg=BG_COLOR)
row1_frame.pack(side=tk.TOP)

row2_frame = Frame(choice_frame, bg=BG_COLOR)
row2_frame.pack(side=tk.TOP)

for i, (text, val) in enumerate(radio_options):
    parent_frame = row1_frame if i < 4 else row2_frame
    rb = Radiobutton(
        parent_frame,
        text=text,
        variable=choice_var,
        value=val,
        command=on_radio_click,
        font=("Tahoma", 12, "bold"),
        bg="#34495E",
        fg=FG_COLOR,
        selectcolor="#0A0909",
        activebackground=BG_COLOR,
        activeforeground=HIGHLIGHT_COLOR,
        indicatoron=0,
        width=20,
        pady=8
    )
    rb.pack(side=tk.LEFT, padx=8, pady=5)

root.mainloop()