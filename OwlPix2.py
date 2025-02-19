import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np

# ---------------- Funkcje obrotu BOXÓW (YOLO) dla kątów 90, -90, 180 ----------------

def rotate_yolo_bbox_90_clockwise(class_id, x_center, y_center, w_box, h_box, orig_size):
    """
    Rotate +90 degrees (clockwise).
    Old width = orig_w, old height = orig_h.
    New width = orig_h, new height = orig_w.

    Formula (x, y) -> (x', y') = ( (orig_h-1) - y, x ), taking into account the corners of the boxes.
    """
    orig_w, orig_h = orig_size
    new_w, new_h = orig_h, orig_w  # obrócony obraz

    # 1) Zamiana YOLO -> piksele
    X = x_center * orig_w
    Y = y_center * orig_h
    W = w_box * orig_w
    H = h_box * orig_h
    # Współrzędne rogów w pikselach
    x_min = X - W / 2
    x_max = X + W / 2
    y_min = Y - H / 2
    y_max = Y + H / 2

    # 2) Obrót rogów (x,y) -> ((orig_h -1) - y, x)
    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    rotated = []
    for (cx, cy) in corners:
        new_x = (orig_h - 1) - cy
        new_y = cx
        rotated.append((new_x, new_y))

    # 3) Nowe min/max w pikselach
    rx = [p[0] for p in rotated]
    ry = [p[1] for p in rotated]
    min_x, max_x = min(rx), max(rx)
    min_y, max_y = min(ry), max(ry)

    # 4) Zamiana piksele -> YOLO
    new_x_center = (min_x + max_x) / 2
    new_y_center = (min_y + max_y) / 2
    new_w_box = (max_x - min_x)
    new_h_box = (max_y - min_y)

    # normalizacja
    new_x_norm = new_x_center / new_w
    new_y_norm = new_y_center / new_h
    new_w_norm = new_w_box / new_w
    new_h_norm = new_h_box / new_h

    return f"{class_id} {new_x_norm:.6f} {new_y_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}"


def rotate_yolo_bbox_90_counterclockwise(class_id, x_center, y_center, w_box, h_box, orig_size):
    """
    Rotate -90 degrees (counterclockwise).
    Formula (x, y) -> (y, (orig_w-1) - x).
    New width = orig_h, new height = orig_w.
    """
    orig_w, orig_h = orig_size
    new_w, new_h = orig_h, orig_w

    # YOLO -> piksele
    X = x_center * orig_w
    Y = y_center * orig_h
    W = w_box * orig_w
    H = h_box * orig_h
    x_min = X - W / 2
    x_max = X + W / 2
    y_min = Y - H / 2
    y_max = Y + H / 2

    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    rotated = []
    for (cx, cy) in corners:
        new_x = cy
        new_y = (orig_w - 1) - cx
        rotated.append((new_x, new_y))

    rx = [p[0] for p in rotated]
    ry = [p[1] for p in rotated]
    min_x, max_x = min(rx), max(rx)
    min_y, max_y = min(ry), max(ry)

    new_x_center = (min_x + max_x) / 2
    new_y_center = (min_y + max_y) / 2
    new_w_box = (max_x - min_x)
    new_h_box = (max_y - min_y)

    new_x_norm = new_x_center / new_w
    new_y_norm = new_y_center / new_h
    new_w_norm = new_w_box / new_w
    new_h_norm = new_h_box / new_h

    return f"{class_id} {new_x_norm:.6f} {new_y_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}"


def rotate_yolo_bbox_180(class_id, x_center, y_center, w_box, h_box, orig_size):
    """
    Rotate by 180 degrees. The image dimensions do not change (e.g. 640x480 -> 640x480).
    Formula (x, y) -> ((orig_w-1) - x, (orig_h-1) - y).
    """
    orig_w, orig_h = orig_size
    # YOLO -> piksele
    X = x_center * orig_w
    Y = y_center * orig_h
    W = w_box * orig_w
    H = h_box * orig_h
    x_min = X - W / 2
    x_max = X + W / 2
    y_min = Y - H / 2
    y_max = Y + H / 2

    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    rotated = []
    for (cx, cy) in corners:
        new_x = (orig_w - 1) - cx
        new_y = (orig_h - 1) - cy
        rotated.append((new_x, new_y))

    rx = [p[0] for p in rotated]
    ry = [p[1] for p in rotated]
    min_x, max_x = min(rx), max(rx)
    min_y, max_y = min(ry), max(ry)

    new_x_center = (min_x + max_x) / 2
    new_y_center = (min_y + max_y) / 2
    new_w_box = (max_x - min_x)
    new_h_box = (max_y - min_y)

    # normalizacja względem oryginalnych wymiarów
    new_x_norm = new_x_center / orig_w
    new_y_norm = new_y_center / orig_h
    new_w_norm = new_w_box / orig_w
    new_h_norm = new_h_box / orig_h

    return f"{class_id} {new_x_norm:.6f} {new_y_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}"

# ---------------- GUI: Funkcje wybierania folderów ----------------

def select_input_folder():
    folder = filedialog.askdirectory(title="Select input folder")
    input_entry.delete(0, tk.END)
    input_entry.insert(0, folder)

def select_output_folder():
    folder = filedialog.askdirectory(title="Select output folder")
    output_entry.delete(0, tk.END)
    output_entry.insert(0, folder)

# ---------------- Główna logika obrotu (wywoływana przez przyciski) ----------------

def start_rotation(angle):
    """
    Rotates by a given angle: -90, 90, 180.
    We do not show any preview window.
    """
    input_folder = input_entry.get()
    output_folder = output_entry.get()

    if not input_folder or not output_folder:
        messagebox.showerror("Error", "Select input and output folder")
        return

    if angle not in [-90, 90, 180]:
        messagebox.showerror("Error", "Angle outside allowed values ​​(-90, 90, 180)")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    files = os.listdir(input_folder)
    if not files:
        print("❌ ERROR: No files found in input folder!")
        return

    for filename in files:
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(input_folder, filename)
            print(f"File processing: {filename}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ ERROR: Could not load {image_path}")
                continue
            else:
                print(f"✅ Obraz wczytany: {image.shape}")  # (wys, szer, channels)

            # 1. Obracamy obraz
            rotated_image = None
            if angle == 90:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                new_size = (image.shape[0], image.shape[1])  # (orig_h, orig_w)
            elif angle == -90:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                new_size = (image.shape[0], image.shape[1])
            elif angle == 180:
                rotated_image = cv2.rotate(image, cv2.ROTATE_180)
                new_size = (image.shape[1], image.shape[0])  # wymiary się nie zmieniają

            # Zapisujemy obrócony obraz
            output_image_path = os.path.join(output_folder, filename)
            success = cv2.imwrite(output_image_path, rotated_image)
            if not success:
                print(f"❌ ERROR: Failed to save image {output_image_path}")
            else:
                print(f"✅ Image saved: {output_image_path}")

            # 2. Jeśli mamy plik z etykietami YOLO, wczytujemy i obracamy
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(input_folder, label_filename)
            if os.path.exists(label_path):
                print(f"Label processing: {label_filename}")
                with open(label_path, "r") as f:
                    labels = f.readlines()

                if not labels:
                    print(f"⚠️ WARNING: {label_filename} it's empty!")
                else:
                    new_labels = []
                    orig_w = image.shape[1]
                    orig_h = image.shape[0]
                    for label in labels:
                        parts = label.split()
                        if len(parts) < 5:
                            print(f"❌ Incorrect format: {label.strip()}")
                            continue
                        class_id = parts[0]
                        x = float(parts[1])
                        y = float(parts[2])
                        w_box = float(parts[3])
                        h_box = float(parts[4])

                        if angle == 90:
                            rotated_label = rotate_yolo_bbox_90_clockwise(
                                class_id, x, y, w_box, h_box, (orig_w, orig_h)
                            )
                        elif angle == -90:
                            rotated_label = rotate_yolo_bbox_90_counterclockwise(
                                class_id, x, y, w_box, h_box, (orig_w, orig_h)
                            )
                        elif angle == 180:
                            rotated_label = rotate_yolo_bbox_180(
                                class_id, x, y, w_box, h_box, (orig_w, orig_h)
                            )
                        else:
                            rotated_label = None

                        if rotated_label:
                            new_labels.append(rotated_label)
                            print(f"  Old label: {label.strip()} -> New: {rotated_label}")

                    # Zapis nowego pliku .txt
                    output_label_path = os.path.join(output_folder, label_filename)
                    with open(output_label_path, "w") as f:
                        f.write("\n".join(new_labels))
                    print(f"✅ Labels saved: {output_label_path}")

    messagebox.showinfo("Ready", f"Turn {angle} degrees finished!")

# ---------------- Konfiguracja GUI ----------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("OwlPix")
app.geometry("800x600")

try:
    app.iconbitmap(r"C:\Users\owlpix.ico")
except:
    pass

# Tło (opcjonalne)
try:
    background_image = Image.open(r"C:\Users\pixowl.png")
    background_photo = ImageTk.PhotoImage(background_image)
    background_label = tk.Label(app, image=background_photo)
    background_label.place(relwidth=1, relheight=1)
except:
    pass

font = ("Orbitron", 14)

# Pola tekstowe: Input / Output folder
input_entry = ctk.CTkEntry(app, width=300, placeholder_text="Input folder", font=font)
input_entry.place(x=250, y=330)
input_button = ctk.CTkButton(app, text="Select", command=select_input_folder,
                             fg_color="purple", font=font, corner_radius=0)
input_button.place(x=560, y=330)

output_entry = ctk.CTkEntry(app, width=300, placeholder_text="Output folder", font=font)
output_entry.place(x=250, y=380)
output_button = ctk.CTkButton(app, text="Select", command=select_output_folder,
                              fg_color="purple", font=font, corner_radius=0)
output_button.place(x=560, y=380)

# Trzy przyciski do obracania (z większym odstępem i corner_radius=0)
button_90 = ctk.CTkButton(app, text="Rotate 90°", command=lambda: start_rotation(90),
                          fg_color="red", font=font, corner_radius=0, width=120)
button_90.place(x=250, y=440)

button_minus_90 = ctk.CTkButton(app, text="Rotate -90°", command=lambda: start_rotation(-90),
                                fg_color="green", font=font, corner_radius=0, width=120)
button_minus_90.place(x=390, y=440)

button_180 = ctk.CTkButton(app, text="Rotate 180°", command=lambda: start_rotation(180),
                           fg_color="blue", font=font, corner_radius=0, width=120)
button_180.place(x=530, y=440)

app.mainloop()

