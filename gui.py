import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ideeplc.ideeplc_core import main as run_ideeplc
import argparse
import os
import requests


PRIMARY_BG = "#1e1e2e"
ACCENT = "#2d2d46"
TEXT_COLOR = "#f5f5f5"
BUTTON_COLOR = "#313244"
BUTTON_HOVER = "#45475a"
FONT = ("Segoe UI", 11)


def run_prediction(input_path, calibrate, finetune):
    if not input_path:
        messagebox.showerror("Error", "Please select an input file.")
        return

    try:
        args = argparse.Namespace(
            input=input_path,
            calibrate=calibrate,
            finetune=finetune,
            save=True,
            log_level="INFO"
        )
        run_ideeplc(args)
        messagebox.showinfo("Success", "Prediction completed successfully!")
    except Exception as e:
        messagebox.showerror("Prediction Failed", str(e))


def browse_file(entry_field):
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, filepath)


def style_button(btn):
    btn.configure(bg=BUTTON_COLOR, fg=TEXT_COLOR, activebackground=BUTTON_HOVER,
                  relief="flat", font=FONT, cursor="hand2")
    btn.bind("<Enter>", lambda e: btn.config(bg=BUTTON_HOVER))
    btn.bind("<Leave>", lambda e: btn.config(bg=BUTTON_COLOR))


def launch_gui():
    root = tk.Tk()
    root.title("iDeepLC Predictor")
    root.geometry("700x550")
    root.configure(bg=PRIMARY_BG)

    # Load and display the image
    img_url = "https://github.com/user-attachments/assets/86e9b793-39be-4f62-8119-5c6a333af487"
    img_path = "logo_temp.jpg"
    if not os.path.exists(img_path):
        with open(img_path, "wb") as f:
            f.write(requests.get(img_url).content)

    image = Image.open(img_path)
    image = image.resize((450, 200), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=photo, bg=PRIMARY_BG)
    image_label.image = photo
    image_label.pack(pady=(20, 10))

    # Input frame
    frame = tk.Frame(root, bg=PRIMARY_BG)
    frame.pack(pady=10)

    tk.Label(frame, text="Input CSV:", bg=PRIMARY_BG, fg=TEXT_COLOR, font=FONT).grid(row=0, column=0, padx=10)
    input_entry = tk.Entry(frame, width=45, font=FONT, bg="#2e2e3f", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                           relief="flat")
    input_entry.grid(row=0, column=1, padx=10)
    browse_btn = tk.Button(frame, text="Browse", command=lambda: browse_file(input_entry))
    style_button(browse_btn)
    browse_btn.grid(row=0, column=2, padx=10)

    # Options frame
    options_frame = tk.Frame(root, bg=PRIMARY_BG)
    options_frame.pack(pady=15)

    calibrate_var = tk.BooleanVar()
    finetune_var = tk.BooleanVar()

    tk.Checkbutton(options_frame, text="Calibrate", variable=calibrate_var,
                   bg=PRIMARY_BG, fg=TEXT_COLOR, selectcolor=ACCENT, font=FONT, activebackground=PRIMARY_BG).pack(side=tk.LEFT, padx=20)
    tk.Checkbutton(options_frame, text="Fine-tune", variable=finetune_var,
                   bg=PRIMARY_BG, fg=TEXT_COLOR, selectcolor=ACCENT, font=FONT, activebackground=PRIMARY_BG).pack(side=tk.LEFT, padx=20)

    # Run button
    run_btn = tk.Button(root, text="Run Prediction",
                        command=lambda: run_prediction(input_entry.get(), calibrate_var.get(), finetune_var.get()))
    style_button(run_btn)
    run_btn.config(font=("Segoe UI", 12, "bold"), width=20)
    run_btn.pack(pady=30)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
