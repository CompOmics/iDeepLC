import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ideeplc.ideeplc_core import main as run_ideeplc
import argparse
import sys
import os


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


def launch_gui():
    root = tk.Tk()
    root.title("iDeepLC Predictor")
    root.geometry("600x400")
    root.configure(bg="#f0f0f0")

    # Load and display the image
    img_url = "https://github.com/user-attachments/assets/86e9b793-39be-4f62-8119-5c6a333af487"
    img_path = "logo_temp.jpg"

    # Download if not exists (for local runs)
    if not os.path.exists(img_path):
        import requests
        with open(img_path, "wb") as f:
            f.write(requests.get(img_url).content)

    image = Image.open(img_path)
    image = image.resize((410,240))
    photo = ImageTk.PhotoImage(image)

    image_label = tk.Label(root, image=photo, bg="#f0f0f0")
    image_label.image = photo
    image_label.pack(pady=(10, 0))

    # Input file selection
    frame = tk.Frame(root, bg="#f0f0f0")
    frame.pack(pady=10)

    tk.Label(frame, text="Input CSV:", bg="#f0f0f0").grid(row=0, column=0, padx=5, sticky='e')
    input_entry = tk.Entry(frame, width=40)
    input_entry.grid(row=0, column=1, padx=5)
    tk.Button(frame, text="Browse", command=lambda: browse_file(input_entry)).grid(row=0, column=2, padx=5)

    # Options
    options_frame = tk.Frame(root, bg="#f0f0f0")
    options_frame.pack(pady=10)

    calibrate_var = tk.BooleanVar()
    finetune_var = tk.BooleanVar()

    tk.Checkbutton(options_frame, text="Calibrate", variable=calibrate_var, bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
    tk.Checkbutton(options_frame, text="Fine-tune", variable=finetune_var, bg="#f0f0f0").pack(side=tk.LEFT, padx=10)

    # Run button
    run_btn = tk.Button(root, text="Run Prediction", bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                        command=lambda: run_prediction(input_entry.get(), calibrate_var.get(), finetune_var.get()))
    run_btn.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
