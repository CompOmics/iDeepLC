import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
import argparse
import io
from ideeplc.ideeplc_core import main as run_ideeplc

# Styling
PRIMARY_BG = "#1e1e2e"
ACCENT = "#2d2d46"
TEXT_COLOR = "#f5f5f5"
TOOLTIP_BG = "#2a2a3b"
TOOLTIP_TEXT = "#ffffff"
BUTTON_COLOR = "#313244"
BUTTON_HOVER = "#45475a"
FONT = ("Segoe UI", 11)


def create_tooltip(widget, text):
    tooltip = tk.Toplevel(widget)
    tooltip.withdraw()
    tooltip.overrideredirect(True)

    label = tk.Label(
        tooltip,
        text=text,
        background=TOOLTIP_BG,
        foreground=TOOLTIP_TEXT,
        relief="solid",
        borderwidth=1,
        font=("Segoe UI", 9),
        padx=6,
        pady=2,
    )
    label.pack()

    def enter(event):
        x = widget.winfo_rootx() + 20
        y = widget.winfo_rooty() + 20
        tooltip.geometry(f"+{x}+{y}")
        tooltip.deiconify()

    def leave(event):
        tooltip.withdraw()

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)


def load_icon_from_url(url, size=(18, 18)):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content)).resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(image)


def style_button(btn):
    btn.configure(
        bg=BUTTON_COLOR,
        fg=TEXT_COLOR,
        activebackground=BUTTON_HOVER,
        relief="flat",
        font=FONT,
        cursor="hand2",
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=BUTTON_HOVER))
    btn.bind("<Leave>", lambda e: btn.config(bg=BUTTON_COLOR))


def run_prediction(input_path, calibrate, finetune, log_widget):
    if not input_path:
        messagebox.showerror("Error", "Please select an input file.")
        return

    try:
        log_widget.insert(tk.END, "Running prediction...\n")
        log_widget.see(tk.END)
        args = argparse.Namespace(
            input=input_path,
            calibrate=calibrate,
            finetune=finetune,
            save=True,
            log_level="INFO",
        )
        run_ideeplc(args)
        log_widget.insert(tk.END, "✅ Prediction completed successfully.\n")
        log_widget.see(tk.END)
    except Exception as e:
        log_widget.insert(tk.END, f"❌ Prediction failed: {str(e)}\n")
        log_widget.see(tk.END)
        messagebox.showerror("Prediction Failed", str(e))


def browse_file(entry_field):
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, filepath)


def launch_gui():
    root = tk.Tk()
    root.title("iDeepLC Predictor")
    root.geometry("720x600")
    root.configure(bg=PRIMARY_BG)

    # Top banner image
    logo_url = "https://github.com/user-attachments/assets/86e9b793-39be-4f62-8119-5c6a333af487"
    logo_image = Image.open(io.BytesIO(requests.get(logo_url).content)).resize(
        (460, 200), Image.LANCZOS
    )
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(root, image=logo_photo, bg=PRIMARY_BG)
    logo_label.image = logo_photo
    logo_label.pack(pady=(20, 10))

    # Load icons (white)
    folder_icon = load_icon_from_url(
        "https://img.icons8.com/ios-filled/50/ffffff/folder-invoices.png"
    )
    rocket_icon = load_icon_from_url(
        "https://img.icons8.com/ios-filled/50/ffffff/rocket--v1.png"
    )

    # File input section
    input_frame = tk.Frame(root, bg=PRIMARY_BG)
    input_frame.pack(pady=10)

    tk.Label(
        input_frame, text="Input CSV:", font=FONT, bg=PRIMARY_BG, fg=TEXT_COLOR
    ).grid(row=0, column=0, padx=10)
    input_entry = tk.Entry(
        input_frame,
        width=45,
        font=FONT,
        bg="#2e2e3f",
        fg=TEXT_COLOR,
        insertbackground=TEXT_COLOR,
        relief="flat",
    )
    input_entry.grid(row=0, column=1, padx=10)

    browse_btn = tk.Button(
        input_frame, image=folder_icon, command=lambda: browse_file(input_entry)
    )
    browse_btn.image = folder_icon
    style_button(browse_btn)
    browse_btn.grid(row=0, column=2, padx=5)
    create_tooltip(browse_btn, "Browse for your input CSV file")

    # Options section
    options_frame = tk.Frame(root, bg=PRIMARY_BG)
    options_frame.pack(pady=10)

    calibrate_var = tk.BooleanVar()
    finetune_var = tk.BooleanVar()

    cal_cb = tk.Checkbutton(
        options_frame,
        text="Calibrate",
        variable=calibrate_var,
        bg=PRIMARY_BG,
        fg=TEXT_COLOR,
        font=FONT,
        selectcolor=ACCENT,
        activebackground=PRIMARY_BG,
    )
    cal_cb.pack(side=tk.LEFT, padx=20)
    create_tooltip(cal_cb, "Apply spline calibration")

    ft_cb = tk.Checkbutton(
        options_frame,
        text="Fine-tune",
        variable=finetune_var,
        bg=PRIMARY_BG,
        fg=TEXT_COLOR,
        font=FONT,
        selectcolor=ACCENT,
        activebackground=PRIMARY_BG,
    )
    ft_cb.pack(side=tk.LEFT, padx=20)
    create_tooltip(ft_cb, "Fine-tune the model with your data")

    # Run button
    run_btn = tk.Button(
        root,
        text=" Run Prediction",
        image=rocket_icon,
        compound="left",
        command=lambda: run_prediction(
            input_entry.get(), calibrate_var.get(), finetune_var.get(), log_output
        ),
    )
    run_btn.image = rocket_icon
    run_btn.config(font=("Segoe UI", 12, "bold"), width=180)
    style_button(run_btn)
    run_btn.pack(pady=20)
    create_tooltip(run_btn, "Start predicting retention times")

    # Log output area
    tk.Label(root, text="Log Output:", bg=PRIMARY_BG, fg=TEXT_COLOR, font=FONT).pack(
        pady=(10, 0)
    )
    log_output = tk.Text(
        root,
        height=8,
        width=80,
        bg="#2e2e3f",
        fg="#eeeeee",
        insertbackground="#eeeeee",
        font=("Courier New", 10),
        relief="flat",
        wrap="word",
    )
    log_output.pack(padx=20, pady=(5, 20))

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
