import tkinter as tk
from tkinter import filedialog, messagebox
import os
from config import DEFAULT_PATCH_SIZE, DEFAULT_MODEL_NAME


def launch_gui():
    def browse_image():
        filename = filedialog.askopenfilename(filetypes=[("GeoTIFF files", "*.tif *.tiff")])
        if filename:
            image_entry.delete(0, tk.END)
            image_entry.insert(0, filename)

    def browse_model():
        filename = filedialog.askopenfilename(filetypes=[("Joblib model", "*.joblib")])
        if filename:
            model_entry.delete(0, tk.END)
            model_entry.insert(0, os.path.splitext(os.path.basename(filename))[0])  # Strip path and .joblib

    def browse_output_dir():
        dirname = filedialog.askdirectory()
        if dirname:
            output_dir_entry.delete(0, tk.END)
            output_dir_entry.insert(0, dirname)

    def run_classification():
        image_path = image_entry.get()
        model_name = model_entry.get()
        patch_size = patch_size_entry.get()
        output_name = output_name_entry.get()
        output_dir = output_dir_entry.get()

        if not image_path or not os.path.isfile(image_path):
            messagebox.showerror("Error", "Please select a valid GeoTIFF image.")
            return
        if not model_name:
            messagebox.showerror("Error", "Please specify a model name.")
            return
        if not output_name:
            output_name = "landcover"

        # Store selections to global config-style dictionary (used by main)
        gui_inputs["image"] = image_path
        gui_inputs["model"] = model_name
        gui_inputs["patch_size"] = int(patch_size)
        gui_inputs["output"] = output_name
        gui_inputs["output_dir"] = output_dir or "."

        root.destroy()

    gui_inputs = {}

    root = tk.Tk()
    root.title("Landcover Classification")

    # Image input
    tk.Label(root, text="Input Image (GeoTIFF):").grid(row=0, column=0, sticky="e")
    image_entry = tk.Entry(root, width=50)
    image_entry.grid(row=0, column=1)
    tk.Button(root, text="Browse", command=browse_image).grid(row=0, column=2)

    # Model
    tk.Label(root, text="Model Name:").grid(row=1, column=0, sticky="e")
    model_entry = tk.Entry(root, width=50)
    model_entry.insert(0, DEFAULT_MODEL_NAME)
    model_entry.grid(row=1, column=1)
    tk.Button(root, text="Browse", command=browse_model).grid(row=1, column=2)

    # Patch size
    tk.Label(root, text="Patch Size:").grid(row=2, column=0, sticky="e")
    patch_size_entry = tk.Entry(root, width=10)
    patch_size_entry.insert(0, str(DEFAULT_PATCH_SIZE))
    patch_size_entry.grid(row=2, column=1, sticky="w")

    # Output name
    tk.Label(root, text="Output Shapefile Name:").grid(row=3, column=0, sticky="e")
    output_name_entry = tk.Entry(root, width=50)
    output_name_entry.insert(0, "landcover")
    output_name_entry.grid(row=3, column=1)

    # Output directory
    tk.Label(root, text="Output Directory:").grid(row=4, column=0, sticky="e")
    output_dir_entry = tk.Entry(root, width=50)
    output_dir_entry.grid(row=4, column=1)
    tk.Button(root, text="Browse", command=browse_output_dir).grid(row=4, column=2)

    # Run button
    tk.Button(root, text="Run", command=run_classification, bg="green", fg="white").grid(row=5, column=1, pady=10)

    root.mainloop()

    return gui_inputs
