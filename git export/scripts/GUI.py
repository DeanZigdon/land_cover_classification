# GUI
"""
GUI script for main project pipeline (excluding appendices)
"""
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
            model_entry.insert(0, os.path.splitext(os.path.basename(filename))[0])  # strip path and .joblib

    def browse_output_dir():
        dirname = filedialog.askdirectory()
        if dirname:
            output_dir_entry.delete(0, tk.END)
            output_dir_entry.insert(0, dirname)
            
    def browse_gt_mask():
        filename = filedialog.askopenfilename(filetypes=[("GeoTIFF files", "*.tif *.tiff")])
        if filename:
            gt_mask_entry.delete(0, tk.END)
            gt_mask_entry.insert(0, filename)
            
    def toggle_fields():
        if mode_var.get() == "predict":
            gt_mask_label.grid_remove()
            gt_mask_entry.grid_remove()
            gt_mask_browse_button.grid_remove()
            output_name_label.grid()
            output_name_entry.grid()
            
        elif mode_var.get() == "evaluate":
            gt_mask_label.grid(row=6, column=0, sticky="e")
            gt_mask_entry.grid(row=6, column=1)
            gt_mask_browse_button.grid(row=6, column=2)
            output_name_label.grid_remove()
            output_name_entry.grid_remove()
            
    def run_command():
        if mode_var.get() == "predict":
            run_classification()
        elif mode_var.get() == "evaluate":
            run_evaluation()

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

        gui_inputs["mode"] = "predict"
        gui_inputs["image"] = image_path
        gui_inputs["model"] = model_name
        gui_inputs["patch_size"] = int(patch_size)
        gui_inputs["output"] = output_name
        gui_inputs["output_dir"] = output_dir or "."

        root.destroy()
        
    def run_evaluation():
        image_path = image_entry.get()
        model_name = model_entry.get()
        patch_size = patch_size_entry.get()
        gt_mask_path = gt_mask_entry.get()
        output_dir = output_dir_entry.get()
        
        if not image_path or not os.path.isfile(image_path):
            messagebox.showerror("Error", "Please select a valid GeoTIFF image for prediction.")
            return
        if not model_name:
            messagebox.showerror("Error", "Please specify a model name.")
            return
        if not gt_mask_path or not os.path.isfile(gt_mask_path):
            messagebox.showerror("Error", "Please select a valid ground truth mask.")
            return
            
        gui_inputs["mode"] = "evaluate"
        gui_inputs["image"] = image_path
        gui_inputs["model"] = model_name
        gui_inputs["patch_size"] = int(patch_size)
        gui_inputs["gt_mask"] = gt_mask_path
        gui_inputs["output_dir"] = output_dir or "."
        root.destroy()

    gui_inputs = {}
    root = tk.Tk()
    root.title("Landcover Classification")

    # mode selection frame
    mode_frame = tk.Frame(root)
    mode_frame.grid(row=0, column=0, columnspan=3, pady=10)
    tk.Label(mode_frame, text="Select Mode:").pack(side=tk.LEFT)
    mode_var = tk.StringVar(value="predict")
    predict_radio = tk.Radiobutton(mode_frame, text="Prediction", variable=mode_var, value="predict", command=toggle_fields)
    predict_radio.pack(side=tk.LEFT, padx=10)
    evaluate_radio = tk.Radiobutton(mode_frame, text="Evaluation", variable=mode_var, value="evaluate", command=toggle_fields)
    evaluate_radio.pack(side=tk.LEFT)

    # image input
    tk.Label(root, text="Input Image (GeoTIFF):").grid(row=1, column=0, sticky="e")
    image_entry = tk.Entry(root, width=50)
    image_entry.grid(row=1, column=1)
    tk.Button(root, text="Browse", command=browse_image).grid(row=1, column=2)

    # model selection
    tk.Label(root, text="Model Name:").grid(row=2, column=0, sticky="e")
    model_entry = tk.Entry(root, width=50)
    model_entry.insert(0, DEFAULT_MODEL_NAME)
    model_entry.grid(row=2, column=1)
    tk.Button(root, text="Browse", command=browse_model).grid(row=2, column=2)

    # patch size
    tk.Label(root, text="Patch Size:").grid(row=3, column=0, sticky="e")
    patch_size_entry = tk.Entry(root, width=10)
    patch_size_entry.insert(0, str(DEFAULT_PATCH_SIZE))
    patch_size_entry.grid(row=3, column=1, sticky="w")

    # output name (in prediction mode)
    output_name_label = tk.Label(root, text="Output Shapefile Name:")
    output_name_label.grid(row=4, column=0, sticky="e")
    output_name_entry = tk.Entry(root, width=50)
    output_name_entry.insert(0, "landcover")
    output_name_entry.grid(row=4, column=1)

    # output directory
    tk.Label(root, text="Output Directory:").grid(row=5, column=0, sticky="e")
    output_dir_entry = tk.Entry(root, width=50)
    output_dir_entry.grid(row=5, column=1)
    tk.Button(root, text="Browse", command=browse_output_dir).grid(row=5, column=2)
    
    # ground truth mask (in evaluation mode) - initially hidden
    gt_mask_label = tk.Label(root, text="Ground Truth Mask:")
    gt_mask_entry = tk.Entry(root, width=50)
    gt_mask_browse_button = tk.Button(root, text="Browse", command=browse_gt_mask)

    # run button
    tk.Button(root, text="Run", command=run_command, bg="blue", fg="white").grid(row=7, column=1, pady=10)
    toggle_fields() # Initial call to set the default layout
    root.mainloop()

    return gui_inputs