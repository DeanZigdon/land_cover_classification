# subset_raster.py
""" 
this helper file is intended to test and debug the 'evaluation mode' mechanism
its main goal is to shorten tests processing time espcially for basic local computing capabilities
this script cuts the input raster and the ground truth raster into compatible and computable sections to run the full evaluation process on them
it also verifies identical alignment in spatial properties (starts at X corner\center, CRS, patch size)
"""
    
import math
import os
import logging
import rasterio
from rasterio.windows import Window,from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import sys
import logging

PATCH_SIZE = 64              # must match config: DEFAULT_PATCH_SIZE
ANCHOR = "CENTER"            # crop area (NW,CENTER,SE etc)
FORCE_EPSG = 6991            # for israel type '6991' to force EPSG:6991, 'None' to skip forcing a specific EPSG
REFERENCE = "imagery"        # which raster to treat as grid reference: "imagery" or "ground_truth"
target_pixels = 30000        # requested output file pixel size

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(encoding='utf-8')


def _ensure_dir(path):
    """verify the directory for the given path exists"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def _reproject_to_match(src_path, ref_path, out_path):
    """
    reprojects src_path to match ref_path (CRS, transform, width, height, pixel size)
    writes out_path.
    """
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_profile = ref.profile.copy()

    with rasterio.open(src_path) as src:
        dst_profile.update({
            "driver": "GTiff",
            "BIGTIFF": "YES",
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "nodata": src.nodata})
        
        _ensure_dir(out_path)
        with rasterio.open(out_path, "w", **dst_profile) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest if b == 1 else Resampling.bilinear,)

def _force_epsg_if_needed(in_path, epsg_code, out_path):
    """
    reprojects raster to a specific EPSG. If EPSG is the same, just copy via window read/write.
    """
    with rasterio.open(in_path) as src:
        if src.crs and src.crs.to_epsg() == epsg_code:
            # already in target EPSG: write a copy (no op crop)
            profile = src.profile.copy()
            _ensure_dir(out_path)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(src.read())
            return out_path

        dst_crs = rasterio.crs.CRS.from_epsg(epsg_code)
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update({
            "BIGTIFF": "YES",
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height})
        
        _ensure_dir(out_path)
        with rasterio.open(out_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest if i == 1 else Resampling.bilinear)
        return out_path

def _crop_top_left(in_path, out_path, target_pixels, patch_size=PATCH_SIZE):
    """
    crops a square window from the 'ANCHOR' variable. side is forced to be a multiple of patch_size 
    to make sure subset tiles are fixed #-X-# (64x64 by default, but must comply with whats written in the GUI before each run) patches

    Returns (out_path, width, height, patch_cols, patch_rows).
    """
    side_raw = int(math.sqrt(max(1, target_pixels)))
    # force side to be >= patch_size and a multiple of patch_size
    side = max(patch_size, (side_raw // patch_size) * patch_size)
    if side % patch_size != 0:
        side = ((side // patch_size) + 1) * patch_size

    with rasterio.open(in_path) as src:
        # clamp side to raster bounds
        side = min(side, src.width, src.height)
        # again enforce multiple of patch_size after clamping
        side = (side // patch_size) * patch_size
        if side < patch_size:
            raise ValueError(f"Subset side ({side}) is smaller than a single patch ({patch_size}). "
                             f"Raster too small for requested subset.")

        window = Window(col_off=0, row_off=0, width=side, height=side)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update({
            "BIGTIFF": "YES",
            "width": side,
            "height": side,
            "transform": transform})

        _ensure_dir(out_path)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(src.read(window=window))

    patch_cols = side // patch_size
    patch_rows = side // patch_size
    return out_path, side, side, patch_cols, patch_rows

def make_aligned_pair(imagery_path,gt_path,out_imagery_path,out_gt_path,target_pixels,patch_size=PATCH_SIZE,reference=REFERENCE,force_epsg=FORCE_EPSG,

fail_on_mismatch=True # set False to only warn instead of raise
):
    """
    creates aligned, same window subsets for imagery and ground truth

    steps:
      1) (optional) Force both to EPSG if force_epsg is not 'None'
      2) reproject the non reference raster to match the reference raster GRID (CRS+transform+res)
      3) compute the intersection (overlap) between ref and moving on the aligned grid
      4) choose a square window inside the overlap:
           - side ~ sqrt(target_pixels), snapped down to a multiple of patch_size,
           - capped by the overlap dimensions
      5) crop both rasters using the SAME pixel window
      6) verify equality of grid/shape; verify ground_truth (GT) has valid labels
    """
    # open rasters to access metadata
    with rasterio.open(imagery_path) as imag, rasterio.open(gt_path) as gt:
        # the variables `ref` and `mov` are defined later in the function.
        # use `imag` and `gt` here to check nodata.
        if imag.nodata is None:
            ref_nodata_val = 255
        else:
            ref_nodata_val = imag.nodata

        if gt.nodata is None:
            gt_nodata_val = 0
        else:
            gt_nodata_val = gt.nodata

    def _fail_or_log(ok, msg, level="error"):
        if ok:
            return
        if level == "error" and fail_on_mismatch:
            raise ValueError(msg)
        (logging.error if level == "error" else logging.warning)(msg)

    #  step 1: optional CRS forcing
    tmp_imag = imagery_path
    tmp_gt = gt_path
    if force_epsg is not None:
        tmp_imag_forced = out_imagery_path.replace(".tif", f".epsg{force_epsg}.tif")
        tmp_gt_forced = out_gt_path.replace(".tif", f".epsg{force_epsg}.tif")
        tmp_imag = _force_epsg_if_needed(imagery_path, force_epsg, tmp_imag_forced)
        tmp_gt = _force_epsg_if_needed(gt_path, force_epsg, tmp_gt_forced)

    # step 2: choose reference / moving 
    ref_path = tmp_imag if reference == "imagery" else tmp_gt
    mov_path = tmp_gt if reference == "imagery" else tmp_imag

    # per-raster checks
    def _raster_checks(path, label, check_unique=False):
        with rasterio.open(path) as src:
            logging.info(f"[Check] {label}: {path}")
            _fail_or_log(src.crs is not None, f"{label}: CRS is undefined!", "error")
            logging.info(f"{label}: CRS = {src.crs}")
            res = src.res
            _fail_or_log(res[0] > 0 and res[1] > 0, f"{label}: Invalid resolution {res}", "error")
            logging.info(f"{label}: Resolution = {res}")
            b = src.bounds
            _fail_or_log(b.left < b.right and b.bottom < b.top, f"{label}: Invalid bounds {b}", "error")
            logging.info(f"{label}: Bounds = {b}")
            logging.info(f"{label}: Nodata = {src.nodata}")
            if check_unique:
                arr = src.read(1)
                mask = np.ones(arr.shape, dtype=bool)
                if src.nodata is not None:
                    mask &= (arr != src.nodata)
                if np.issubdtype(arr.dtype, np.floating):
                    mask &= ~np.isnan(arr)
                valid = arr[mask]
                logging.info(f"{label}: Valid label pixels = {valid.size}")
                if valid.size == 0:
                    _fail_or_log(False, f"{label}: No valid labeled pixels (all nodata/NaN).", "error")
                else:
                    uniq, cnts = np.unique(valid, return_counts=True)
                    logging.info(f"{label}: Label histogram = {dict(zip(uniq.tolist(), cnts.tolist()))}")

    _raster_checks(tmp_imag, "Imagery", check_unique=False)
    _raster_checks(tmp_gt, "Ground Truth", check_unique=True)

    # step 2b: overlap test before reprojection
    with rasterio.open(ref_path) as ref, rasterio.open(mov_path) as mov:
        rb, mb = ref.bounds, mov.bounds
        prelim_overlap = not (rb.right <= mb.left or rb.left >= mb.right or rb.top <= mb.bottom or rb.bottom >= mb.top)
        _fail_or_log(prelim_overlap, "No spatial overlap between imagery and ground truth (pre-reproject).", "error")

    # step 3: reproject moving to reference GRID
    reproj_moving_path = mov_path.replace(".tif", ".reproj_to_ref.tif")
    _reproject_to_match(mov_path, ref_path, reproj_moving_path)

    # verify grid alignment after reprojection
    with rasterio.open(ref_path) as ref, rasterio.open(reproj_moving_path) as mov:
        _fail_or_log(ref.crs == mov.crs, "Post-reproject: CRS still differs.", "error")
        _fail_or_log(ref.res == mov.res, f"Post-reproject: resolution differs (ref={ref.res}, mov={mov.res}).", "error")
        _fail_or_log(ref.transform == mov.transform, "Post-reproject: transforms differ.", "error")
        logging.info("Post-reproject: CRS/resolution/transform aligned.")

        #step 4: compute intersection window on the aligned grid 
        left = max(ref.bounds.left, mov.bounds.left)
        right = min(ref.bounds.right, mov.bounds.right)
        bottom = max(ref.bounds.bottom, mov.bounds.bottom)
        top = min(ref.bounds.top, mov.bounds.top)
        _fail_or_log(left < right and bottom < top, "No spatial overlap after reprojection.", "error")

        # window covering the full overlap in ref pixel coords
        overlap_win_f = from_bounds(left, bottom, right, top, transform=ref.transform)
        # clamp to dataset bounds to avoid invalid windows
        overlap_win_f = overlap_win_f.intersection(Window(0, 0, ref.width, ref.height))

        # Snap offsets up, sizes down to integers
        col0 = int(np.ceil(overlap_win_f.col_off))
        row0 = int(np.ceil(overlap_win_f.row_off))
        w_full = int(np.floor(overlap_win_f.width))
        h_full = int(np.floor(overlap_win_f.height))
        _fail_or_log(w_full > 0 and h_full > 0, "Overlap window is empty in pixel space.", "error")

        # desired square side from target_pixels, snapped down to multiple of patch_size and capped by the overlap window size
        desired_side = int(np.sqrt(max(1, int(target_pixels))))
        desired_side = max(patch_size, (desired_side // patch_size) * patch_size)
        side_cap_w = (w_full // patch_size) * patch_size
        side_cap_h = (h_full // patch_size) * patch_size
        side = min(desired_side, side_cap_w, side_cap_h)
        _fail_or_log(side >= patch_size, f"Overlap too small for one patch ({patch_size}px). Overlap {w_full}x{h_full}px.", "error")

        # final window anchored based on the ANCHOR setting
        if ANCHOR == "NW":  # editable
            col_off = col0
            row_off = row0
        elif ANCHOR == "CENTER":  # editable
            # find the central coordinates of the overlap area
            center_col = col0 + w_full // 2
            center_row = row0 + h_full // 2

            # calculate the top left offset for the centered window
            col_off = max(0, center_col - side // 2)
            row_off = max(0, center_row - side // 2)

        # verify the window fits within both datasets
        if col_off + side > ref.width:
            col_off = max(0, ref.width - side)
        if row_off + side > ref.height:
            row_off = max(0, ref.height - side)

        window = Window(col_off=col_off, row_off=row_off, width=side, height=side)
        ref_transform_sub = ref.window_transform(window)
             
# step 5: crop both rasters using same window 
        # determine outputs by role of reference
        out_ref_path = out_imagery_path if reference == "imagery" else out_gt_path
        out_mov_path = out_gt_path if reference == "imagery" else out_imagery_path

        # write reference subset
        _ensure_dir(out_ref_path)
        ref_profile = ref.profile.copy()
        ref_profile.update({"BIGTIFF": "YES", "width": side, "height": side, "transform": ref_transform_sub})
        with rasterio.open(out_ref_path, "w", **ref_profile) as dst:
            dst.write(ref.read(window=window))

        # write moving subset
        _ensure_dir(out_mov_path)
        mov_profile = mov.profile.copy()
        mov_profile.update({"BIGTIFF": "YES", "width": side, "height": side, "transform": ref_transform_sub})
        with rasterio.open(out_mov_path, "w", **mov_profile) as dst:
            dst.write(mov.read(window=window))

    # step 6: final verification + GT label sanity 
    with rasterio.open(out_imagery_path) as out_img, rasterio.open(out_gt_path) as out_gt:
        _fail_or_log(out_img.crs == out_gt.crs, "Post-crop: CRS mismatch.", "error")
        _fail_or_log(out_img.res == out_gt.res, "Post-crop: resolution mismatch.", "error")
        _fail_or_log(out_img.transform == out_gt.transform, "Post-crop: transform mismatch.", "error")
        _fail_or_log((out_img.width, out_img.height) == (out_gt.width, out_gt.height),
                     f"Post-crop: dimension mismatch (img={out_img.width}x{out_img.height}, gt={out_gt.width}x{out_gt.height}).",
                     "error")

        # count valid GT labels (exclude nodata/NaN) and log histogram
        gt_arr = out_gt.read(1)
        mask = np.ones(gt_arr.shape, dtype=bool)
        if out_gt.nodata is not None:
            mask &= (gt_arr != out_gt.nodata)
        if np.issubdtype(gt_arr.dtype, np.floating):
            mask &= ~np.isnan(gt_arr)
        valid = gt_arr[mask]
        _fail_or_log(valid.size > 0, "Ground Truth (output): no valid labeled pixels after crop.", "error")
        uniq, cnts = np.unique(valid, return_counts=True)
        logging.info(f"Ground Truth (output): valid label pixels = {valid.size}")
        logging.info(f"Ground Truth (output): label histogram = {dict(zip(uniq.tolist(), cnts.tolist()))}")

    # log patch grid info
    pcols = side // patch_size
    prows = side // patch_size
    logging.info("Aligned subsets created.")
    logging.info(f"     Output imagery:      {out_imagery_path}")
    logging.info(f"     Output ground truth: {out_gt_path}")
    logging.info(f"     Subset size (px):    {side} x {side}   |   patches: {pcols} x {prows} (patch_size={patch_size})")
    logging.info(f"     Total patches:       {pcols * prows}")

    # cleanup temp file (reprojected moving)
    try:
        os.remove(reproj_moving_path)
    except Exception:
        pass

# -------------------------------
# Simple GUI for subset_raster.py
# -------------------------------
def launch_subset_gui():
    """Launches the graphical user interface for the raster subsetting tool."""
    def browse_input_raster():
        """Opens a file dialog to select the input raster."""
        filename = filedialog.askopenfilename(filetypes=[("GeoTIFF files", "*.tif *.tiff")])
        if filename:
            input_raster_entry.delete(0, tk.END)
            input_raster_entry.insert(0, filename)

    def browse_input_gt():
        """Opens a file dialog to select the ground truth raster."""
        filename = filedialog.askopenfilename(filetypes=[("GeoTIFF files", "*.tif *.tiff")])
        if filename:
            input_gt_entry.delete(0, tk.END)
            input_gt_entry.insert(0, filename)

    def browse_output_raster():
        """Opens a save file dialog to specify the output raster path."""
        filename = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("GeoTIFF files", "*.tif")])
        if filename:
            output_raster_entry.delete(0, tk.END)
            output_raster_entry.insert(0, filename)

    def browse_output_gt():
        """Opens a save file dialog to specify the output ground truth path."""
        filename = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("GeoTIFF files", "*.tif")])
        if filename:
            output_gt_entry.delete(0, tk.END)
            output_gt_entry.insert(0, filename)

    def run_subset():
        """Collects GUI inputs and runs the raster subsetting process."""
        input_raster = input_raster_entry.get()
        input_gt = input_gt_entry.get()
        output_raster = output_raster_entry.get()
        output_gt = output_gt_entry.get()
        target_pixels = target_pixels_entry.get()

        if not input_raster or not os.path.isfile(input_raster):
            messagebox.showerror("Error", "Please select a valid input raster.")
            return
        if not input_gt or not os.path.isfile(input_gt):
            messagebox.showerror("Error", "Please select a valid ground truth raster.")
            return
        if not output_raster:
            messagebox.showerror("Error", "Please specify output raster file path.")
            return
        if not output_gt:
            messagebox.showerror("Error", "Please specify output ground truth file path.")
            return
        
        # config logging for GUI run
        log_file = os.path.join(os.path.dirname(__file__), "subset_raster.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="w"),
                logging.StreamHandler(sys.stdout),])
        
        try:
            logging.info("Starting subsetting from GUI...")
            make_aligned_pair(
                input_raster,
                input_gt,
                output_raster,
                output_gt,
                target_pixels=int(target_pixels))
            messagebox.showinfo("Success", f"Subset created successfully.\n\nLogs saved to:\n{log_file}")
            try:
                subprocess.Popen(["notepad.exe", log_file])
            except Exception as e:
                logging.warning(f"Could not auto-open log file: {e}")
        except Exception as e:
            logging.exception("❌ Cropping failed")
            messagebox.showerror("Error", f"An error occurred: {e}\nSee log file for details.")


    root = tk.Tk()
    root.title("Subset Raster")

    tk.Label(root, text="Input Raster:").grid(row=0, column=0, sticky="e")
    input_raster_entry = tk.Entry(root, width=50)
    input_raster_entry.grid(row=0, column=1)
    tk.Button(root, text="Browse", command=browse_input_raster).grid(row=0, column=2)

    tk.Label(root, text="Ground Truth Raster:").grid(row=1, column=0, sticky="e")
    input_gt_entry = tk.Entry(root, width=50)
    input_gt_entry.grid(row=1, column=1)
    tk.Button(root, text="Browse", command=browse_input_gt).grid(row=1, column=2)

    tk.Label(root, text="Output Raster:").grid(row=2, column=0, sticky="e")
    output_raster_entry = tk.Entry(root, width=50)
    output_raster_entry.grid(row=2, column=1)
    tk.Button(root, text="Browse", command=browse_output_raster).grid(row=2, column=2)

    tk.Label(root, text="Output Ground Truth:").grid(row=3, column=0, sticky="e")
    output_gt_entry = tk.Entry(root, width=50)
    output_gt_entry.grid(row=3, column=1)
    tk.Button(root, text="Browse", command=browse_output_gt).grid(row=3, column=2)

    tk.Label(root, text="Target Pixels:").grid(row=4, column=0, sticky="e")
    target_pixels_entry = tk.Entry(root, width=20)
    target_pixels_entry.insert(0, "30000")
    target_pixels_entry.grid(row=4, column=1, sticky="w")

    tk.Button(root, text="Run", command=run_subset, bg="blue", fg="white").grid(row=5, column=1, pady=10)
    root.mainloop()


# modify main() to allow GUI launch if no CLI args
if __name__ == "__main__":
    if len(sys.argv) == 1:
        launch_subset_gui()
    else:
        # config logging for CLI run
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(os.path.dirname(__file__), "subset_raster.log"), mode="w")])
        main()