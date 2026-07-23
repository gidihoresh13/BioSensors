"""
Minimal Tkinter GUI for hyperspectral reporter detection.

Wraps the module-04 detection engine (hsi_detect: HyperspectralImage +
Spectrum + HierarchicalKMeansUnmixer) in a desktop UI with two tabs:

TAB 1 - "Detection"
  1. Open HSI cube        -> loads an ENVI .hdr (+raw) and shows an RGB render
  2. Open reference        -> loads a 2xN .npy/.csv reference spectrum
  3. Run detection         -> fit/classify on a background thread (UI stays live)
  4. Threshold slider      -> re-applies threshold on the cached score map (no re-fit)
  5. Click a pixel         -> plots that pixel's spectrum vs. the reference
  6. Export                -> saves score TIFF + heatmap PNG + params JSON

TAB 2 - "Spectrum sampler"  (independent of the detection cube)
  Builds a difference spectrum from three polygons on one image:
  1. Open image (.hdr)     -> loads a SEPARATE ENVI hyperspectral cube
  2. Zoom                   -> mouse wheel zooms on the image (cursor-centered);
                              nav toolbar adds pan / zoom-rectangle / Home reset
  3. Sample P1 (normalizer) -> its mean reflectance's min & max define a Spectronon-
                              style min-max normalization of the WHOLE image:
                              Nx = (Rx - min(R1)) / (max(R1) - min(R1))
  4. Sample P2 and P3      -> sampled after normalization (each normalized by P1's
                              min/max; equivalent to averaging then rescaling)
  5. Subtract              -> N2 - N3  (or N3 - N2; direction is selectable)
  6. Export (CSV)          -> writes wavelength + the subtracted spectrum

This folder is self-contained: it ships its own copy of the `hsi_detect`
package (in `./hsi_detect`), so it does not depend on any other module.

Run:  py -3.11 gui_detect.py       (or double-click run_gui.bat)
"""
import os
import sys
import json
import queue
import threading
from datetime import date

# hsi_detect is bundled in this folder; make sure it is importable regardless
# of the current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    import numpy as np
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.widgets import PolygonSelector
    from matplotlib.path import Path as MplPath
    from PIL import Image
    from hsi_detect.image import HyperspectralImage
    from hsi_detect.spectrum import Spectrum
    from hsi_detect.classifier import HierarchicalKMeansUnmixer
except ModuleNotFoundError as e:
    raise SystemExit(
        f"\nMissing dependency: {e.name!r}.\n"
        "This GUI requires Python 3.11 with the packages in requirements.txt.\n"
        "  - Install:  py -3.11 -m pip install -r requirements.txt\n"
        "  - Run:      py -3.11 gui_detect.py   (or use run_gui.bat)\n"
    )


class DetectorGUI:
    def __init__(self, root):
        self.root = root
        root.title("bioHSI - reporter detection")
        root.geometry("1180x760")

        # ---- detection-tab state ----
        self.hsi = None            # HyperspectralImage (possibly downsampled)
        self.ref_path = None       # path to reference spectrum file
        self.ref_disp = None       # Spectrum for display (interp to cube if loaded)
        self.raw_scores = None     # 2D float score map (unthresholded)

        # ---- spectrum-sampler-tab state (independent of the detection cube) ----
        self.samp_hsi = None       # separate HyperspectralImage for sampling
        self.samp_selector = None  # matplotlib PolygonSelector while active
        self.samp_active = False   # True while in polygon-drawing mode
        self.samp_wl = None        # wavelengths (cube band centers)
        # raw mean reflectance per polygon slot; P1 = normalizer, P2/P3 subtracted
        self.samp_slots = {"P1": None, "P2": None, "P3": None}
        self.samp_slot_n = {"P1": 0, "P2": 0, "P3": 0}   # pixel counts per slot
        self.current_mean = None   # raw mean reflectance of the last-drawn polygon
        self.current_n = 0

        self.q = queue.Queue()     # worker -> UI messages

        self._build_ui()
        self.root.after(100, self._poll)

    # ================= UI construction =================
    def _build_ui(self):
        # shared status bar at the very bottom
        self.status = tk.StringVar(value="Detection: open a cube.  Sampler: open an image to build a reference spectrum.")
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN,
                  anchor=tk.W, padding=4).pack(side=tk.BOTTOM, fill=tk.X)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        det_tab = ttk.Frame(self.nb)
        samp_tab = ttk.Frame(self.nb)
        self.nb.add(det_tab, text="Detection")
        self.nb.add(samp_tab, text="Spectrum sampler")

        self._build_detection_tab(det_tab)
        self._build_sampler_tab(samp_tab)

    def _build_detection_tab(self, parent):
        bar = ttk.Frame(parent, padding=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Open cube (.hdr)", command=self.open_cube).pack(side=tk.LEFT)
        ttk.Button(bar, text="Open reference", command=self.open_reference).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(bar, text="Downsample:").pack(side=tk.LEFT, padx=(12, 2))
        self.v_down = tk.IntVar(value=2)
        ttk.Spinbox(bar, from_=1, to=8, width=3, textvariable=self.v_down).pack(side=tk.LEFT)

        self.run_btn = ttk.Button(bar, text="Run detection", command=self.run_detection)
        self.run_btn.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(bar, text="Export", command=self.export).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(bar, text="Threshold:").pack(side=tk.LEFT, padx=(12, 2))
        self.v_thr = tk.DoubleVar(value=0.0)
        self.thr_scale = ttk.Scale(bar, from_=0.0, to=0.3, variable=self.v_thr,
                                   command=lambda _=None: self._redraw_scores(), length=160)
        self.thr_scale.pack(side=tk.LEFT)

        # params row
        prow = ttk.Frame(parent, padding=(6, 0))
        prow.pack(side=tk.TOP, fill=tk.X)
        self.v_dims = tk.IntVar(value=3)
        self.v_nclust = tk.IntVar(value=500)
        self.v_filter = tk.DoubleVar(value=0.9)
        self.v_dist = tk.DoubleVar(value=0.005)
        for lbl, var, w in [("reduced_dims", self.v_dims, 4),
                            ("n_init_clusters", self.v_nclust, 6),
                            ("filter_threshold", self.v_filter, 5),
                            ("distance_threshold", self.v_dist, 6)]:
            ttk.Label(prow, text=lbl).pack(side=tk.LEFT, padx=(8, 2))
            ttk.Entry(prow, textvariable=var, width=w).pack(side=tk.LEFT)

        # figure: RGB | score | clicked spectrum
        self.fig = Figure(figsize=(11, 5.2), constrained_layout=True)
        self.ax_rgb = self.fig.add_subplot(1, 3, 1)
        self.ax_score = self.fig.add_subplot(1, 3, 2)
        self.ax_spec = self.fig.add_subplot(1, 3, 3)
        for a, t in [(self.ax_rgb, "RGB"), (self.ax_score, "detection score"),
                     (self.ax_spec, "pixel spectrum")]:
            a.set_title(t)
        self.ax_rgb.axis("off")
        self.ax_score.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def _build_sampler_tab(self, parent):
        bar = ttk.Frame(parent, padding=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        self.samp_open_btn = ttk.Button(bar, text="Open image (.hdr)",
                                        command=self.open_sample_image)
        self.samp_open_btn.pack(side=tk.LEFT)

        self.samp_poly_btn = ttk.Button(bar, text="Sample polygon",
                                        command=self.toggle_polygon)
        self.samp_poly_btn.pack(side=tk.LEFT, padx=(12, 0))

        # which slot the next completed polygon fills. P1 = normalizer.
        ttk.Label(bar, text="  store into:").pack(side=tk.LEFT, padx=(12, 2))
        self.samp_target = tk.StringVar(value="P1")
        for val, txt in (("P1", "P1 (normalizer)"), ("P2", "P2"), ("P3", "P3")):
            ttk.Radiobutton(bar, text=txt, value=val,
                            variable=self.samp_target).pack(side=tk.LEFT)

        # second row: subtraction direction, export, clear
        bar2 = ttk.Frame(parent, padding=(6, 0))
        bar2.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar2, text="Subtract:").pack(side=tk.LEFT)
        self.samp_dir = tk.StringVar(value="P2 - P3")
        dir_cb = ttk.Combobox(bar2, textvariable=self.samp_dir, width=9, state="readonly",
                              values=["P2 - P3", "P3 - P2"])
        dir_cb.pack(side=tk.LEFT, padx=(4, 0))
        dir_cb.bind("<<ComboboxSelected>>", lambda _=None: self._update_diff_plot())
        ttk.Button(bar2, text="Export difference (CSV)...",
                   command=self.export_difference).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(bar2, text="Clear P1/P2/P3",
                   command=self.clear_slots).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(bar2, text="Save current mean (.npy)...",
                   command=self.save_spectrum).pack(side=tk.LEFT, padx=(6, 0))

        # figure: RGB (draw here) | normalized means + difference
        self.samp_fig = Figure(figsize=(11, 5.2), constrained_layout=True)
        self.samp_ax_rgb = self.samp_fig.add_subplot(1, 2, 1)
        self.samp_ax_spec = self.samp_fig.add_subplot(1, 2, 2)
        self.samp_ax_rgb.set_title("image (open one to begin)")
        self.samp_ax_rgb.axis("off")
        self.samp_ax_spec.set_title("normalized means + difference")
        self.samp_ax_spec.set_xlabel("wavelength (nm)")
        self.samp_ax_spec.set_ylabel("normalized reflectance")

        self.samp_canvas = FigureCanvasTkAgg(self.samp_fig, master=parent)

        # matplotlib navigation toolbar: pan, zoom-to-rectangle, Home = reset view.
        # It coexists with the PolygonSelector (drawing works whenever no pan/zoom
        # mode is active); the selector auto-ignores clicks while a nav mode is on.
        tb_frame = ttk.Frame(parent)
        tb_frame.pack(side=tk.TOP, fill=tk.X)
        self.samp_toolbar = NavigationToolbar2Tk(self.samp_canvas, tb_frame,
                                                 pack_toolbar=True)
        self.samp_toolbar.update()

        self.samp_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # mouse-wheel zoom on the image, centered on the cursor (no mode toggle)
        self.samp_canvas.mpl_connect("scroll_event", self._on_scroll)

        # full-image extent, captured when an image is drawn, used to cap zoom-out
        self.samp_full_xlim = None
        self.samp_full_ylim = None

    # ================= shared helpers =================
    def _set_status(self, msg):
        self.status.set(msg)

    def _busy(self, on):
        """Busy state for the detection tab."""
        self.run_btn.configure(state=(tk.DISABLED if on else tk.NORMAL))
        self.root.configure(cursor=("watch" if on else ""))

    def _samp_busy(self, on):
        """Busy state for the sampler tab."""
        self.samp_open_btn.configure(state=(tk.DISABLED if on else tk.NORMAL))
        self.root.configure(cursor=("watch" if on else ""))

    def _poll(self):
        """Drain worker messages on the Tk main thread."""
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    self._set_status(payload)
                elif kind == "cube_done":
                    self.hsi = payload
                    self._draw_rgb()
                    self._reinterp_reference()
                    self._busy(False)
                    self._set_status(f"Cube loaded: {self.hsi.image.shape}. Load a reference, then Run.")
                elif kind == "scores_done":
                    self.raw_scores = payload
                    mx = float(np.nanmax(self.raw_scores)) if self.raw_scores.size else 0.3
                    self.thr_scale.configure(to=max(mx, 1e-3))
                    self._redraw_scores()
                    self._busy(False)
                    self._set_status(f"Detection done. score max={mx:.3f}. Adjust threshold or Export.")
                elif kind == "samp_cube_done":
                    self.samp_hsi = payload
                    self._deactivate_polygon()       # drop any stale selector
                    self.samp_slots = {"P1": None, "P2": None, "P3": None}
                    self.samp_slot_n = {"P1": 0, "P2": 0, "P3": 0}
                    self.current_mean = None
                    self.samp_wl = np.asarray(self.samp_hsi.centers, dtype=float)
                    self._draw_samp_rgb()
                    self._update_diff_plot()
                    self._samp_busy(False)
                    self._set_status(f"Image loaded: {self.samp_hsi.image.shape}. "
                                     "Select a slot (P1/P2/P3), click 'Sample polygon' and draw.")
                elif kind == "error":
                    self._busy(False)
                    messagebox.showerror("Error", payload)
                    self._set_status("Error: " + payload.splitlines()[-1])
                elif kind == "samp_error":
                    self._samp_busy(False)
                    messagebox.showerror("Error", payload)
                    self._set_status("Error: " + payload.splitlines()[-1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ================= detection tab =================
    def open_cube(self):
        path = filedialog.askopenfilename(title="Select ENVI header",
                                          filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")])
        if not path:
            return
        d = max(1, int(self.v_down.get()))
        self._busy(True)
        self._set_status("Loading cube (this can take a few seconds)...")

        def work():
            try:
                hsi = HyperspectralImage(path)
                if d > 1:
                    hsi.image = hsi.image[::d, ::d, :]
                    hsi.rgb = None
                hsi.make_rgb()
                self.q.put(("cube_done", hsi))
            except Exception:
                import traceback
                self.q.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def open_reference(self):
        path = filedialog.askopenfilename(title="Select reference spectrum",
                                          filetypes=[("Spectrum", "*.npy *.csv"), ("All files", "*.*")])
        if not path:
            return
        self.ref_path = path
        self._reinterp_reference()
        self._set_status(f"Reference loaded: {os.path.basename(path)}")

    def _reinterp_reference(self):
        if not self.ref_path:
            return
        try:
            spec = Spectrum(file_path=self.ref_path)
            if self.hsi is not None:
                spec.interpolate_spectrum(self.hsi.centers)
            self.ref_disp = spec
            self.ax_spec.clear()
            self.ax_spec.set_title("pixel spectrum")
            self.ax_spec.plot(spec.wavelengths, spec.intensities, "k--", lw=1, label="reference")
            self.ax_spec.legend(fontsize=7)
            self.canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("Reference error", str(e))

    def run_detection(self):
        if self.hsi is None:
            messagebox.showwarning("No cube", "Open an HSI cube first.")
            return
        if not self.ref_path:
            messagebox.showwarning("No reference", "Open a reference spectrum first.")
            return
        params = dict(reduced_dims=int(self.v_dims.get()),
                      n_init_clusters=int(self.v_nclust.get()),
                      filter_threshold=float(self.v_filter.get()),
                      distance_threshold=float(self.v_dist.get()))
        self._busy(True)
        self._set_status("Running detection on a background thread...")

        def work():
            try:
                ref = Spectrum(file_path=self.ref_path)
                ref.interpolate_spectrum(self.hsi.centers)
                model = HierarchicalKMeansUnmixer(**params)
                model.fit(self.hsi, ref)
                scores = model.classify(ref, threshold=None)  # raw; threshold applied in UI
                self.q.put(("scores_done", np.asarray(scores)))
            except Exception:
                import traceback
                self.q.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _draw_rgb(self):
        self.ax_rgb.clear()
        self.ax_rgb.set_title("RGB")
        self.ax_rgb.axis("off")
        self.ax_rgb.imshow(self.hsi.rgb.astype(np.uint8))
        self.canvas.draw_idle()

    def _redraw_scores(self):
        if self.raw_scores is None:
            return
        thr = float(self.v_thr.get())
        disp = np.where(self.raw_scores >= thr, self.raw_scores, 0.0)
        self.ax_score.clear()
        self.ax_score.set_title(f"detection score (thr={thr:.3f})")
        self.ax_score.axis("off")
        self.ax_score.imshow(disp, cmap="magma")
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes not in (self.ax_rgb, self.ax_score) or self.hsi is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        h, w = self.hsi.image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        self.ax_spec.clear()
        self.ax_spec.set_title(f"pixel spectrum ({x},{y})")
        self.ax_spec.plot(self.hsi.centers, self.hsi.image[y, x, :], lw=1, label="pixel")
        if self.ref_disp is not None:
            r = self.ref_disp.intensities
            r = r / (np.nanmax(np.abs(r)) or 1) * np.nanmax(self.hsi.image[y, x, :])
            self.ax_spec.plot(self.ref_disp.wavelengths, r, "k--", lw=1, label="reference (scaled)")
        self.ax_spec.legend(fontsize=7)
        self.canvas.draw_idle()

    def export(self):
        if self.raw_scores is None:
            messagebox.showwarning("Nothing to export", "Run detection first.")
            return
        outdir = filedialog.askdirectory(title="Select output directory")
        if not outdir:
            return
        stamp = date.today().strftime("%d%b%Y")
        thr = float(self.v_thr.get())
        disp = np.where(self.raw_scores >= thr, self.raw_scores, 0.0)
        # raw score as float TIFF
        Image.fromarray(self.raw_scores.astype(np.float32)).save(os.path.join(outdir, f"{stamp}_score.tiff"))
        # thresholded heatmap PNG
        self.fig.savefig(os.path.join(outdir, f"{stamp}_detection_view.png"), dpi=150)
        # params
        params = dict(reference=os.path.basename(self.ref_path or ""),
                      downsample=int(self.v_down.get()),
                      reduced_dims=int(self.v_dims.get()),
                      n_init_clusters=int(self.v_nclust.get()),
                      filter_threshold=float(self.v_filter.get()),
                      distance_threshold=float(self.v_dist.get()),
                      display_threshold=thr,
                      score_max=float(np.nanmax(self.raw_scores)),
                      detected_fraction=float(np.mean(disp > 0)))
        with open(os.path.join(outdir, f"{stamp}_params.json"), "w") as f:
            json.dump(params, f, indent=2)
        self._set_status(f"Exported to {outdir}")

    # ================= spectrum sampler tab =================
    def open_sample_image(self):
        """Load a SEPARATE hyperspectral cube to sample a reference spectrum from."""
        path = filedialog.askopenfilename(
            title="Select ENVI header of the image to sample",
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")])
        if not path:
            return
        self._samp_busy(True)
        self._set_status("Loading image to sample (this can take a few seconds)...")

        def work():
            try:
                # full resolution: the sampler needs precise region boundaries,
                # and striding here would not save the (already-done) full load
                hsi = HyperspectralImage(path)
                hsi.make_rgb()
                self.q.put(("samp_cube_done", hsi))
            except Exception:
                import traceback
                self.q.put(("samp_error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _draw_samp_rgb(self):
        self.samp_ax_rgb.clear()
        self.samp_ax_rgb.set_title("image")
        self.samp_ax_rgb.axis("off")
        self.samp_ax_rgb.imshow(self.samp_hsi.rgb.astype(np.uint8))
        # remember the full-image view so scroll-out can't zoom past it, and so
        # the toolbar Home button resets here
        self.samp_full_xlim = self.samp_ax_rgb.get_xlim()
        self.samp_full_ylim = self.samp_ax_rgb.get_ylim()
        self.samp_canvas.draw_idle()

    def _on_scroll(self, event):
        """Wheel up = zoom in, wheel down = zoom out, centered on the cursor."""
        ax = self.samp_ax_rgb
        if event.inaxes is not ax or self.samp_hsi is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        base = 1.3
        scale = (1.0 / base) if event.button == "up" else base
        x, y = event.xdata, event.ydata
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        new_xlim = (x - (x - xlim[0]) * scale, x + (xlim[1] - x) * scale)
        new_ylim = (y - (y - ylim[0]) * scale, y + (ylim[1] - y) * scale)
        # cap zoom-out at the full image extent
        fx, fy = self.samp_full_xlim, self.samp_full_ylim
        if fx and fy and abs(new_xlim[1] - new_xlim[0]) >= abs(fx[1] - fx[0]):
            new_xlim, new_ylim = fx, fy
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.samp_canvas.draw_idle()

    def toggle_polygon(self):
        """Enter/leave polygon-drawing mode on the sampler image."""
        if self.samp_hsi is None:
            messagebox.showwarning("No image", "Open an image to sample first.")
            return
        if self.samp_active:
            self._deactivate_polygon()
            self._set_status("Polygon sampling stopped.")
            return
        # onselect fires when the polygon is closed AND whenever a vertex is
        # dragged, so the previewed mean spectrum updates live.
        self.samp_selector = PolygonSelector(
            self.samp_ax_rgb, self._on_polygon, useblit=True,
            props=dict(color="cyan", linewidth=1.5, alpha=0.8))
        self.samp_active = True
        self.samp_poly_btn.configure(text="Stop sampling")
        self._set_status("Polygon mode: click on the image to add vertices; "
                         "click the first vertex to close it. Esc restarts the shape.")
        self.samp_canvas.draw_idle()

    def _deactivate_polygon(self):
        if self.samp_selector is not None:
            try:
                self.samp_selector.disconnect_events()
            except Exception:
                pass
            self.samp_selector = None
        self.samp_active = False
        self.samp_poly_btn.configure(text="Sample polygon")
        # redraw the RGB axis to clear the selector's overlay
        if self.samp_hsi is not None:
            self._draw_samp_rgb()

    def _polygon_mask(self, verts):
        """Boolean (H, W) mask of pixels whose centers fall inside `verts`."""
        h, w = self.samp_hsi.image.shape[:2]
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        x0 = max(0, int(np.floor(min(xs))))
        x1 = min(w, int(np.ceil(max(xs))) + 1)
        y0 = max(0, int(np.floor(min(ys))))
        y1 = min(h, int(np.ceil(max(ys))) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        # only test pixels inside the polygon's bounding box (fast on big cubes)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        inside = MplPath(verts).contains_points(pts).reshape(y1 - y0, x1 - x0)
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = inside
        return mask

    def _on_polygon(self, verts):
        if self.samp_hsi is None or len(verts) < 3:
            return
        mask = self._polygon_mask(verts)
        if mask is None or not mask.any():
            self._set_status("Polygon encloses no pixels - draw a larger shape.")
            return
        pixels = self.samp_hsi.image[mask]            # (n_pixels, n_bands)
        self.samp_wl = np.asarray(self.samp_hsi.centers, dtype=float)
        self.current_mean = np.asarray(np.nanmean(pixels, axis=0), dtype=float)
        self.current_n = int(mask.sum())
        # store this polygon's raw mean reflectance into the selected slot
        target = self.samp_target.get()
        self.samp_slots[target] = self.current_mean.copy()
        self.samp_slot_n[target] = self.current_n
        self._update_diff_plot()
        self._set_status(f"Stored {self.current_n}px into {target}.  "
                         f"slots: {self._slots_summary()}.  "
                         "Pick the next slot & draw, or Export difference (CSV).")

    # ---- normalization + difference ----
    def _norm_by_p1(self, mean):
        """Min-max normalize `mean` using P1's own min & max. None if P1 unset/flat."""
        m1 = self.samp_slots["P1"]
        if m1 is None or mean is None:
            return None
        lo, hi = np.nanmin(m1), np.nanmax(m1)
        denom = hi - lo
        if not np.isfinite(denom) or denom == 0:
            return None
        return (mean - lo) / denom

    def _difference(self):
        """N2 - N3 (or N3 - N2), both normalized by P1. None if not all set."""
        n2 = self._norm_by_p1(self.samp_slots["P2"])
        n3 = self._norm_by_p1(self.samp_slots["P3"])
        if n2 is None or n3 is None:
            return None
        return (n2 - n3) if self.samp_dir.get() == "P2 - P3" else (n3 - n2)

    def _slots_summary(self):
        return " ".join(f"{k}[{'set' if self.samp_slots[k] is not None else '-'}]"
                        for k in ("P1", "P2", "P3"))

    def _update_diff_plot(self):
        ax = self.samp_ax_spec
        ax.clear()
        ax.set_xlabel("wavelength (nm)")
        wl = self.samp_wl
        have_p1 = self.samp_slots["P1"] is not None
        colors = {"P1": "0.6", "P2": "tab:blue", "P3": "tab:green"}
        any_slot = False
        for key in ("P1", "P2", "P3"):
            m = self.samp_slots[key]
            if m is None or wl is None:
                continue
            any_slot = True
            y = self._norm_by_p1(m) if have_p1 else m
            if y is None:
                continue
            n = self.samp_slot_n[key]
            if key == "P1":
                ax.plot(wl, y, ":", color=colors[key], lw=1.0,
                        label=f"P1 normalizer (n={n})")
            else:
                lbl = f"{key} norm (n={n})" if have_p1 else f"{key} raw (n={n})"
                ax.plot(wl, y, "-", color=colors[key], lw=1.0, label=lbl)
        diff = self._difference()
        if diff is not None:
            ax.plot(wl, diff, "r-", lw=1.7, label=self.samp_dir.get())
            ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
            ax.set_ylabel("min-max normalized (by P1) / difference")
            ax.set_title(f"difference {self.samp_dir.get()}")
        else:
            ax.set_ylabel("min-max normalized (by P1)" if have_p1 else "reflectance (raw)")
            ax.set_title("sample P1 (normalizer), then P2 and P3")
        if any_slot:
            ax.legend(fontsize=7)
        self.samp_canvas.draw_idle()

    def export_difference(self):
        diff = self._difference()
        if diff is None:
            messagebox.showwarning(
                "Not ready",
                "Need P1 (normalizer), P2 and P3 all sampled first.\n"
                f"Current: {self._slots_summary()}")
            return
        path = filedialog.asksaveasfilename(
            title="Export subtracted spectrum",
            defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        col = self.samp_dir.get().replace(" ", "")   # e.g. P2-P3
        np.savetxt(path, np.column_stack([self.samp_wl, diff]), delimiter=",",
                   header=f"wavelength,{col}", comments="")
        self._set_status(f"Exported {self.samp_dir.get()} difference "
                         f"({len(diff)} bands) to {os.path.basename(path)}")

    def clear_slots(self):
        self.samp_slots = {"P1": None, "P2": None, "P3": None}
        self.samp_slot_n = {"P1": 0, "P2": 0, "P3": 0}
        self._update_diff_plot()
        self._set_status("Cleared P1/P2/P3. Sample again.")

    def save_spectrum(self):
        """Save the last-drawn polygon's raw mean reflectance as a 2xN .npy/.csv."""
        if self.current_mean is None:
            messagebox.showwarning("Nothing to save",
                                   "Sample a polygon first (draw one on the image).")
            return
        path = filedialog.asksaveasfilename(
            title="Save current polygon mean as",
            defaultextension=".npy",
            filetypes=[("NumPy 2xN", "*.npy"), ("CSV", "*.csv")])
        if not path:
            return
        wl, inten = self.samp_wl, self.current_mean
        if path.lower().endswith(".csv"):
            np.savetxt(path, np.column_stack([wl, inten]), delimiter=",",
                       header="wavelength,mean_reflectance", comments="")
        else:
            np.save(path, np.stack([wl, inten]))
        self._set_status(f"Saved current polygon mean ({len(wl)} bands, {self.current_n} px) "
                         f"to {os.path.basename(path)}")


def main():
    root = tk.Tk()
    DetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
