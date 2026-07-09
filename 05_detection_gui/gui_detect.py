"""
Minimal Tkinter GUI for hyperspectral reporter detection.

Wraps the module-04 detection engine (hsi_detect: HyperspectralImage +
Spectrum + HierarchicalKMeansUnmixer) in a desktop UI:

  1. Open HSI cube        -> loads an ENVI .hdr (+raw) and shows an RGB render
  2. Open reference        -> loads a 2xN .npy/.csv reference spectrum
  3. Run detection         -> fit/classify on a background thread (UI stays live)
  4. Threshold slider      -> re-applies threshold on the cached score map (no re-fit)
  5. Click a pixel         -> plots that pixel's spectrum vs. the reference
  6. Export                -> saves score TIFF + heatmap PNG + params JSON

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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        root.geometry("1150x720")

        # state
        self.hsi = None            # HyperspectralImage (possibly downsampled)
        self.ref_path = None       # path to reference spectrum file
        self.ref_disp = None       # Spectrum for display (interp to cube if loaded)
        self.raw_scores = None     # 2D float score map (unthresholded)
        self.q = queue.Queue()     # worker -> UI messages

        self._build_ui()
        self.root.after(100, self._poll)

    # ---------- UI construction ----------
    def _build_ui(self):
        bar = ttk.Frame(self.root, padding=6)
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
        prow = ttk.Frame(self.root, padding=(6, 0))
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

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.status = tk.StringVar(value="Open an HSI cube (.hdr) to begin.")
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN,
                  anchor=tk.W, padding=4).pack(side=tk.BOTTOM, fill=tk.X)

    # ---------- helpers ----------
    def _set_status(self, msg):
        self.status.set(msg)

    def _busy(self, on):
        self.run_btn.configure(state=(tk.DISABLED if on else tk.NORMAL))
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
                elif kind == "error":
                    self._busy(False)
                    messagebox.showerror("Error", payload)
                    self._set_status("Error: " + payload.splitlines()[-1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ---------- actions ----------
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

    # ---------- drawing ----------
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


def main():
    root = tk.Tk()
    DetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
