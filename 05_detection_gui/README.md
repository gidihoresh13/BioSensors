# 05 - Detection GUI

A minimal **desktop GUI** (Tkinter) with two tabs:

- **Detection** -- load a hyperspectral cube, pick a reference spectrum, run the
  unmixing classifier, tune the detection threshold live, inspect per-pixel
  spectra, and export results.
- **Spectrum sampler** -- load a *separate* hyperspectral image, draw a polygon
  on it, preview the **mean spectrum** of the enclosed pixels on a graph, and
  save it to a `.npy` you name. That file is a ready-made reference spectrum for
  the Detection tab.

This folder is **self-contained** -- it ships its own copy of the detection
engine (the [`hsi_detect`](hsi_detect) package), so it has no dependency on
other modules. (The engine originates from `04_image_processing`.)

## Tab 1 -- Detection

| Step | Action | Under the hood |
|------|--------|----------------|
| 1 | **Open cube (.hdr)** | `HyperspectralImage` loads the ENVI cube, smooths, renders RGB |
| 2 | **Open reference** | `Spectrum` loads a 2xN `.npy`/`.csv`, interpolated to the cube's bands |
| 3 | **Run detection** | `HierarchicalKMeansUnmixer.fit` + `.classify` on a background thread |
| 4 | **Threshold slider** | re-applies the threshold to the cached score map (no re-fit) |
| 5 | **Click a pixel** | plots that pixel's spectrum against the reference |
| 6 | **Export** | writes `<date>_score.tiff`, `<date>_detection_view.png`, `<date>_params.json` |

Layout: **RGB | detection score | pixel spectrum**.

## Tab 2 -- Spectrum sampler

Builds a reference spectrum by averaging a hand-drawn region of a hyperspectral
image. It is fully **independent** of the Detection tab -- open any cube here,
including a different one from the detection cube.

> **The image must be a hyperspectral cube (ENVI `.hdr`).** A "mean spectrum"
> only exists if the file has spectral bands; an ordinary RGB photo has just
> three channels and cannot produce one.

1. **Open image (.hdr)** -- loads the cube at full resolution and renders it
   (left panel), so region boundaries stay precise.
2. **Sample polygon** (button turns into *Stop sampling*).
3. Click on the image to drop vertices; click the first vertex to close the
   shape. Press **Esc** to discard and restart the current shape.
4. The **mean spectrum** of every pixel inside the polygon is graphed live in the
   right panel. Dragging a vertex re-computes it, so you can preview before saving.
5. **Save mean spectrum (.npy)...** -- type a filename and save. The default
   `.npy` is a 2xN array (row 0 = wavelengths, row 1 = intensities) that loads
   straight back via **Open reference** on the Detection tab. `.csv` is also
   offered.

Layout: **image (draw here) | mean spectrum (preview)**.

### Zooming the image

- **Mouse wheel** over the image zooms in/out centered on the cursor. This needs
  no mode switch, so you can zoom and keep clicking polygon vertices. Zooming out
  stops at the full image.
- The **navigation toolbar** under the image adds *Pan* and *Zoom-to-rectangle*
  (drag a box). While either is active, clicks drive navigation instead of the
  polygon -- click the tool again to turn it off and resume drawing. The **Home**
  button resets to the full image.

## Requirements

- **Python 3.11** (this is important -- see the note below).
- Packages in [`requirements.txt`](requirements.txt): numpy, scipy, scikit-learn,
  matplotlib, pillow, spectral. `tkinter` ships with Python (no install).
- The `hsi_detect` engine is **bundled in this folder** (`./hsi_detect`), so no
  external module is required.

> **Interpreter note.** This machine has multiple Python installs and the `py`
> launcher defaults to **3.14**, which does **not** have these packages. Running
> under 3.14 fails with `ModuleNotFoundError: No module named 'spectral'`
> (and sklearn, matplotlib, PIL). Always use **3.11** -- `run_gui.bat` does this
> for you, or select `C:\Program Files\Python311\python.exe` as your interpreter
> in VS Code.

## Install

```powershell
py -3.11 -m pip install -r requirements.txt
```

## Run

Any of:

```powershell
# from this folder
py -3.11 gui_detect.py
```

- **Double-click `run_gui.bat`** (forces Python 3.11).
- In **VS Code**: `Ctrl+Shift+P` -> *Python: Select Interpreter* ->
  `C:\Program Files\Python311\python.exe`, then Run.

## Inputs

The file dialogs accept **any** path, so your own data can live anywhere -- e.g.
a `data/` folder next to this one at the repo root. The upstream sample data
also works; it sits in `from iati-levin/`.

**Hyperspectral cubes** (ENVI `.hdr` + matching raw file in the same folder),
e.g. the upstream samples in `../from iati-levin/04_image_processing/00_data`:
- `rg_on_sand_induction_54m/raw_0_rd_rf_or.hdr`
- `uav_flight_hsi/raw_0_rd_rf_or.hdr`

If the `uav_flight_hsi/` folder is missing, extract it first:
`bash ../from\ iati-levin/04_image_processing/download_data.sh` (or unzip
`00_data/uav_flight_hsi.zip`).

**Reference spectrum** -- a 2-row array (row 0 = wavelengths in nm, row 1 =
intensities), `.npy` or `.csv`. Upstream examples in
`../from iati-levin/04_image_processing/00_data/absorbance_data/`, e.g.
`bpHO-smURFP_infered_absorbance_from_ecoli_pellets_10Jul2024.npy`.

## Notes

- **Downsample** (Detection tab, default 2) strides the cube spatially for
  faster detection iteration. Set to 1 for full resolution. The full cube still
  loads into RAM first (~1-2 GB for the ~900 MB cubes), so give it a moment. The
  Spectrum sampler tab has no downsample -- it always loads full resolution.
- Detection runs on a **worker thread** so the window stays responsive; results
  are marshaled back to the Tk main thread via a queue.
- The GUI only uses `hsi_detect`'s `image`, `spectrum`, `classifier`, and
  `utils` modules. The bundled `hsi_detect/grid_analysis.py` is **not** used by
  the GUI and needs extra packages (pandas, dash, plotly, scikit-image, tqdm)
  that are intentionally left out of `requirements.txt`. On this machine
  `scikit-image`/`opencv` also have a NumPy-2 ABI mismatch -- irrelevant here
  because the GUI never imports them.

## Files

- `gui_detect.py` -- the application.
- `hsi_detect/` -- bundled detection engine (`image`, `spectrum`, `classifier`,
  `utils`, plus the unused `grid_analysis`).
- `requirements.txt` -- Python dependencies.
- `run_gui.bat` -- Windows launcher that uses Python 3.11.
- `README.md` -- this file.
