# PhaseExtractor_SK.py
# Fast, low-RAM phase extractor with:
# - masked polynomial background (fit ignores yellow rectangles),
# - OPTIONAL clipping of the normalized diff map to [-1, +1] rad BEFORE BG fit,
# - SIMPLE sign fix using ONE tiny ROI (magenta): if mean<0 → flip whole map,
# - paginated plots (10 pairs/page):
#     * Raw+BG (top=raw used for BG fit, bottom=polynomial BG)
#     * After subtraction (single row grid; title shows μ in ROI and "(flipped)" if applied)
# - Rectangles use the SAME coordinate system as the image arrays.
#   All full-image imshow() calls include extent=(0, W, 0, H) with origin="lower".
# - Console progress prints for each pair processed.
# - IMPORTANT: When show_plots_flag=False, NOTHING is shown but EVERYTHING is saved.
#   (show_pages and show_final_avg are gated by show_plots_flag.)
# - Saves AvgPhase.png and AvgPhase.txt.

import os
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.restoration import unwrap_phase
import cv2
import scipy.ndimage as ndi

import FourierRegionSelector_SK as FourierRegionSelector_SK


# --------------------------- utility: plotting limits ------------------------

def _sym_percentile_limits(arr_list, p=99.0):
    """Robust symmetric limits from a list of 2D arrays (per-page)."""
    if not arr_list:
        return -1.0, 1.0
    cap = 50_000
    rng = np.random.default_rng(1234)
    samps = []
    for a in arr_list:
        flat = np.abs(np.asarray(a, dtype=np.float32)).ravel()
        if flat.size <= cap:
            samps.append(flat)
        else:
            samps.append(flat[rng.choice(flat.size, size=cap, replace=False)])
    v = np.concatenate(samps)
    hi = float(np.percentile(v, p))
    return -hi, hi


# --------------------------- rectangles & overlays ---------------------------

def _sanitize_rect(r0, r1, c0, c1, H, W):
    r0 = int(np.clip(r0, 0, H)); r1 = int(np.clip(r1, 0, H))
    c0 = int(np.clip(c0, 0, W)); c1 = int(np.clip(c1, 0, W))
    if r1 < r0: r0, r1 = r1, r0
    if c1 < c0: c0, c1 = c1, c0
    return r0, r1, c0, c1

def build_ignore_mask(shape, rects: Optional[List[Tuple[int,int,int,int]]]):
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    if not rects:
        return mask
    for (r0, r1, c0, c1) in rects:
        r0, r1, c0, c1 = _sanitize_rect(r0, r1, c0, c1, H, W)
        if r1 > r0 and c1 > c0:
            mask[r0:r1, c0:c1] = True
    return mask

def _dilate_mask(mask, px):
    if not px or px <= 0:
        return mask
    k = int(px)
    se = np.ones((2*k+1, 2*k+1), dtype=bool)
    return ndi.binary_dilation(mask, structure=se)

def _draw_rects(ax, rects, H, W, color="yellow", lw=1.5):
    """
    Draw rectangles using the same row/col indices as array slicing.
    With origin='lower' and extent=(0,W,0,H), a pixel (row=r, col=c) is at (x=c, y=r).
    """
    if not rects:
        return
    for (r0, r1, c0, c1) in rects:
        r0, r1, c0, c1 = _sanitize_rect(r0, r1, c0, c1, H, W)
        if r1 <= r0 or c1 <= c0:
            continue
        ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
                               fill=False, edgecolor=color, linewidth=lw))

def _make_square_centered(r: int, c: int, half: int, H: int, W: int):
    """Return (r0,r1,c0,c1) for a 2*half x 2*half box centered at (r,c), clipped to image."""
    r0 = r - half
    r1 = r + half
    c0 = c - half
    c1 = c + half
    r0, r1, c0, c1 = _sanitize_rect(r0, r1, c0, c1, H, W)
    return (r0, r1, c0, c1)


# --------------------------- masked polynomial background --------------------

def _poly_terms(order):
    ij = []
    for i in range(order+1):
        for j in range(order+1-i):
            ij.append((i, j))
    return ij

def _build_poly_basis_full(H, W, order):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    Xn = 2.0 * (xs / (W - 1)) - 1.0
    Yn = 2.0 * (ys / (H - 1)) - 1.0
    ij = _poly_terms(order)
    cols = []
    for (i, j) in ij:
        cols.append((Xn ** i) * (Yn ** j))
    basis = np.stack(cols, axis=2).reshape(-1, len(ij)).astype(np.float32)
    return basis, ij

def _prepare_poly_fitter(H, W, order, ignore_mask, downsample=8, dilate_px=12):
    ig = _dilate_mask(ignore_mask, dilate_px)
    step = max(1, int(downsample))

    valid_ds = (~ig)[::step, ::step]
    if not np.any(valid_ds):
        raise RuntimeError("Masked polynomial fit: no valid pixels outside ignore regions.")

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    Xn_full = 2.0 * (xs / (W - 1)) - 1.0
    Yn_full = 2.0 * (ys / (H - 1)) - 1.0
    Xs = Xn_full[::step, ::step][valid_ds]
    Ys = Yn_full[::step, ::step][valid_ds]

    terms = _poly_terms(order)
    A_cols = []
    for (i, j) in terms:
        A_cols.append((Xs ** i) * (Ys ** j))
    A = np.stack(A_cols, axis=1).astype(np.float32)   # (Nsamples x nterms)

    A_pinv = np.linalg.pinv(A, rcond=1e-6).astype(np.float32)
    basis_full, _ = _build_poly_basis_full(H, W, order)
    return {
        "A_pinv": A_pinv,               # (nterms x Nsamples)
        "basis_full": basis_full,       # (H*W x nterms)
        "valid_ds": valid_ds,           # boolean mask on DS grid
        "step": step,
        "nterms": A.shape[1],
        "order": order
    }

def _fit_bg_poly_fast(img, fitter):
    step = fitter["step"]
    valid_ds = fitter["valid_ds"]
    A_pinv = fitter["A_pinv"]
    basis_full = fitter["basis_full"]

    z_ds = img[::step, ::step][valid_ds].astype(np.float32).ravel()
    coeffs = A_pinv @ z_ds                              # (nterms,)
    bg = (basis_full @ coeffs).reshape(img.shape)       # (H,W)
    return bg.astype(np.float32)


# --------------------------- page rendering (no ROI zoom) --------------------

def _save_page_raw_bg(raw_list, bg_list, out_path, ncols=5, cmap="RdBu",
                      clim_percentile=99.0, show=False, rects=None, color="yellow", lw=1.5,
                      titles=None):
    assert len(raw_list) == len(bg_list)
    N = len(raw_list)
    if N == 0: return
    H, W = raw_list[0].shape
    vmin, vmax = _sym_percentile_limits(raw_list + bg_list, p=clim_percentile)

    ncols_used = min(ncols, N)
    nrows_pairs = int(np.ceil(N / ncols_used))
    nrows = 2 * nrows_pairs
    figsize = (min(3 * ncols_used, 30), min(6 * nrows_pairs, 36))
    fig, axes = plt.subplots(nrows, ncols_used, figsize=figsize, dpi=150, constrained_layout=True)
    axes = np.atleast_2d(axes)
    if axes.ndim == 1: axes = axes[:, None]

    k = 0; im = None
    for rp in range(nrows_pairs):
        for c in range(ncols_used):
            if k >= N:
                axes[2*rp, c].axis("off"); axes[2*rp+1, c].axis("off"); continue
            ax_top = axes[2*rp, c]; ax_bot = axes[2*rp+1, c]

            im = ax_top.imshow(raw_list[k], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                               extent=(0, W, 0, H), interpolation="nearest")
            ax_top.set_xticks([]); ax_top.set_yticks([])
            title_top = titles[k] if titles and k < len(titles) else f"Pair {k}"
            ax_top.set_title(title_top, fontsize=9)
            _draw_rects(ax_top, rects or [], H, W, color=color, lw=lw)

            ax_bot.imshow(bg_list[k], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                          extent=(0, W, 0, H), interpolation="nearest")
            ax_bot.set_xticks([]); ax_bot.set_yticks([])
            ax_bot.set_title(f"{title_top} — background", fontsize=9)
            _draw_rects(ax_bot, rects or [], H, W, color=color, lw=lw)
            k += 1

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.9); cbar.set_label("Phase [rad]")

    fig.suptitle("Raw (top) & Background (bottom)", fontsize=12)
    d = os.path.dirname(out_path); d and os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show: plt.show()
    plt.close(fig)

def _save_page_detr_simple(detr_list, out_path, ncols=5, cmap="RdBu",
                           clim_percentile=99.0, show=False,
                           rects=None, color="yellow", lw=1.5,
                           pos_rects=None, titles=None):
    """
    Detrended page (single row grid):
      - full detrended map,
      - yellow: ignore_rects (BG fit),
      - magenta: sign ROI (mean printed in title, plus '(flipped)' if applied).
    """
    N = len(detr_list)
    if N == 0: return
    H, W = detr_list[0].shape
    vmin, vmax = _sym_percentile_limits(detr_list, p=clim_percentile)

    ncols_used = min(ncols, N)
    nrows = int(np.ceil(N / ncols_used))
    figsize = (min(3 * ncols_used, 30), min(3 * nrows, 30))
    fig, axes = plt.subplots(nrows, ncols_used, figsize=figsize, dpi=150, constrained_layout=True)
    axes = np.atleast_2d(axes)
    if axes.ndim == 1: axes = axes[None, :]

    have_pos = bool(pos_rects) and len(pos_rects) > 0
    if have_pos:
        rpa, rpb, cpa, cpb = pos_rects[0]
        rpa, rpb, cpa, cpb = _sanitize_rect(rpa, rpb, cpa, cpb, H, W)

    k = 0; im = None
    for r in range(nrows):
        for c in range(ncols_used):
            ax = axes[r, c]
            if k < N:
                if vmin < -0.3:
                    vmin = -0.3
                if vmax > 0.3:
                    vmax = 0.3
                im = ax.imshow(detr_list[k], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                               extent=(0, W, 0, H), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                title = titles[k] if titles and k < len(titles) else f"Pair {k}"
                ax.set_title(title, fontsize=9)
                _draw_rects(ax, rects or [], H, W, color=color, lw=lw)  # ignore (yellow)
                if have_pos:
                    _draw_rects(ax, [(rpa, rpb, cpa, cpb)], H, W, color="magenta", lw=1.5)
                k += 1
            else:
                ax.axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.9); cbar.set_label("Phase [rad]")

    fig.suptitle("After background subtraction", fontsize=12)
    d = os.path.dirname(out_path); d and os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show: plt.show()
    plt.close(fig)


# --------------------------------- main extractor ----------------------------

def PhaseExtractor(
    RawInterferograms,
    numShots,
    saveloc,
    boundary_points_sig,
    show_plots_flag,
    diag_angle_deg,
    diag_dist,
    fourier_window_size,
    *,
    # Paging / layout
    pairs_per_fig: int = 10,
    grid_ncols: int = 5,
    cmap: str = "RdBu",
    clim_percentile: float = 99.0,
    # Normalization (pre-BG)
    flip_strategy: str = "median",     # "none" | "median"
    offset_strategy: str = "median",   # "none" | "median"
    # FAST background: masked polynomial
    poly_order: int = 2,
    poly_downsample: int = 8,
    mask_dilate_px: int = 12,
    # Ignore rectangles (for BG fit and plotting)
    ignore_rects: Optional[List[Tuple[int,int,int,int]]] = None,
    overlay_rect_color: str = "yellow",
    overlay_rect_lw: float = 1.5,
    # SIGN via ONE tiny ROI (magenta)
    sign_fix: bool = True,
    pos_center_rc: Optional[Tuple[int,int]] = None,  # (row, col) center for sign box
    sign_half_size: int = 5,                          # 5 → 10x10
    # Plotting control
    show_pages: bool = False,        # can request pages, but final "show" is gated by show_plots_flag
    max_show_pages: Optional[int] = None,
    show_final_avg: bool = True,     # can request final window, but gated by show_plots_flag
    # Final outputs
    save_avg_plot: bool = True,
    # Memory control
    return_cubes: bool = False,
    # -------- NEW: robust clipping before BG fit & post-processing --------
    clip_before_bg: bool = True,
    clip_min: float = -1.0,
    clip_max: float =  1.0,
):
    """
    Returns:
        AvgPhase,
        DiffCube (or None),
        DiffCubeNorm (or None),  # NOTE: now holds the map actually used for BG fit (clipped if enabled)
        DiffCubeDetr (or None),
        BgCube (or None)
    """
    # Effective window flags: if show_plots_flag is False → never pop windows
    show_pages_eff = show_pages and show_plots_flag
    show_final_avg_eff = show_final_avg and show_plots_flag

    # Unpack
    RawInterferogramssig = np.double(RawInterferograms[0])
    RawInterferogramsbg  = np.double(RawInterferograms[1])

    nframes, nrows, ncols = RawInterferogramssig.shape
    print(f"[PhaseExtractor] Frames: {nframes}, shape per frame: {nrows}x{ncols}", flush=True)

    # ---------- FFT window selection on first signal ----------
    F0 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(RawInterferogramssig[0])))
    fourier_image_sig = np.log(np.abs(F0) + 1)
    os.makedirs(saveloc, exist_ok=True)
    png_fft = os.path.join(saveloc, '2DFFT_sig.png')
    img_scaled_sig = cv2.normalize(fourier_image_sig, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(png_fft, img_scaled_sig)
    if boundary_points_sig is None:
        boundary_points_sig = FourierRegionSelector_SK.show_mouse_select(png_fft, fourier_window_size)
        print('fourier boundary points:', boundary_points_sig, flush=True)

    # (row_top, row_bot, col_left, col_right)
    boundary_sig = (
        boundary_points_sig[0][1], boundary_points_sig[1][1],
        boundary_points_sig[0][0], boundary_points_sig[1][0]
    )
    print(f"[PhaseExtractor] Fourier window rows={boundary_sig[0]}..{boundary_sig[1]}, "
          f"cols={boundary_sig[2]}..{boundary_sig[3]}", flush=True)

    # ---------- Precompute 2D window, centering, diagonal mask ----------
    r0, r1, c0, c1 = boundary_sig
    win2d = np.zeros((nrows, ncols), dtype=np.float32)
    win2d[r0:r1, c0:c1] = np.outer(np.hanning(r1 - r0), np.hanning(c1 - c0)).astype(np.float32)

    col_shift = int(nrows/2) - int(boundary_sig[0] + (boundary_sig[1] - boundary_sig[0]) / 2)
    row_shift = int(ncols/2) - int(boundary_sig[2] + (boundary_sig[3] - boundary_sig[2]) / 2)

    if diag_angle_deg >= 0:
        quadrants = ['Q1', 'Q3']
    else:
        quadrants = ['Q2', 'Q4']
    print("[PhaseExtractor] Masking quadrants:", quadrants, flush=True)

    ang = np.deg2rad(diag_angle_deg)
    xo = int(diag_dist * np.cos(abs(ang)))
    yo = int(np.sin(abs(ang)) * diag_dist)
    x1 = int(ncols/2) + xo; y1 = int(nrows/2) + yo
    x2 = int(ncols/2) - xo; y2 = int(nrows/2) + yo
    x3 = int(ncols/2) - xo; y3 = int(nrows/2) - yo
    x4 = int(ncols/2) + xo; y4 = int(nrows/2) - yo
    side = 100
    m2d = np.ones((nrows, ncols), dtype=np.float32)
    if 'Q1' in quadrants: m2d[y1:y1+side, x1:x1+side] = 0
    if 'Q2' in quadrants: m2d[y2:y2+side, x2-side:x2] = 0
    if 'Q3' in quadrants: m2d[y3-side:y3, x3-side:x3] = 0
    if 'Q4' in quadrants: m2d[y4-side:y4, x4:x4+side] = 0

    if show_plots_flag:
        F_demo = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(RawInterferogramssig[0])))
        F_demo = F_demo * win2d
        F_demo = np.roll(F_demo, row_shift, axis=-1)
        F_demo = np.roll(F_demo, col_shift, axis=-2)
        F_demo = F_demo * m2d
        im = np.log(np.abs(F_demo) + 1)
        plt.figure(); plt.imshow(im, origin='lower'); plt.title("FFT after window/mask (frame 0)"); plt.show()

    # ---------- precompute masked polynomial fitter (FAST) ----------
    ignore_mask = build_ignore_mask((nrows, ncols), ignore_rects or [])
    poly_fitter = _prepare_poly_fitter(
        H=nrows, W=ncols, order=poly_order,
        ignore_mask=ignore_mask,
        downsample=poly_downsample,
        dilate_px=mask_dilate_px
    )

    # ---------- small sign ROI (magenta) ----------
    pos_rect_list: List[Tuple[int,int,int,int]] = []
    if pos_center_rc is not None:
        pr, pc = int(pos_center_rc[0]), int(pos_center_rc[1])
        pos_rect_list = [_make_square_centered(pr, pc, sign_half_size, nrows, ncols)]
        (rpa, rpb, cpa, cpb) = pos_rect_list[0]
        print(f"[PhaseExtractor] Sign ROI (magenta): rows={rpa}..{rpb}, cols={cpa}..{cpb}", flush=True)
    elif sign_fix:
        print("WARNING: sign_fix=True but pos_center_rc is None → no sign correction will be applied.", flush=True)

    # ---------- streaming buffers ----------
    page_raw, page_bg, page_detr = [], [], []
    page_titles_rb, page_titles_dt = [], []
    page_idx = 1
    shown_pages = 0
    out_dir_rb = os.path.join(saveloc, "pairs_raw_vs_bg"); os.makedirs(out_dir_rb, exist_ok=True)
    out_dir_dt = os.path.join(saveloc, "pairs_after_subtraction"); os.makedirs(out_dir_dt, exist_ok=True)

    # optional cubes (off by default)
    if return_cubes:
        DiffCube = np.zeros((nframes, nrows, ncols), np.float32)
        DiffCubeNorm = np.zeros_like(DiffCube)   # will store the map used for BG fit (clipped if enabled)
        BgCube = np.zeros_like(DiffCube)
        DiffCubeDetr = np.zeros_like(DiffCube)
    else:
        DiffCube = DiffCubeNorm = BgCube = DiffCubeDetr = None

    # accumulator for final average
    accum = np.zeros((nrows, ncols), dtype=np.float64)
    total_flipped = 0

    # ---------- per-frame processing ----------
    for i in range(nframes):
        print(f"[{i+1}/{nframes}] Processing pair...", flush=True)

        # SIGNAL
        Fsig = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(RawInterferogramssig[i])))
        Fsig *= win2d
        Fsig = np.roll(Fsig, row_shift, axis=-1)
        Fsig = np.roll(Fsig, col_shift, axis=-2)
        Fsig *= m2d

        # BACKGROUND
        Fbg = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(RawInterferogramsbg[i])))
        Fbg *= win2d
        Fbg = np.roll(Fbg, row_shift, axis=-1)
        Fbg = np.roll(Fbg, col_shift, axis=-2)
        Fbg *= m2d

        # IFFT → phase
        isg = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Fsig)))
        ibg = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Fbg)))
        ph_s = np.angle(isg).astype(np.float32)
        ph_b = np.angle(ibg).astype(np.float32)

        # unwrap & de-piston
        uf = unwrap_phase(ph_s).astype(np.float32); uf -= np.median(uf)
        ub = unwrap_phase(ph_b).astype(np.float32); ub -= np.median(ub)

        # raw difference (pre-BG)
        diff = uf - ub
        if flip_strategy == "median" and np.median(diff) < 0:
            diff = -diff
        if offset_strategy == "median":
            diff = diff - np.median(diff)

        # This is the normalized difference map (pre-BG)
        diff_norm = diff.astype(np.float32)

        # -------- NEW: clip spikes before BG fit and all post-processing --------
        if clip_before_bg:
            # Count and report (lightweight) how many pixels were clipped
            n_clip_low = int(np.count_nonzero(diff_norm < clip_min))
            n_clip_high = int(np.count_nonzero(diff_norm > clip_max))
            n_clip = n_clip_low + n_clip_high
            if n_clip > 0:
                print(f"  - Clipping {n_clip} px to [{clip_min}, {clip_max}] (low={n_clip_low}, high={n_clip_high})", flush=True)
            diff_for_bg = np.clip(diff_norm, clip_min, clip_max).astype(np.float32)
        else:
            diff_for_bg = diff_norm

        # masked polynomial background on the (possibly clipped) map
        bg = _fit_bg_poly_fast(diff_for_bg, poly_fitter)

        # subtract & re-center (continue with the clipped data)
        detr = diff_for_bg - bg
        detr = detr - np.median(detr)

        # -------- SIMPLE SIGN FIX: mean in magenta box; if mean<0 → flip --------
        flipped = False
        title_info = ""
        roi_mean_display = ""
        if sign_fix and pos_rect_list:
            (rpa, rpb, cpa, cpb) = pos_rect_list[0]
            roi_pos = detr[rpa:rpb, cpa:cpb]
            mpos = float(np.mean(roi_pos)) if roi_pos.size else 0.0
            if mpos < 0.0:
                detr = -detr
                flipped = True
                mpos = -mpos  # for display consistency
            roi_mean_display = f" μROI={mpos:+.2e}"
        title_info = ((" (flipped)" if flipped else "") + roi_mean_display)

        if flipped:
            total_flipped += 1

        # accumulate AFTER sign correction
        accum += detr

        # optional cubes
        if return_cubes:
            DiffCube[i]      = diff_norm            # original pre-clip normalized diff (for traceability)
            DiffCubeNorm[i]  = diff_for_bg          # the map actually used for BG fit (clipped if enabled)
            BgCube[i]        = bg
            DiffCubeDetr[i]  = detr

        # add to pages (copies) and titles
        # Show the map that was actually used for BG fitting to keep visuals faithful
        page_raw.append(diff_for_bg.copy())
        page_bg.append(bg.copy())
        page_detr.append(detr.copy())

        t_global = f"Pair {i}"
        page_titles_rb.append(t_global)
        page_titles_dt.append(t_global + title_info)

        # flush full pages
        if len(page_raw) == pairs_per_fig:
            show_now = show_pages_eff and (max_show_pages is None or shown_pages < max_show_pages)
            _save_page_raw_bg(page_raw, page_bg,
                              os.path.join(out_dir_rb, f"pairs_raw_vs_bg_page{page_idx:02d}.png"),
                              ncols=grid_ncols, cmap=cmap, clim_percentile=clim_percentile,
                              show=show_now, rects=ignore_rects,
                              color=overlay_rect_color, lw=overlay_rect_lw,
                              titles=page_titles_rb)
            _save_page_detr_simple(page_detr,
                                   os.path.join(out_dir_dt, f"pairs_after_sub_page{page_idx:02d}.png"),
                                   ncols=grid_ncols, cmap=cmap, clim_percentile=clim_percentile,
                                   show=show_now, rects=ignore_rects,
                                   color=overlay_rect_color, lw=overlay_rect_lw,
                                   pos_rects=pos_rect_list, titles=page_titles_dt)
            if show_now:
                shown_pages += 1
            page_idx += 1
            page_raw, page_bg, page_detr = [], [], []
            page_titles_rb, page_titles_dt = [], []

    # flush partial page
    if page_raw:
        show_now = show_pages_eff and (max_show_pages is None or shown_pages < max_show_pages)
        _save_page_raw_bg(page_raw, page_bg,
                          os.path.join(out_dir_rb, f"pairs_raw_vs_bg_page{page_idx:02d}.png"),
                          ncols=grid_ncols, cmap=cmap, clim_percentile=clim_percentile,
                          show=show_now, rects=ignore_rects,
                          color=overlay_rect_color, lw=overlay_rect_lw,
                          titles=page_titles_rb)
        _save_page_detr_simple(page_detr,
                               os.path.join(out_dir_dt, f"pairs_after_sub_page{page_idx:02d}.png"),
                               ncols=grid_ncols, cmap=cmap, clim_percentile=clim_percentile,
                               show=show_now, rects=ignore_rects,
                               color=overlay_rect_color, lw=overlay_rect_lw,
                               pos_rects=pos_rect_list, titles=page_titles_dt)

    # final average (of sign-corrected detrended maps)
    AvgPhase = (accum / max(1, nframes)).astype(np.float32)

    # save final average
    if save_avg_plot:
        vmin, vmax = _sym_percentile_limits([AvgPhase], p=clim_percentile)
        if vmin < -0.3:
            vmin = -0.3
        if vmax > 0.3:
            vmax = 0.3
        fig = plt.figure(figsize=(6, 5), dpi=150)
        ax = fig.add_subplot(111)
        im = ax.imshow(AvgPhase, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax,
                       extent=(0, ncols, 0, nrows), interpolation="nearest")
        ax.set_title("Final average phase map (after background subtraction)")
        cbar = fig.colorbar(im, ax=ax); cbar.set_label("Phase [rad]")
        fig.tight_layout()
        out_png = os.path.join(saveloc, "AvgPhase.png")
        fig.savefig(out_png, dpi=150)
        if show_final_avg_eff:
            plt.show()
        plt.close(fig)
        print("Saved:", out_png, flush=True)

    out_txt = os.path.join(saveloc, "AvgPhase.txt")
    np.savetxt(out_txt, AvgPhase)
    print(f"Saved: {out_txt}", flush=True)
    print(f"Sign flips (single-ROI) on {total_flipped} / {nframes} pairs.", flush=True)

    return AvgPhase, DiffCube, DiffCubeNorm, DiffCubeDetr, BgCube
