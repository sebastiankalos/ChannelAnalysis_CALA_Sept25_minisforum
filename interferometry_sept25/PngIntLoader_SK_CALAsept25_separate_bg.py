# This is a script to load tif interferograms from CALA
# specifically to load data from a channel measurement campaign in September 2025
# by Sebastian Kalos, University of Oxford, September 2025

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

import InterferogramRegionSelector_SK as InterferogramRegionSelector_SK

# --- Timestamp parsing from filename ---

_TS_RE = re.compile(
    r"""^
        Interferometry(?P<chan>[12])        # channel 1 or 2
        _shot_
        (?P<ymd>\d{8})                      # YYYYMMDD
        _
        (?P<hms>\d{6})                      # HHMMSS
        _
        (?P<ms>\d{3})                       # milliseconds
        _n(?P<seq>\d+)                      # sequence/counter
        $""",
    re.VERBOSE
)

def extract_timestamp_ms_from_filename(filename: str) -> Optional[int]:
    """
    Extract UNIX timestamp in milliseconds from a filename of the form:
    Interferometry[1|2]_shot_YYYYMMDD_HHMMSS_mmm_nXXX(.tif/.tiff)

    Returns None if the pattern doesn't match.
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    m = _TS_RE.match(base_name)
    if not m:
        return None
    ymd = m.group("ymd")
    hms = m.group("hms")
    ms  = int(m.group("ms"))

    # Assume timestamps are UTC (change tzinfo if you want local time)
    dt = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + ms


# --- Helpers for listing/sorting files ---

def _tif_like(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".tif", ".tiff")

def list_tif_files_sorted(folder_path: str) -> List[str]:
    """
    Return a list of .tif/.tiff filenames in `folder_path`, sorted by:
      1) embedded timestamp if available,
      2) else file modification time,
      3) else lexicographic name.
    """
    files = []
    for fn in os.listdir(folder_path):
        fp = os.path.join(folder_path, fn)
        if not os.path.isfile(fp):
            continue
        if not _tif_like(fn):
            continue
        ts = extract_timestamp_ms_from_filename(fn)
        if ts is None:
            try:
                ts = int(os.path.getmtime(fp) * 1000)
            except Exception:
                ts = None
        files.append((fn, ts))
    # Sort: those with ts first by ts, then by name; those without ts go to the end by name
    files.sort(key=lambda x: (0 if x[1] is not None else 1, x[1] if x[1] is not None else 0, x[0]))
    return [fn for fn, _ in files]

def count_tif_and_timestamped(folder_path: str) -> Tuple[int, int]:
    """Return (#.tif/.tiff files, # that match the timestamp pattern)."""
    total = 0
    stamped = 0
    for fn in os.listdir(folder_path):
        fp = os.path.join(folder_path, fn)
        if not os.path.isfile(fp) or not _tif_like(fn):
            continue
        total += 1
        if extract_timestamp_ms_from_filename(fn) is not None:
            stamped += 1
    return total, stamped


# --- Pairing with 1s + 10ms target gap (legacy mode) ---

def find_tif_file_pairs(
    folder_path: str,
    expected_interval_ms: int = 1010,  # 1s + 10ms
    tolerance_ms: int = 100            # ±100ms leeway
) -> List[Tuple[str, str]]:
    """
    Finds pairs of '.tif' / '.tiff' files in a folder whose timestamps differ by
    ~expected_interval_ms within ±tolerance_ms.

    Returns: List[(earlier_file, later_file)]
    """
    print(f"Looking in folder: {folder_path}")
    file_info = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        _, ext = os.path.splitext(filename)
        if ext.lower() not in (".tif", ".tiff"):
            continue

        ts_ms = extract_timestamp_ms_from_filename(filename)
        if ts_ms is None:
            # keep quiet here; we count these in the summary later
            continue
        file_info.append((filename, ts_ms))

    file_info.sort(key=lambda x: x[1])  # sort by timestamp

    pairs: List[Tuple[str, str]] = []
    used = set()
    i = 0
    while i < len(file_info) - 1:
        if i in used:
            i += 1
            continue
        fname_i, t_i = file_info[i]
        j = i + 1
        matched = False
        while j < len(file_info):
            if j in used:
                j += 1
                continue
            fname_j, t_j = file_info[j]
            dt = t_j - t_i
            if abs(dt - expected_interval_ms) <= tolerance_ms:
                pairs.append((fname_i, fname_j))
                used.add(i); used.add(j)
                matched = True
                break
            if dt > expected_interval_ms + tolerance_ms:
                break
            j += 1
        i += 1 if matched else 1

    print(f"Found {len(pairs)} pairs (Δt≈{expected_interval_ms} ms, ±{tolerance_ms} ms).")
    return pairs


# --- Image utilities ---

def bin_image(image, bin_factor=1):
    """Bins a 2D image by the given binning factor."""
    if bin_factor <= 1:
        return image
    h, w = image.shape
    h_binned = h // bin_factor
    w_binned = w // bin_factor
    image = image[:h_binned * bin_factor, :w_binned * bin_factor]  # crop excess
    image_binned = image.reshape(h_binned, bin_factor, w_binned, bin_factor).mean(axis=(1, 3))
    return image_binned


# --- Pretty printing ---

def _print_pairing_summary(stats: Dict[str, Any]) -> None:
    print("\n=== Pairing summary ===")
    mode = stats.get("mode", "?")
    print(f"Mode: {mode}")
    if mode == "bgpath":
        print(f"Signal frames in sigpath:     {stats.get('signal_total', 0)}")
        print(f"Background frames in bgpath:  {stats.get('background_total', 0)}")
    else:
        print(f"TIF files in folder:          {stats.get('folder_total_tif', 0)}")
        print(f"Timestamped filenames:        {stats.get('folder_timestamped', 0)}")
        print(f"Pairs found by Δt:            {stats.get('pairs_found', 0)}")
        if "dt_ms_stats" in stats:
            s = stats["dt_ms_stats"]
            print("Δt stats [ms] over selected pairs: "
                  f"min={s['min']:.0f}, p50={s['median']:.0f}, mean={s['mean']:.1f}, "
                  f"std={s['std']:.1f}, max={s['max']:.0f}")
    print(f"Requested pairs:              {stats.get('requested', 0)}")
    print(f"Available (before cap):       {stats.get('available', 0)}")
    print(f"Loaded pairs:                 {stats.get('loaded', 0)}")
    print("========================\n")


# --- Main loader ---

def PngIntLoader(
    sigpath,
    sigheader,
    numShots,
    loadFull,
    bin_factor=1,
    expected_interval_ms=1010,
    tolerance_ms=100,
    *,
    bgpath: Optional[str] = None,
    return_stats: bool = False
):
    """
    Load signal/background interferograms.

    Modes
    -----
    - bgpath is None  : pair S/B inside `sigpath` by timestamp gap (~expected_interval_ms ± tolerance_ms).
    - bgpath provided : ignore time; pair k-th signal in `sigpath` with k-th background in `bgpath`
                        after sorting each folder independently.

    Returns
    -------
    If return_stats=False (default): [sig_stack, bg_stack]
    If return_stats=True : ([sig_stack, bg_stack], stats_dict)
    """

    # Optional region selection (spatial crop)
    if not loadFull:
        points = InterferogramRegionSelector_SK.show_mouse_select(sigpath, sigheader)
        loadRegion = (points[1][1], points[0][1], points[1][0], points[0][0])
        print('These are the coordinates: ' + str(loadRegion))

    stats: Dict[str, Any] = {}

    # -------------------------
    # Build (sig, bg) path lists
    # -------------------------
    if bgpath is None:
        # Legacy: find pairs within sigpath by time gap
        total_tif, total_stamped = count_tif_and_timestamped(sigpath)
        stats.update({
            "mode": "legacy",
            "folder_total_tif": total_tif,
            "folder_timestamped": total_stamped,
        })

        pairs = find_tif_file_pairs(sigpath, expected_interval_ms=expected_interval_ms, tolerance_ms=tolerance_ms)
        stats["pairs_found"] = len(pairs)

        # dt stats for selected pairs
        dt_list = []
        for f1, f2 in pairs:
            t1 = extract_timestamp_ms_from_filename(f1)
            t2 = extract_timestamp_ms_from_filename(f2)
            if t1 is not None and t2 is not None:
                dt_list.append(t2 - t1)
        if dt_list:
            arr = np.asarray(dt_list, dtype=np.float64)
            stats["dt_ms_stats"] = {
                "min": float(arr.min()),
                "median": float(np.median(arr)),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "max": float(arr.max()),
            }

        sigpathslist = [os.path.join(sigpath, f1) for (f1, _) in pairs]
        bgpathslist  = [os.path.join(sigpath, f2) for (_, f2) in pairs]

    else:
        # New: independent folders; pair by index after sorting
        stats["mode"] = "bgpath"
        print(f"Now processing SIGNAL folder: {sigpath}")
        sig_files = list_tif_files_sorted(sigpath)
        print(f"Found {len(sig_files)} signal frames.")

        print(f"Now processing BACKGROUND folder: {bgpath}")
        bg_files  = list_tif_files_sorted(bgpath)
        print(f"Found {len(bg_files)} background frames.")

        stats["signal_total"] = len(sig_files)
        stats["background_total"] = len(bg_files)

        if len(sig_files) == 0 or len(bg_files) == 0:
            raise RuntimeError("No signal or background frames found. Check sigpath/bgpath.")

        sigpathslist = [os.path.join(sigpath, f) for f in sig_files]
        bgpathslist  = [os.path.join(bgpath,  f) for f in bg_files]

    # -------------------------
    # Decide how many frames to load
    # -------------------------
    if isinstance(numShots, int):
        requested = numShots
    else:
        requested = len(numShots)
    available = min(len(sigpathslist), len(bgpathslist))
    num_frames = min(requested, available)

    stats["requested"] = int(requested)
    stats["available"] = int(available)
    stats["loaded"]    = int(num_frames)

    if num_frames <= 0:
        _print_pairing_summary(stats)
        raise RuntimeError("No pairs available to load.")

    # -------------------------
    # Inspect first images for shape; set up arrays
    # -------------------------
    first_sig = plt.imread(sigpathslist[0]).astype('float32')
    if first_sig.ndim == 3:
        first_sig = first_sig[..., :3].mean(axis=-1)
    first_sig = bin_image(first_sig, bin_factor)

    first_bg  = plt.imread(bgpathslist[0]).astype('float32')
    if first_bg.ndim == 3:
        first_bg = first_bg[..., :3].mean(axis=-1)
    first_bg = bin_image(first_bg, bin_factor)

    if first_sig.shape != first_bg.shape:
        _print_pairing_summary(stats)
        raise RuntimeError(f"Signal and background shapes differ after binning: {first_sig.shape} vs {first_bg.shape}")

    binned_shape = first_sig.shape
    print(f"Binned shape: {binned_shape}")

    RawInterferogramssig = np.zeros((num_frames, binned_shape[0], binned_shape[1]), dtype='float32')
    RawInterferogramsbg  = np.zeros((num_frames, binned_shape[0], binned_shape[1]), dtype='float32')

    # Put first ones in (already loaded)
    RawInterferogramssig[0] = first_sig
    RawInterferogramsbg[0]  = first_bg

    # Load the rest
    for i in range(1, num_frames):
        tempSig = plt.imread(sigpathslist[i]).astype('float32')
        tempBg  = plt.imread(bgpathslist[i]).astype('float32')
        if tempSig.ndim == 3:
            tempSig = tempSig[..., :3].mean(axis=-1)
        if tempBg.ndim == 3:
            tempBg = tempBg[..., :3].mean(axis=-1)
        RawInterferogramssig[i] = bin_image(tempSig, bin_factor)
        RawInterferogramsbg[i]  = bin_image(tempBg,  bin_factor)
        if not isinstance(numShots, int):
            # mirror your earlier "print index" behavior if a list is passed
            try:
                print(numShots[i])
            except Exception:
                print(i)

    # Optional crop
    if not loadFull:
        newRawInterferogramssig = RawInterferogramssig[:, loadRegion[0]:loadRegion[1], loadRegion[2]:loadRegion[3]]
        newRawInterferogramsbg  = RawInterferogramsbg[:,  loadRegion[0]:loadRegion[1], loadRegion[2]:loadRegion[3]]
    else:
        newRawInterferogramssig = RawInterferogramssig
        newRawInterferogramsbg  = RawInterferogramsbg

    # Final summary print
    _print_pairing_summary(stats)

    if return_stats:
        return [newRawInterferogramssig[:num_frames], newRawInterferogramsbg[:num_frames]], stats
    else:
        return [newRawInterferogramssig[:num_frames], newRawInterferogramsbg[:num_frames]]
