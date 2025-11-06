# This is a script to load tif interferograms from CALA
# specifically to load data from a channel measurement campaign in September 2025
# by Sebastian Kalos, University of Oxford, September 2025

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Union

import InterferogramRegionSelector_SK as InterferogramRegionSelector_SK

# --- Timestamp parsing from filename ---

# Old format:
# Interferometry1_shot_YYYYMMDD_HHMMSS_mmm_n001(.tif/.tiff)
_TS_RE_OLD = re.compile(
    r"""^
        Interferometry(?P<chan>[12])
        _shot_
        (?P<ymd>\d{8})
        _
        (?P<hms>\d{6})
        _
        (?P<ms>\d{3})
        _n(?P<seq>\d+)
        $""",
    re.VERBOSE
)

# New format (with "trigd" and ISO-ish T + dot for milliseconds):
# Interferometry1_shot_trigdYYYYMMDDTHHMMSS.mmm_n001(.tif/.tiff)
_TS_RE_TRIGD = re.compile(
    r"""^
        Interferometry(?P<chan>[12])
        _shot_
        trigd
        (?P<ymd>\d{8})
        T
        (?P<hms>\d{6})
        \.
        (?P<ms>\d{3})
        _n(?P<seq>\d+)
        $""",
    re.VERBOSE
)

def extract_timestamp_ms_from_filename(filename: str) -> Optional[int]:
    """
    Extract UNIX timestamp in milliseconds from supported filename patterns:

    Old: Interferometry[1|2]_shot_YYYYMMDD_HHMMSS_mmm_nXXX(.tif/.tiff)
    New: Interferometry[1|2]_shot_trigdYYYYMMDDTHHMMSS.mmm_nXXX(.tif/.tiff)

    Returns None if the pattern doesn't match.
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]

    m = _TS_RE_TRIGD.match(base_name)
    if not m:
        m = _TS_RE_OLD.match(base_name)
        if not m:
            return None

    ymd = m.group("ymd")
    hms = m.group("hms")
    ms  = int(m.group("ms"))

    # Treat timestamps as UTC (change tzinfo if you want local time)
    dt = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + ms

# --- Pairing with 30 ms target gap (±50 ms) ---

def find_tif_file_pairs(
    folder_path: str,
    expected_interval_ms: int = 30,   # default: 30 ms
    tolerance_ms: int = 50            # ±50 ms leeway
) -> List[Tuple[str, str]]:
    """
    Finds pairs of '.tif' / '.tiff' files in a folder whose timestamps differ by
    ~expected_interval_ms within ±tolerance_ms.

    Returns: List[(earlier_file, later_file)]
    """
    print(f"Looking in folder: {folder_path}")
    file_info: List[Tuple[str, int]] = []

    total_files = 0
    matched_files = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        _, ext = os.path.splitext(filename)
        if ext.lower() not in (".tif", ".tiff"):
            continue

        total_files += 1
        ts_ms = extract_timestamp_ms_from_filename(filename)
        if ts_ms is None:
            # Not matching either pattern — skip quietly or print if you prefer:
            # print(f"Skipping file with unmatched pattern: {filename}")
            continue

        file_info.append((filename, ts_ms))
        matched_files += 1

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
    print(f"TIF/TIFF files considered: {matched_files} / {total_files} total in folder.")
    return pairs

# --- Image utilities ---

def bin_image(image: np.ndarray, bin_factor: int = 1) -> np.ndarray:
    """Bins a 2D image by the given binning factor."""
    if bin_factor <= 1:
        return image
    h, w = image.shape
    h_binned = h // bin_factor
    w_binned = w // bin_factor
    image = image[:h_binned * bin_factor, :w_binned * bin_factor]  # crop excess
    image_binned = image.reshape(h_binned, bin_factor, w_binned, bin_factor).mean(axis=(1, 3))
    return image_binned.astype(image.dtype, copy=False)

# --- Main loader ---

def PngIntLoader(
    sigpath: str,
    sigheader: str,
    numShots: Union[int, List[int]],
    loadFull: bool,
    bin_factor: int = 1,
    expected_interval_ms: int = 30,   # default now 30 ms
    tolerance_ms: int = 50
):
    """
    Loads pairs separated by ~expected_interval_ms (default 30 ms) with ±tolerance_ms.
    Returns [sig_stack, bg_stack] shaped (N, H, W).
    """
    if not loadFull:
        points = InterferogramRegionSelector_SK.show_mouse_select(sigpath, sigheader)
        loadRegion = (points[1][1], points[0][1], points[1][0], points[0][0])
        print('These are the coordinates: ' + str(loadRegion))

    pairs = find_tif_file_pairs(sigpath, expected_interval_ms=expected_interval_ms, tolerance_ms=tolerance_ms)
    print('number of pairs: {:.0f}'.format(len(pairs)))

    sigpathslist, bgpathslist = [], []
    for file1, file2 in pairs:
        sigpathslist.append(os.path.join(sigpath, file1))
        bgpathslist.append(os.path.join(sigpath, file2))

    if len(sigpathslist) == 0:
        print("Signal interferograms not found.")
        raise SystemExit(0)

    # Load first image and bin it to get final shape
    test_img = plt.imread(sigpathslist[0]).astype('float32')
    if test_img.ndim == 3:
        test_img = test_img[..., :3].mean(axis=-1)  # convert to grayscale if RGB(A)
    test_img_binned = bin_image(test_img, bin_factor)
    binned_shape = test_img_binned.shape

    print(f"Original shape: {test_img.shape}, Binned shape: {binned_shape}")

    # Support numShots as int or sequence
    if isinstance(numShots, int):
        num_frames = min(numShots, len(pairs))
        requested_str = str(numShots)
    else:
        num_frames = min(len(numShots), len(pairs))
        requested_str = str(numShots)
    print(f"Load shots {num_frames} (requested: {requested_str}).")

    RawInterferogramssig = np.zeros((num_frames, binned_shape[0], binned_shape[1]), dtype='float32')
    RawInterferogramsbg  = np.zeros((num_frames, binned_shape[0], binned_shape[1]), dtype='float32')

    for i in range(num_frames):
        tempSig = plt.imread(sigpathslist[i]).astype('float32')
        tempBg  = plt.imread(bgpathslist[i]).astype('float32')
        if tempSig.ndim == 3:
            tempSig = tempSig[..., :3].mean(axis=-1)
        if tempBg.ndim == 3:
            tempBg = tempBg[..., :3].mean(axis=-1)
        RawInterferogramssig[i] = bin_image(tempSig, bin_factor)
        RawInterferogramsbg[i]  = bin_image(tempBg,  bin_factor)

    if loadFull:
        newRawInterferogramssig = RawInterferogramssig
        newRawInterferogramsbg  = RawInterferogramsbg
    else:
        newRawInterferogramssig = RawInterferogramssig[:, loadRegion[0]:loadRegion[1], loadRegion[2]:loadRegion[3]]
        newRawInterferogramsbg  = RawInterferogramsbg[:,  loadRegion[0]:loadRegion[1], loadRegion[2]:loadRegion[3]]

    return [newRawInterferogramssig, newRawInterferogramsbg]
