
##########################################################################################
# LIBRARIES:

import os
import PngIntLoader_SK_CALAsept25_separate_bg as PngIntLoader_SK_CALAsept25
import PhaseExtractor_SK as PhaseExtractor_SK
import numpy as np


##########################################################################################
# GLOBAL VARIABLES (need to be set manually)

# Default interferogram parameters
sigHeader='tif' # a string common in the names of all interferograms
numShots=list(np.arange(50)) # a list containing indices of images to actually consider (i.e. [0,2,3] will load first, third and fourth image
loadFull=True #load the full interferogram? (or do you want a selected region...?)

show_plots_flag = True #set to True to see plots of the processing steps, set False to run without interruption
FFT_window_size = 110 # size of the square window in the 2D FFT to isolate the sideband; set carefully - noise vs resolution trade-off!

# settings for fft filtering, legacy from time of small Wollaston aperture in the CALA interferogram required filtering parasitic frequencies
diag_angle = 34 # angle (degs) of the main noise diagonal in the 2D FFT
diag_dist = 300 # diagonal distance to FFT corner cut-out; if set to a number much higher than FFT_window_size, no cut-out will be applied

# manual boundary points for the signal interferogram region (y1,x1),(y2,x2); set to None to select region with mouse initially, and then use same region for all images during an automated run
#boundary_points_sig = [(1639, 804), (1698, 863)]#IR
boundary_points_sig = None

common_path = r"D:\kPAC\2025_09_05PMOPA" #portion of path common to both raw AND processed data
run_path = r'\100mbar_Bdel552_t0' #unique bit of the path to the folder for a specific processing run
sigPath_IR = common_path + run_path + r'\Interferometry2' #rest of the path to the folder containing signal-background pairs of images: 1030 nm (IR)
sigPath_Green = common_path + run_path + r'\Interferometry1' #rest of the path to the folder containing signal-background pairs of images: 515 nm (Green)
saveloc_IR = r'C:\Users\kalos\Documents\kPAC\ChannelAnalysis_CALA_Sept25\channel_analysis_sept25\phase_maps_sept25' + run_path + r'\Interferometry2' #location to save the processed data of this specific processing run
saveloc_Green = r'C:\Users\kalos\Documents\kPAC\ChannelAnalysis_CALA_Sept25\channel_analysis_sept25\phase_maps_sept25' + run_path + r'\Interferometry1' #location to save the processed data of this specific processing run
bgpath_IR = r'D:\kPAC\2025_09_09PMOPAbackgroundOnly\Interferometry2' #location of background images for 1030 nm (IR)
bgpath_Green = r'D:\kPAC\2025_09_09PMOPAbackgroundOnly\Interferometry1' #location of background images for 515 nm (Green)
ignore_rects_IR = [(130, 360, 150, 1280),  # rect 1
                    (590, 770, 600, 1280)]  # rect 2

ignore_rects_Green = [(150, 260, 0, 1280),  # rect 1
                      (530, 680, 0, 1280)]  # rect 2

for sigPath,saveloc,bgpath,ignore_rects in zip([sigPath_IR, sigPath_Green],
                                  [saveloc_IR, saveloc_Green],
                                  [bgpath_IR,  bgpath_Green],
                                  [ignore_rects_IR,ignore_rects_Green]): #loop over both wavelengths
    
    print('Now processing interferograms in folder: ' + sigPath)
    if not os.path.exists(saveloc):
        os.makedirs(saveloc) #create the folder if it doesn't exist 
    ## LOADING TIFF FILES
    
    RawInterferograms = PngIntLoader_SK_CALAsept25.PngIntLoader(sigpath=sigPath,   # all signal frames
                                                                sigheader=sigHeader,
                                                                numShots=numShots,                         # or a list / smaller number
                                                                loadFull=loadFull,
                                                                bin_factor=1,
                                                                bgpath=bgpath,
                                                                return_stats=False)  # <--- new background folder)
    # PHASE EXTRACTION


    AvgPhase, DiffCube, DiffCubeNorm, DiffCubeDetr, BgCube = PhaseExtractor_SK.PhaseExtractor(RawInterferograms=RawInterferograms,   # (sig_stack, bg_stack)
                                                                            numShots=numShots,
                                                                            saveloc=saveloc,
                                                                            boundary_points_sig=boundary_points_sig,
                                                                            show_plots_flag=show_plots_flag,
                                                                            diag_angle_deg=diag_angle,
                                                                            diag_dist=diag_dist,
                                                                            fourier_window_size=FFT_window_size,
                                                                            # paging + plots
                                                                            pairs_per_fig=10, grid_ncols=5,
                                                                            show_pages=True,          # keep UI responsive
                                                                            max_show_pages=1,          # if you flip show_pages=True later, limit windows
                                                                            show_final_avg=True,       # always pop the final average
                                                                            # background (FAST masked polynomial)
                                                                            poly_order=5,              # try 2 first; 3 if a touch more curvature remains
                                                                            poly_downsample=4,         # 6–12 is a good range
                                                                            mask_dilate_px=12,         # small guard band around ignore rects
                                                                            ignore_rects=ignore_rects,
                                                                            # outputs
                                                                            save_avg_plot=True,
                                                                            return_cubes=False         # set True only if you really need the stacks
                                                                        )
