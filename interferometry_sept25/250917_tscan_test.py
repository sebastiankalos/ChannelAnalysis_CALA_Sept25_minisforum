
##########################################################################################
# LIBRARIES:

import os
import PngIntLoader_SK_CALAsept25 as PngIntLoader_SK_CALAsept25
import PhaseExtractor_SK as PhaseExtractor_SK
import numpy as np


##########################################################################################
# GLOBAL VARIABLES (need to be set manually)

# Default interferogram parameters
sigHeader='tif' # a string common in the names of all interferograms
numShots=list(np.arange(70)) # a list containing indices of images to actually consider (i.e. [0,2,3] will load first, third and fourth image
loadFull=True #load the full interferogram? (or do you want a selected region...?)

show_plots_flag = False #set to True to see plots of the processing steps, set False to run without interruption
FFT_window_size = 150 # size of the square window in the 2D FFT to isolate the sideband; set carefully - noise vs resolution trade-off!

# settings for fft filtering, legacy from time of small Wollaston aperture in the CALA interferogram required filtering parasitic frequencies
diag_angle = 34 # angle (degs) of the main noise diagonal in the 2D FFT
diag_dist = 300 # diagonal distance to FFT corner cut-out; if set to a number much higher than FFT_window_size, no cut-out will be applied

# manual boundary points for the signal interferogram region (y1,x1),(y2,x2); set to None to select region with mouse initially, and then use same region for all images during an automated run
#boundary_points_sig = None
boundary_points_IR = [(650, 307), (799, 605)]
boundary_points_Green = [(736, 490), (885, 788)]


ignore_rects_IR = [(220, 380, 0, 1280),  # rect 1
                    (680, 820, 0, 1280)]  # rect 2

ignore_rects_Green = [(330, 470, 0, 1280),  # rect 1
                      (770, 900, 0, 1280)]  # rect 2

peak_center_IR = (310, 835) #y,x coordinates of a small region in the phase map where the phase should be positive (for sign disambiguation)
peak_center_Green = (410, 835) #y,x coordinates of a small

pressures = [100,80,60,40,20] #list of pressures to loop over
B_delays = [582,560] #list of B delays to loop over
times = [1.0,2.0,2.5,3.0,3.5,4.0]

common_path = r"D:\kPAC\2025_09_17BackfillTimescan\\" #portion of path common to both raw AND processed data

for delay in B_delays: #loop over all B delays
    for pressure in pressures: #loop over all pressures
        for time in times: #loop over all times
            print('Now processing data for pressure = ' + str(pressure) + ' mbar, B-delay = ' + str(delay) + ' fs, time = ' + str(time) + ' ns')
            run_path = str(pressure)+r'mbar_Bdel'+str(delay)+r'_t'+str(time)+r'ns' #unique bit of the path to the folder for a specific processing run
            sigPath_IR = common_path + run_path + r'\Interferometry2' #rest of the path to the folder containing signal-background pairs of images: 1030 nm (IR)
            sigPath_Green = common_path + run_path + r'\Interferometry1' #rest of the path to the folder containing signal-background pairs of images: 515 nm (Green)
            saveloc_IR = r'C:\Users\kalos\Documents\kPAC\ChannelAnalysis_CALA_Sept25\channel_analysis_250917_timescan\phase_maps' + run_path + r'\Interferometry2' #location to save the processed data of this specific processing run
            saveloc_Green = r'C:\Users\kalos\Documents\kPAC\ChannelAnalysis_CALA_Sept25\channel_analysis_250917_timescan\phase_maps' + run_path + r'\Interferometry1' #location to save the processed data of this specific processing run

            for sigPath,saveloc,peak_center,ignore_rects,boundary_points_sig in zip([sigPath_IR, sigPath_Green],
                                                                [saveloc_IR, saveloc_Green],
                                                                [peak_center_IR, peak_center_Green],
                                                                [ignore_rects_IR,ignore_rects_Green],
                                                                [boundary_points_IR,boundary_points_Green]): #loop over both wavelengths
                
                print('Now processing interferograms in folder: ' + sigPath)
                if not os.path.exists(saveloc):
                    os.makedirs(saveloc) #create the folder if it doesn't exist 
                
                ## LOADING TIFF FILES
                RawInterferograms = PngIntLoader_SK_CALAsept25.PngIntLoader(sigPath,
                                                                            sigHeader,
                                                                            numShots,
                                                                            loadFull)
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
                                                                                        # simple sign fix from a tiny ROI (magenta overlay on detrended pages)
                                                                                        sign_fix=True,
                                                                                        pos_center_rc=peak_center,
                                                                                        sign_half_size=30,           # 10x10 box
                                                                                        # outputs
                                                                                        save_avg_plot=True,
                                                                                        return_cubes=False         # set True only if you really need the stacks
                                                                                    )
