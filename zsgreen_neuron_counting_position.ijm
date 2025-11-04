// ImageJ Macro: ZsGreen+ Neuron Counting and Position Analysis in CA3
//
// This macro performs comprehensive analysis of ZsGreen+ neurons in the CA3 region:
// 1. Counts ZsGreen+, mCherry+, and DAPI+ cells
// 2. Calculates infection ratios (mCherry+/DAPI+) and expression ratios (ZsGreen+/mCherry+, ZsGreen+/DAPI+)
// 3. Extracts normalized positions of ZsGreen+ neurons along the CA3 axis
//
// Expected directory structure:
//   - Parent directory contains experiment folders (treatment groups)
//   - Each experiment folder contains mouse folders
//   - Each mouse folder contains image files named as: ID_Slide_Section_*.tif
//
// Image requirements:
//   - Multi-channel images (C1: mCherry, C2: ZsGreen, C3: DAPI)
//   - Images are rotated 270° and optionally flipped for consistent orientation
//
// Output:
//   - CSV file with columns: ID, Treatment, Slide, Section, cell counts (ZsGreen+, mCherry+, DAPI+),
//     infection ratio, expression ratios, followed by normalized X and Y positions of neurons
//
// Author: Dario Cupolillo

setOption("ExpandableArrays", true);

// Get input directory and file list
dir = getDirectory("Select parent directory containing experiment folders");
exp_list = getFileList(dir);

// Get output file path
output_dir = getDirectory("Select directory for output file");
output_path = output_dir + "neuron_counting_position.csv";

// Initialize CSV header
print("ID,Treatment,Slide,Section,ZsGreen+,mCherry+,DAPI+,Infection_ratio(%),ZsGreen/mCherry(%),ZsGreen/DAPI(%),CA3_axis_length,Neuron_positions_X,Neuron_positions_Y\n");

// Initialize arrays for storing counts
DAPI_total = newArray();
ZsGreen_total = newArray();
mCherry_total = newArray();
infected_total = newArray();
ratio_zsgreen_mcherry = newArray();
ratio_zsgreen_dapi = newArray();

// Loop through experiment folders (treatment groups)
for (j = 0; j < exp_list.length; j++) {
    split_name = split(exp_list[j], "/");
    treatment = split_name[0];
    
    mouse_list = getFileList(dir + exp_list[j]);
    
    // Loop through mouse folders
    for (k = 0; k < mouse_list.length; k++) {
        file_list = getFileList(dir + exp_list[j] + mouse_list[k]);
        
        // Loop through image files
        for (i = 0; i < file_list.length; i++) {
            
            lines = newArray();
            
            // Parse filename for metadata
            info = split(file_list[i], "_");
            
            mouse_id = info[0];
            slide_number = info[1];
            section_id = info[2];
            
            // Initialize workspace
            run("Close All");
            roiManager("Reset");
            run("Set Measurements...", "area area_fraction display redirect=None decimal=3");
            run("Clear Results");
            
            // Process only files with 4 underscore-separated parts
            if (lengthOf(info) == 4) {
                
                // Open image
                open(dir + exp_list[j] + mouse_list[k] + file_list[i]);
                img = getTitle();
                
                // Set threshold values based on bit depth
                bitdepth = bitDepth();
                if (bitdepth == 16) {
                    threshold_start = 8000;
                    threshold_end = 65535;
                    mcherry_threshold = 1500;
                } else {
                    if (bitdepth < 12) {
                        threshold_start = 500;
                        threshold_end = 4096;
                        mcherry_threshold = 300;
                    } else {
                        exit("Unsupported bit depth: " + bitdepth);
                    }
                }
                
                // Rotate image for consistent orientation
                run("Rotate... ", "angle=270 enlarge");
                
                // Flip horizontally if section is from right hemisphere
                if (substring(section_id, 2) == 'R') {
                    run("Flip Horizontally", "stack");
                }
                
                // Split and rename channels
                selectWindow(img);
                run("Split Channels");
                
                selectWindow("C1-" + file_list[i]);
                rename(mouse_id + "-" + section_id + "-" + "mCherry");
                channel_red = getTitle();
                
                selectWindow("C2-" + file_list[i]);
                rename(mouse_id + "-" + section_id + "-" + "ZsGreen");
                channel_green = getTitle();
                
                selectWindow("C3-" + file_list[i]);
                rename(mouse_id + "-" + section_id + "-" + "DAPI");
                channel_blue = getTitle();
                
                run("Tile");
                
                // Create template for ROI drawing (merge mCherry and DAPI)
                selectWindow(channel_blue);
                run("Duplicate...", " ");
                selectWindow(channel_red);
                run("Duplicate...", " ");
                run("Merge Channels...", "c1=" + channel_red + " c4=" + channel_blue);
                selectWindow("RGB");
                rename(mouse_id + "-" + section_id + "-" + "mCherry-DAPI");
                run("Maximize");
                
                // Manual ROI selection for CA3 region
                run("Select None");
                setTool("polygon");
                waitForUser("Draw the CA3 region");
                
                // Confirm ROI selection
                confirmed = getBoolean("Is the CA3 ROI correct?");
                while (confirmed == 0) {
                    setTool("polygon");
                    waitForUser("Draw the CA3 region");
                    confirmed = getBoolean("Is the CA3 ROI correct?");
                }
                
                // Save CA3 ROI
                roiManager("Add");
                roiManager("select", 0);
                roiManager("Rename", "CA3_area");
                
                // ===== Process DAPI channel =====
                selectWindow(channel_blue + "-1");
                roiManager("Select", 0);
                
                // Enhance contrast and apply filters
                run("Enhance Contrast...", "saturated=5");
                run("Gaussian Blur...", "sigma=2");
                run("Median...", "radius=3");
                run("Enhance Contrast...", "saturated=5");
                
                // Auto-threshold
                setAutoThreshold("Huang");
                setOption("BlackBackground", true);
                run("Convert to Mask");
                run("Options...", "iterations=1 count=1 do=Nothing");
                run("Open");
                run("Watershed");
                
                // Count DAPI+ cells
                selectWindow(channel_blue + "-1");
                run("Clear Results");
                roiManager("Select", 0);
                run("Analyze Particles...", "size=35-Infinity show=Masks summarize in_situ");
                run("Create Selection");
                wait(500);
                
                roiManager("Add");
                roiManager("Select", 4);
                roiManager("Rename", "DAPI_mask_total");
                DAPI_total[i] = nResults();
                
                // Close summary and clear results
                if (isOpen("Summary")) {
                    selectWindow("Summary");
                    run("Close");
                }
                if (isOpen("Results")) {
                    run("Clear Results");
                }
                
                selectWindow(channel_blue + "-1");
                close();
                
                // ===== Process mCherry channel =====
                selectWindow(channel_red + "-1");
                run("Maximize");
                run("Median...", "radius=3");
                
                // Apply DAPI mask and threshold
                selectWindow(channel_red + "-1");
                roiManager("Select", 4);
                run("Clear Outside");
                run("Select None");
                call("ij.plugin.frame.ThresholdAdjuster.setMode", "B&W");
                setThreshold(mcherry_threshold, 65535);
                run("Convert to Mask");
                
                // Count mCherry+ cells
                run("Analyze Particles...", "size=60-Infinity show=[Bare Outlines] summarize in_situ");
                wait(500);
                mCherry_total[i] = nResults() - DAPI_total[i];
                
                if (isOpen("Summary")) {
                    selectWindow("Summary");
                    run("Close");
                }
                if (isOpen("Results")) {
                    run("Clear Results");
                }
                
                // ===== Process ZsGreen channel for counting =====
                selectWindow(channel_green);
                run("Duplicate...", " ");
                rename(channel_green + "-2");
                
                selectWindow(channel_green);
                run("Maximize");
                run("Median...", "radius=2");
                setThreshold(threshold_start, threshold_end);
                setOption("BlackBackground", true);
                run("Convert to Mask");
                
                // Count ZsGreen+ cells
                roiManager("Select", 0);
                run("Analyze Particles...", "size=45-Infinity circularity=0.3-1 show=Nothing summarize");
                wait(500);
                ZsGreen_total[i] = nResults() - DAPI_total[i] - mCherry_total[i];
                
                if (isOpen("Summary")) {
                    selectWindow("Summary");
                    run("Close");
                }
                if (isOpen("Results")) {
                    run("Clear Results");
                }
                
                // ===== Process ZsGreen channel for position analysis =====
                selectWindow(channel_green + "-2");
                run("Maximize");
                run("Median...", "radius=2");
                
                // Clear outside CA3 region
                roiManager("Select", 0);
                run("Clear Outside");
                
                // Threshold and create mask
                setThreshold(threshold_start, threshold_end);
                setOption("BlackBackground", true);
                run("Convert to Mask");
                run("Open");
                
                // Create selection from mask
                run("Create Selection");
                wait(500);
                
                // Check if neurons detected
                type = selectionType();
                if (type == -1) {
                    empty = newArray();
                    lines = Array.concat(lines, empty);
                    Array.print(lines);
                    waitForUser("No neurons detected", "No ZsGreen+ neurons found in this section");
                    continue;
                }
                
                // Save ZsGreen mask
                roiManager("Add");
                roiManager("Select", 1);
                roiManager("Rename", "ZsGreen_mask");
                
                // Detect individual neurons
                run("Analyze Particles...", "size=45-Infinity circularity=0.3-1 show=Nothing clear record");
                wait(500);
                
                // ===== Draw CA3 axis =====
                selectWindow(mouse_id + "-" + section_id + "-" + "mCherry-DAPI");
                run("Select None");
                setTool("polyline");
                waitForUser("Draw CA3 axis from CA2 to DG");
                
                // Confirm axis
                confirmed = getBoolean("Is the CA3 axis correct?");
                while (confirmed == 0) {
                    setTool("polyline");
                    waitForUser("Draw CA3 axis from CA2 to DG");
                    confirmed = getBoolean("Is the CA3 axis correct?");
                }
                
                roiManager("Add");
                roiManager("Select", 2);
                roiManager("Rename", "CA3_axis");
                
                // ===== Straighten CA3 axis =====
                selectWindow(channel_green + "-2");
                roiManager("Select", 2);
                setLineWidth(165);
                run("Straighten...");
                setLineWidth(165);
                run("Flip Horizontally");
                wait(500);
                
                // Get straightened image dimensions and calculate axis length
                getDimensions(im_width, im_height, im_channels, im_slices, im_frames);
                getPixelSize(unit, pw, ph, pd);
                axis_length = im_width * pw;
                
                // Clear ROIs
                roiManager("Deselect");
                roiManager("Delete");
                
                // ===== Create new neuronal mask from straightened image =====
                run("Convert to Mask");
                run("Open");
                run("Watershed");
                wait(500);
                run("Create Selection");
                wait(500);
                
                roiManager("Add");
                roiManager("Select", 0);
                roiManager("Rename", "Straightened_neurons");
                
                // Detect neurons in straightened image
                run("Analyze Particles...", "size=45-Infinity circularity=0.3-1 show=Nothing clear record");
                wait(500);
                
                // Create individual neuron ROIs
                for (b = 0; b < nResults; b++) {
                    x = getResult('XStart', b);
                    y = getResult('YStart', b);
                    doWand(x, y);
                    roiManager("add");
                    roiManager("Select", b + 1);
                    roiManager("Rename", "straight_neuron_" + (b + 1));
                }
                
                // ===== Extract neuron centroids =====
                nROIs = roiManager("count");
                straight_ROI_centroids_x = newArray();
                straight_ROI_centroids_y = newArray();
                getPixelSize(unit, pw, ph, pd);
                
                d = 0;
                for (b = 1; b < nROIs; b++) {
                    roiManager("Select", b);
                    roiManager("Measure");
                    straight_ROI_centroids_x[d] = getResult("X", d) / pw;
                    straight_ROI_centroids_y[d] = getResult("Y", d) / ph;
                    d++;
                }
                
                // Normalize positions to 0-1 scale
                getDimensions(width, height, channels, slices, frames);
                neuron_position_x = newArray();
                neuron_position_y = newArray();
                
                for (b = 0; b < straight_ROI_centroids_x.length; b++) {
                    neuron_position_x[b] = straight_ROI_centroids_x[b] / width;
                    neuron_position_y[b] = straight_ROI_centroids_y[b] / height;
                }
                
                // Sort positions from CA2 to DG
                neuron_position_x = Array.sort(neuron_position_x);
                neuron_position_y = Array.sort(neuron_position_y);
                
                // Concatenate position data
                lines = Array.concat(neuron_position_x, neuron_position_y);
                
                // ===== Calculate ratios =====
                infected_total[i] = mCherry_total[i] / DAPI_total[i] * 100;
                ratio_zsgreen_mcherry[i] = ZsGreen_total[i] / mCherry_total[i] * 100;
                ratio_zsgreen_dapi[i] = ZsGreen_total[i] / DAPI_total[i] * 100;
                
                // Output results
                print(mouse_id + "," + treatment + "," + slide_number + "," + section_id + "," + 
                      ZsGreen_total[i] + "," + mCherry_total[i] + "," + DAPI_total[i] + "," + 
                      infected_total[i] + "," + ratio_zsgreen_mcherry[i] + "," + ratio_zsgreen_dapi[i] + "," +
                      axis_length);
                Array.print(lines);
                print("\n");
            }
        }
    }
}

// Save output
selectWindow("Log");
saveAs("txt", output_path);
exit();
