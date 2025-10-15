if file_type == 'nd2':
    # Process ND2 files
    for file_path in files_to_process:
        condition = file_path.stem
        print(f"\n{'='*50}")
        print(f"Processing {condition}...")
        print(f"{'='*50}")
        
        try:
            array = nd2.imread(file_path, xarray=True, dask=True)
            with nd2.ND2File(file_path) as nd2_file:
                pixel_size_um = nd2_file.voxel_size().x
                num_positions = nd2_file.sizes['P']
                
                for p in tqdm(range(num_positions), desc=f"Processing {condition}"):
                    print(f"\n  Field of view {p+1}/{num_positions}")
                    
                    # Load and process one field of view
                    seg_image, spots_image = process_field_of_view(array, p, channels_params)
                    
                    # Common processing starts here
                    corrected_spots = skimage.morphology.white_tophat(spots_image, footprint=footprint)
                    
                    # Segmentation
                    print("    Running segmentation...")
                    masks = memory_efficient_segmentation(seg_image, segmentation_params)
                    num_rois = masks.max()
                    print(f"    Found {num_rois} gel bags")
                    
                    if num_rois == 0:
                        print("    ⚠️ No ROIs detected in this field of view, skipping analysis and QC.")
                        continue
                    
                    # Count Spots
                    coords_spotiflow, details = detect_spots_spotiflow(corrected_spots)
                    
                    # Analyze ROIs
                    roi_results, coords_spotiflow = analyze_rois_memory_efficient(masks, coords_spotiflow, pixel_size_um)
                    
                    # Add metadata to results
                    for result in roi_results:
                        result.update({
                            'Experiment': experiment_folder.name,
                            'Condition': condition,
                            'Image_Number': p + 1
                        })
                    
                    results.extend(roi_results)
                    
                    # Create QC figure
                    qc_figure_path = figures_folder / f"{condition}_image_{p+1:03d}_QC.png"
                    qc = MicroscopyQC(figsize=(10, 8), dpi=300)
                    qc.create_qc_figure(segmentation_image=seg_image,
                                        spot_std=spots_image,
                                        coordinates=coords_spotiflow,
                                        corrected_spots=corrected_spots,
                                        masks=masks,
                                        flow_details=details.flow,
                                        condition=condition,
                                        image_num=p + 1,
                                        output_path=qc_figure_path,
                                        pixel_size_um=pixel_size_um
                                    )
                    
                    total_spots = len(coords_spotiflow)
                    print(f"    Average spots detected: {total_spots / num_rois}")
                    print(f"    QC figure saved: {qc_figure_path.name}")
                    
                    # Force garbage collection
                    del seg_image, spots_image, corrected_spots, masks, roi_results, coords_spotiflow, details
                    gc.collect()
                    
        except Exception as e:
            print(f"❌ Error processing {condition}: {str(e)}")
            warnings.warn(f"Failed to process {condition}: {str(e)}")


'''
load params
detect images and types
preprocess before segmentation and detection
segmentation
detection
qc figure
save data
summary figures
'''