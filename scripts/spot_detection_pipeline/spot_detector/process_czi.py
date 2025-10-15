else:  # file_type == 'czi'
    # Process CZI files
    for condition, file_list in condition_groups.items():
        print(f"\n{'='*50}")
        print(f"Processing {condition}...")
        print(f"{'='*50}")
        
        for file_path, field_num in tqdm(file_list, desc=f"Processing {condition}"):
            print(f"\n  Field of view {field_num}")
            
            try:
                # Load CZI file - adjust channel numbers as needed
                czi = CziFile(file_path)
                
                seg_image, seg_info = czi.read_image(C=channels_params['brightfield'])
                spots_image, spots_info = czi.read_image(C=channels_params['spots'])
                
                seg_image = np.squeeze(seg_image)
                spots_image = np.squeeze(spots_image)
                
                seg_image = np.std(seg_image, axis=0)
                del seg_info
    # Cell 7: Save results to files
                spots_image = np.std(spots_image, axis=0)
                del spots_info
                
                img = AICSImage(file_path)
                pixel_size_um = img.physical_pixel_sizes.X
                
                # Common processing starts here (same as ND2)
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
                        'Image_Number': field_num
                    })
                
                results.extend(roi_results)
                
                # Create QC figure
                qc_figure_path = figures_folder / f"{condition}_image_{field_num:03d}_QC.png"
                qc = MicroscopyQC(figsize=(10, 8), dpi=300)
                qc.create_qc_figure(segmentation_image=seg_image,
                                    spot_std=spots_image,
                                    coordinates=coords_spotiflow,
                                    corrected_spots=corrected_spots,
                                    masks=masks,
                                    flow_details=details.flow,
                                    condition=condition,
                                    image_num=field_num,
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
                print(f"❌ Error processing {file_path.name}: {str(e)}")
                warnings.warn(f"Failed to process {file_path.name}: {str(e)}")

print(f"\n🎉 Processing complete! Analyzed {len(results)} ROIs total.")