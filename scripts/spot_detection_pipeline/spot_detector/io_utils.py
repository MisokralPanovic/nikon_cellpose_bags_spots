'''
this should be a function
'''

def find_raw_files(raw_data_folder):
    # Find all .czi and .nd2 files 
    nd2_files = list(raw_data_folder.glob('*.nd2'))
    czi_files = list(raw_data_folder.glob('*.czi'))

    # Check what files we have
    total_files = len(nd2_files) + len(czi_files)
    if total_files == 0:
        raise FileNotFoundError("No .nd2 or .czi files found in raw_data folder")

    # Determine file type and print summary
    if nd2_files and czi_files:
        raise ValueError("Mixed file types found. Please process .nd2 and .czi files separately.")
    
    elif nd2_files:
        file_type = 'nd2'
        files_to_process = nd2_files
        print(f"Found {len(nd2_files)} .nd2 files to process:")
        for file_path in nd2_files:
            print(f"  - {file_path.name}")
    
    elif czi_files:
        file_type = 'czi'
        files_to_process = czi_files
        print(f"Found {len(czi_files)} .czi files to process:")
        for file_path in czi_files:
            print(f"  - {file_path.name}")
    
    return file_type, files_to_process


'''
this should be a function
'''

def process_field_of_view_nd2(array, p, channels_params):
    seg_image = array.isel(P=p, C=channels_params['brightfield'])
    spots_image = array.isel(P=p, C=channels_params['spots'])
    
    return seg_image, spots_image

def process_field_of_view_czi(array, p, channels_params):
    czi = CziFile(file_path)
    seg_image, seg_info = czi.read_image(C=channels_params['brightfield'])
    spots_image, spots_info = czi.read_image(C=channels_params['spots'])
    
    del seg_info
    del spots_info
    
    return seg_image, spots_image

# Group CZI files by condition if needed
if file_type == 'czi':
    condition_groups = {}
    for file_path in files_to_process:
        filename = file_path.stem
        parts = filename.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            condition = '_'.join(parts[:-1])
            field_num = int(parts[-1])
        else:
            condition = filename
            field_num = 1
        
        if condition not in condition_groups:
            condition_groups[condition] = []
        condition_groups[condition].append((file_path, field_num))
    
    # Sort files within each condition by field number
    for condition in condition_groups:
        condition_groups[condition].sort(key=lambda x: x[1])

results = []




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
                