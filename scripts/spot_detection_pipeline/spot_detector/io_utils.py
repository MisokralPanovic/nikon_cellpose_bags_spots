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