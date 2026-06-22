from bioio import BioImage
import numpy as np
import pandas as pd
from pathlib import Path

# import from files
from spot_detector.config import load_config
from spot_detector.utils import parse_condition_from_name, ModelBundle
from spot_detector.segmentation_detection import segment_2d, segment_3d, detect_spots_spotiflow, assign_spots_to_mask
from spot_detector.obejct_measurement import measure_objects
from spot_detector.qc_figures import make_run_summary_figure, make_scene_summary_figure, make_qc_figure

def run_pipeline(config_path: Path) -> None:
    """_summary_

    Args:
        config (dict, optional): _description_.
    """
    config = load_config(config_path)
    
    # define dim mode
    do_3d = config["mode"]["do_3d"]
    mode = "3d" if do_3d else "2d"
    
    # establish folder structure
    data_folder = Path(config["paths"]["data"])
    out_dir = Path(config["paths"]["out_dir"])
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Pipeline starting | mode={mode.upper()} ===")
    print(f"Data folder: {data_folder}")
    
    models = ModelBundle.load(config=config, do_3d=do_3d)
    
    experiment = Path(__file__).parent.parent.name
    all_run_records = []
    
    file_list = [p for p in data_folder.iterdir() if p.is_file()]
    print(f"Found {len(file_list)} file(s): {[p.name for p in file_list]}")
    
    for filepath in file_list:
        condition = parse_condition_from_name(filepath.stem)
        source_file = filepath.name
        print(f"\n--- Processing: {filepath.name} ---")
        
        img = BioImage(filepath)
        all_scene_records = []
        
        num_scenes = len(img.scenes)
        print(f"  Scenes: {num_scenes}")
        
        
        for scene in range(num_scenes):
            img.set_scene(scene)
            print(f"  Scene {scene:02d} / {num_scenes - 1}")
            
            objects_stack = img.get_image_data("YX" if "Z" not in img.dims.order else "ZYX", C=config["channels"]["segmentation_image"]).astype(np.float32)
            spots_stack = img.get_image_data("YX" if "Z" not in img.dims.order else "ZYX", C=config["channels"]["spot_image"]).astype(np.float32)

            dx = img.physical_pixel_sizes.X or 1.0
            dz = img.physical_pixel_sizes.Z or 1.0
            
            print(f"    Segmenting ({mode})...")
            if do_3d:
                masks = segment_3d(
                    bf_stack=objects_stack, 
                    model_cellpose=models.cellpose, 
                    factor=config["segmentation"]["bin_factor"], 
                    stitch_threshold=config["segmentation"]["stitch_threshold"])
            else:
                masks = segment_2d(
                    bf_stack=objects_stack, 
                    model_cellpose=models.cellpose, 
                    factor=config["segmentation"]["bin_factor"])

            n_obj = len(np.unique(masks)) -1
            print(f"    Found {n_obj} object(s) after border clearing")
            
            print(f"    Detecting spots ({mode})...")
            points, details = detect_spots_spotiflow(
                spot_stack=spots_stack,
                model_spotiflow=models.spotiflow,
                prob_thresh=config["detection"]["prob_thresh"],
                min_distance=config["detection"]["min_distance"],
                do_3d=do_3d
            )

            spot_labels = assign_spots_to_mask(coordinates=points, masks=masks)
            print(f"    Detected {len(points)} spot(s), {(spot_labels > 0).sum()} assigned")
            
            scene_df = measure_objects(
                masks=masks,
                spot_labels=spot_labels,
                dx=dx,
                dz=dz,
                mode=mode,
                condition=condition,
                source_file=source_file,
                experiment=experiment,
                scene=scene,
            )
            all_scene_records.append(scene_df)
            
            print("    Generating QC figure...")
            qc_path = fig_dir / f"{condition}_S{scene:02d}_{mode}_qc.png"
            make_qc_figure(
                condition=condition,
                scene=scene,
                mode=mode,
                out_path=qc_path,
                segmentation_image=objects_stack,
                spots_image=spots_stack,
                masks=masks,
                coordinates=points,
                spot_labels=spot_labels,
                flow_details=details,
                dx=dx,
                dz=dz,
                config=config
            )
            
        if all_scene_records:
            combined_df = pd.concat(all_scene_records, ignore_index=True)
            csv_path = tab_dir / f"{condition}_objects_{mode}.csv"
            combined_df.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path.name}  ({len(combined_df)} rows)")
            all_run_records.append(combined_df)
            
            print("  Generating summary figure...")
            summary_path = fig_dir / f"{condition}_summary_{mode}.png"
            make_scene_summary_figure(
                df = combined_df, 
                condition = condition, 
                mode = mode, 
                out_path = summary_path)

    if all_run_records:
        run_df = pd.concat(all_run_records, ignore_index=True)
        run_csv_path = tab_dir / f"_run_objects_{mode}.csv"
        run_df.to_csv(run_csv_path, index=False)
        print(f"\nSaved run CSV: {run_csv_path.name}  ({len(run_df)} rows, {run_df['Condition'].nunique()} condition(s))")

        print("Generating run summary figure...")
        run_summary_path = fig_dir / f"_run_summary_{mode}.png"
        make_run_summary_figure(
            df = run_df,
            experiment = experiment, 
            mode = mode, 
            out_path = run_summary_path)