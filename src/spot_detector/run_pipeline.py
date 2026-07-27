from bioio import BioImage
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from spot_detector.utils import parse_condition_from_name, ModelBundle
from spot_detector.segmentation_detection import (
    segment_2d,
    segment_3d,
    detect_spots_spotiflow,
    assign_spots_to_mask,
)
from spot_detector.obejct_measurement import measure_objects
from spot_detector.qc_figures import (
    make_run_summary_figure,
    make_scene_summary_figure,
    make_qc_figure,
)

logger = logging.getLogger(__name__)


def run_pipeline(config: dict) -> pd.DataFrame | None:
    """Run the full segmentation + spot detection pipeline.

    Args:
        config (dict): Pipeline configuration, matching config.yml schema.

    Returns:
        pd.DataFrame | None: Combined run-level results, or None if no files processed.
    """
    # define dim mode
    do_3d = config["mode"]["do_3d"]
    mode = "3d" if do_3d else "2d"

    # establish folder structure
    data_folder = Path(config["paths"]["raw_data_dir"]).resolve()
    experiment = data_folder.parent.name
    out_dir = Path(config["paths"]["out_dir"])
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Pipeline starting | mode={mode.upper()} ===")
    logger.info(f"Data folder: {data_folder}")

    models = ModelBundle.load(config=config, do_3d=do_3d)

    all_run_records = []
    failures = []

    file_list = [p for p in data_folder.iterdir() if p.is_file()]

    for filepath in file_list:
        try:
            scene_df = _process_file(
                filepath=filepath,
                config=config,
                models=models,
                mode=mode,
                do_3d=do_3d,
                experiment=experiment,
                fig_dir=fig_dir,
                tab_dir=tab_dir,
                failures=failures,
            )
            if scene_df is not None:
                all_run_records.append(scene_df)

        except Exception as e:
            logger.exception(f"Processing file {filepath.name} failed ({e}), skipping")
            failures.append(
                {
                    "Experiment": experiment,
                    "Source_File": filepath.name,
                    "Condition": parse_condition_from_name(filepath.stem),
                    "Scene": np.nan,
                    "Error": str(e),
                    "Error_Type": type(e).__name__,
                }
            )

    if failures:
        failures_df = pd.DataFrame(failures)
        run_failures_csv_path = tab_dir / f"_run_failures_{mode.upper()}.csv"
        failures_df.to_csv(run_failures_csv_path, index=False)
        logger.info(
            f"Saved run failures CSV: {run_failures_csv_path.name} ({len(failures_df)} rows)"
        )

    if not all_run_records:
        return None

    run_df = pd.concat(all_run_records, ignore_index=True)
    run_csv_path = tab_dir / f"_run_objects_{mode.upper()}.csv"
    run_df.to_csv(run_csv_path, index=False)
    logger.info(
        f"Saved run CSV: {run_csv_path.name} ({len(run_df)} rows, {run_df['Condition'].nunique()} condition(s))"
    )

    make_run_summary_figure(
        df=run_df,
        experiment=experiment,
        mode=mode,
        out_path=fig_dir / f"_run_summary_{mode.upper()}.png",
    )
    return run_df


def _process_file(
    filepath: Path,
    config: dict,
    models: ModelBundle,
    mode: str,
    do_3d: bool,
    experiment: str,
    fig_dir: Path,
    tab_dir: Path,
    failures: list,
) -> pd.DataFrame | None:
    """Process a single multi-scene image file. Returns combined scene DataFrame or None."""
    condition = parse_condition_from_name(filepath.stem)
    img = BioImage(filepath)

    all_scene_records = []

    logger.info(f"--- Processing: {filepath.name} ---")

    num_scenes = len(img.scenes)
    logger.info(f"Scenes: {num_scenes}")

    with logging_redirect_tqdm():
        for scene in tqdm(range(num_scenes)):
            try:
                img.set_scene(scene)
                logger.debug(f"Scene {scene:02d} / {num_scenes - 1}")
                scene_df = _process_scene(
                    img=img,
                    scene=scene,
                    filepath=filepath,
                    condition=condition,
                    config=config,
                    models=models,
                    mode=mode,
                    do_3d=do_3d,
                    experiment=experiment,
                    fig_dir=fig_dir,
                )
                if scene_df is not None:
                    all_scene_records.append(scene_df)

            except Exception as e:
                logger.exception(
                    f"Scene {scene} on {filepath.name} ({condition}) failed ({e}), skipping"
                )
                failures.append(
                    {
                        "Experiment": experiment,
                        "Source_File": filepath.name,
                        "Condition": condition,
                        "Scene": scene,
                        "Error": str(e),
                        "Error Type": type(e).__name__,
                    }
                )

    if not all_scene_records:
        return None

    combined_df = pd.concat(all_scene_records, ignore_index=True)

    csv_path = tab_dir / f"{condition}_objects_{mode.upper()}.csv"
    combined_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path.name}  ({len(combined_df)} rows)")

    make_scene_summary_figure(
        df=combined_df,
        condition=condition,
        mode=mode,
        out_path=fig_dir / f"{condition}_summary_{mode.upper()}.png",
    )
    return combined_df


def _process_scene(
    img: BioImage,
    *,
    config: dict,
    do_3d: bool,
    models: ModelBundle,
    mode: str,
    condition: str,
    filepath: Path,
    experiment: str,
    scene: int,
    fig_dir: Path,
) -> pd.DataFrame:
    """Process a single scene: segment, detect, measure, and produce QC figure."""
    dim_order = "YX" if "Z" not in img.dims.order else "ZYX"
    objects_stack = img.get_image_data(
        dim_order, C=config["channels"]["segmentation_image"]
    ).astype(np.float32)
    spots_stack = img.get_image_data(
        dim_order, C=config["channels"]["spot_image"]
    ).astype(np.float32)

    dx = img.physical_pixel_sizes.X or 1.0
    dz = img.physical_pixel_sizes.Z or 1.0

    logger.debug(f"Segmenting ({mode.upper()})...")
    if do_3d:
        masks = segment_3d(
            bf_stack=objects_stack,
            model_cellpose=models.cellpose,
            factor=config["segmentation"]["bin_factor"],
            stitch_threshold=config["segmentation"]["stitch_threshold"],
        )
    else:
        masks = segment_2d(
            bf_stack=objects_stack,
            model_cellpose=models.cellpose,
            factor=config["segmentation"]["bin_factor"],
        )

    n_obj = len(np.unique(masks)) - 1
    logger.info(f"Found {n_obj} object(s) after border clearing")

    logger.debug(f"Detecting spots ({mode.upper()})...")
    points, details = detect_spots_spotiflow(
        spot_stack=spots_stack,
        model_spotiflow=models.spotiflow,
        prob_thresh=config["detection"]["prob_thresh"],
        min_distance=config["detection"]["min_distance"],
        do_3d=do_3d,
    )
    spot_labels = assign_spots_to_mask(coordinates=points, masks=masks)
    logger.info(f"Detected {len(points)} spot(s), {(spot_labels > 0).sum()} assigned")

    scene_df = measure_objects(
        masks=masks,
        spot_labels=spot_labels,
        dx=dx,
        dz=dz,
        mode=mode,
        condition=condition,
        filepath=filepath.name,
        experiment=experiment,
        scene=scene,
    )

    make_qc_figure(
        condition=condition,
        scene=scene,
        mode=mode,
        out_path=fig_dir / f"{condition}_S{scene:02d}_{mode.upper()}_qc.png",
        segmentation_image=objects_stack,
        spots_image=spots_stack,
        masks=masks,
        coordinates=points,
        spot_labels=spot_labels,
        flow_details=details,
        dx=dx,
        dz=dz,
        config=config,
    )
    return scene_df
