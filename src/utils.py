import re
from dataclasses import dataclass, field

def parse_condition_from_name(filename_stem: str) -> str:
    """Extracts the base condition name from a filename stem.

    Examples:
        'Control_01'         -> 'Control'
        'Treated-DrugA_FOV3' -> 'Treated-DrugA'
        'WT_high_res'        -> 'WT_high_res'
    """
    return re.split(r'[-_]\d+$', filename_stem)[0]

####### Have a look at this and finish it!!

@dataclass
class ModelBundle:
    cellpose: models.CellposeModel
    spotiflow: Spotiflow

    @classmethod
    def load(cls, config: dict, do_3d: bool) -> "ModelBundle":
        cellpose = cls._load_cellpose(config)
        spotiflow = cls._load_spotiflow(config, do_3d)
        return cls(cellpose=cellpose, spotiflow=spotiflow)

    @staticmethod
    def _load_cellpose(config: dict) -> models.CellposeModel:
        print("Loading Cellpose model...")
        return models.CellposeModel(
            gpu=True,
            pretrained_model=str(config["cellpose_models_path"])
        )

    @staticmethod
    def _load_spotiflow(config: dict, do_3d: bool) -> Spotiflow:
        print("Loading Spotiflow model...")
        model = ModelBundle._load_spotiflow_from_config(config, do_3d)
        model = ModelBundle._validate_spotiflow_mode(model, do_3d)
        return model

    @staticmethod
    def _load_spotiflow_from_config(config: dict, do_3d: bool) -> Spotiflow:
        try:
            return Spotiflow.from_folder(str(config["spotiflow_models_path"]))
        except Exception as e:
            fallback = "smfish_3d" if do_3d else "synth_complex"
            print(f"Custom model failed ({e}), falling back to '{fallback}'...")
            return Spotiflow.from_pretrained(fallback)

    @staticmethod
    def _validate_spotiflow_mode(model: Spotiflow, do_3d: bool) -> Spotiflow:
        model_is_3d = model.config.is_3d
        if model_is_3d == do_3d:
            return model  # no conflict

        mode_str = "3D" if do_3d else "2D"
        fallback = "smfish_3d" if do_3d else "synth_complex"
        print(f"Mode conflict: model is {'3D' if model_is_3d else '2D'} "
              f"but pipeline is {mode_str}. Overriding with '{fallback}'...")
        return Spotiflow.from_pretrained(fallback)