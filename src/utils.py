import re

def parse_condition_from_name(filename_stem: str) -> str:
    """Extracts the base condition name from a filename stem.

    Examples:
        'Control_01'         -> 'Control'
        'Treated-DrugA_FOV3' -> 'Treated-DrugA'
        'WT_high_res'        -> 'WT_high_res'
    """
    return re.split(r'[-_]\d+$', filename_stem)[0]