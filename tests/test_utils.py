import pytest
from spot_detector.utils import parse_condition_from_name


@pytest.mark.parametrize(
    "filename, stem",
    [
        # Original examples
        ("Control_01", "Control"),
        ("Treated-DrugA_FOV3", "Treated-DrugA"),
        ("WT_high_res", "WT_high_res"),
        # Numbers in the middle of the condition
        ("Condition2_01", "Condition2"),
        ("Drug-5mg-Variant_03", "Drug-5mg-Variant"),
        # Varying digit lengths
        ("Control_1", "Control"),
        ("Control_12345", "Control"),
        # Missing delimiters or numbers
        ("Control", "Control"),
        ("Control_", "Control_"),
        ("Control99", "Control99"),
        # Hyphen delimiter variation
        ("Treated-DrugA-02", "Treated-DrugA"),
        # Edge cases
        ("", ""),
        ("Control _01", "Control "),
    ],
)
def test_parse_condition_from_name(filename, stem):
    assert parse_condition_from_name(filename) == stem
