class SpotDetectorError(Exception):
    """Root for all exceptions defined explicitly by this package."""


class FatalPipelineError(SpotDetectorError):
    """Indicates a structural mismatch that will recur identically on every item in the run - never catch-and-continue this; let it abort the run."""


class DimensionMismatchError(FatalPipelineError):
    """do_3d disagrees with the actual data dimentionality."""

    def __init__(self, message: str, *, expected_ndim: int, actual_ndim: int) -> None:
        super().__init__(message)
        self.expected_ndim = expected_ndim
        self.actual_ndim = actual_ndim
