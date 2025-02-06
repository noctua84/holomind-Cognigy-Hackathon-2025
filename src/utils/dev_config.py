import warnings
from contextlib import contextmanager

@contextmanager
def dev_warning_filters():
    """Context manager for development warning filters"""
    with warnings.catch_warnings():
        # Configure warning filters
        warnings.filterwarnings(
            "ignore",
            category=PydanticDeprecatedSince20,
            module="mlflow.*"
        )
        
        warnings.filterwarnings(
            "ignore",
            message="Function 'semver.compare' is deprecated",
            category=PendingDeprecationWarning
        )
        
        warnings.filterwarnings(
            "ignore",
            message="Support for class-based `config` is deprecated",
            category=PydanticDeprecatedSince20
        )
        
        yield 