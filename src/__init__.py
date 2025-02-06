"""HoloMind - A Continual Learning Framework"""

__version__ = "0.1.0"

import warnings
from pydantic import PydanticDeprecatedSince20

def setup_warning_filters():
    """Setup warning filters for the application"""
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

# Call this during application initialization
setup_warning_filters() 