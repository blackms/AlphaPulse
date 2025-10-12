"""
Common service-layer exceptions.
"""


class ServiceConfigurationError(RuntimeError):
    """Raised when a service is missing required dependencies or configuration."""

    def __init__(self, message: str):
        super().__init__(message)
