"""Error for a saved model that does not exist."""


class ModelNotFoundError(LookupError):
    """Raised when a saved model is requested but has not been trained yet."""
