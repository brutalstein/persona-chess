class PersonaChessError(Exception):
    """Base exception for package-level errors."""


class PgnReadError(PersonaChessError):
    """Raised when a PGN file cannot be read."""


class PlayerNotFoundError(PersonaChessError):
    """Raised when no games are found for the requested player."""


class ModelNotFittedError(PersonaChessError):
    """Raised when inference is requested before fitting a model."""


class ArtifactError(PersonaChessError):
    """Raised when a persona artifact cannot be loaded or saved."""


class OptionalDependencyError(PersonaChessError):
    """Raised when an optional backend dependency is not installed."""
