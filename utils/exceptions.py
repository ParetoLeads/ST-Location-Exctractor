"""
Custom exception classes for KMZ Location Scraper.
"""


class KMZParseError(Exception):
    """Raised when there's an error parsing a KMZ file."""
    pass


class OSMQueryError(Exception):
    """Raised when there's an error querying OpenStreetMap."""
    pass


class GPTAPIError(Exception):
    """Raised when there's an error with the GPT API."""
    pass


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class ConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass
