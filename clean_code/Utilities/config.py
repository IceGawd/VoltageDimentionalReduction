"""
Configuration Parameter Management Module

This module implements a singleton pattern for managing global configuration parameters
across the application. It provides a single shared dictionary that can be imported
and accessed from any part of the codebase.

Example:
    >>> from config import params
    >>> params['batch_size'] = 32
    >>> # In another module:
    >>> from config import params
    >>> print(params['batch_size'])  # Outputs: 32

Note:
    - The params dictionary is shared across all imports of this module
    - Keys can be added or modified at runtime
    - Use string keys for better maintainability
    - Consider using type hints when accessing values:
        batch_size: int = params['batch_size']
"""

from typing import Dict, Any

params = None  # type: Dict[str, Any]
