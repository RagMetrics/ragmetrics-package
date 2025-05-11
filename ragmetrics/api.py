"""
RagMetrics API module that provides the main client interface.
Core error types and utility functions for the RagMetrics package.
"""

import logging
import time
import uuid
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# --- Custom Exceptions --- 
class RagMetricsError(Exception):
    """Base exception class for RagMetrics errors."""
    pass

class RagMetricsConfigError(RagMetricsError):
    """Exception for configuration-related errors."""
    pass

class RagMetricsHttpError(RagMetricsError):
    """Base exception for HTTP-related errors (auth and API errors)."""
    def __init__(self, message, status_code=None, response_text=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text

    def __str__(self):
        s = super().__str__()
        if self.status_code: 
            s += f" (Status Code: {self.status_code})"
        if hasattr(self, 'response_text') and self.response_text:
            s += f" Response: {self.response_text[:200]}..." 
        return s

# Create specific HTTP error types as subclasses
class RagMetricsAuthError(RagMetricsHttpError): pass
class RagMetricsAPIError(RagMetricsHttpError): pass

def _serialize_default(o: Any) -> Any:
    """Safe serializer for common non-serializable types."""
    # Handle Pydantic models and similar objects with model_dump or dict methods
    for method_name in ("model_dump", "dict"):
        if hasattr(o, method_name) and callable(getattr(o, method_name)):
            try: 
                return getattr(o, method_name)()
            except Exception:
                pass  # Try next method if this one fails
    
    # Handle common types
    if isinstance(o, uuid.UUID) or type(o).__name__ == 'datetime': 
        return str(o)
    if isinstance(o, time.struct_time): 
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", o)
    if isinstance(o, Exception):
        return str(o)
    
    # Last resort - try repr
    try: 
        return repr(o)
    except Exception:
        return f"<unserializable:{type(o).__name__}>"

# Import the implementations from their new modules
from .client import ragmetrics_client

def login(key: Optional[str]=None, base_url: Optional[str]=None, off: bool=False) -> bool:
    """Authenticate the global RagMetrics client."""
    try:
        return ragmetrics_client.login(key=key, base_url=base_url, off=off)
    except RagMetricsError as e:
        logger.error(f"RagMetrics login failed: {e}")
        raise

def monitor(client: Any, metadata: Optional[dict]=None, 
            callback: Optional[Callable[[Any, Any], dict]]=None) -> Any:
    """Monitor an LLM client for tracing."""
    return ragmetrics_client.monitor(client, metadata, callback)
