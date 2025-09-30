from typing import Optional
from charter_compiler.executor.triton_client_production import ProductionTritonClient

# Global client instance (singleton pattern)
_default_client: Optional[ProductionTritonClient] = None

def init(url: str = "localhost:8001", model_name: str = "ensemble"):
    """Initialize the default Charter client."""
    global _default_client
    _default_client = ProductionTritonClient(url=url, model_name=model_name)
    return _default_client

def infer_with_cache(prompt: str, cache_id: Optional[int] = None, **kwargs):
    """
    Simple inference function using the default client.
    
    Usage:
        >>> import charter_compiler as charter
        >>> charter.init(url="localhost:8001")
        >>> result = charter.infer_with_cache("Hello", cache_id=1)
    """
    if _default_client is None:
        raise RuntimeError("Client not initialized. Call charter.init() first.")
    
    return _default_client.infer_with_cache(prompt, cache_id=cache_id, **kwargs)

def get_client() -> ProductionTritonClient:
    """Get the default client instance."""
    if _default_client is None:
        raise RuntimeError("Client not initialized. Call charter.init() first.")
    return _default_client