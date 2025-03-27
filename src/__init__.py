"""Main package initialization."""

# Check for available backends
import importlib.util
import logging

# Set up module-level logger
logger = logging.getLogger(__name__)

# Check for JAX availability
jax_spec = importlib.util.find_spec("jax")
JAX_AVAILABLE = jax_spec is not None

# Configure JAX at startup if available
if JAX_AVAILABLE:
    try:
        from .utils.jax_config import configure_jax
        configure_jax()
        logger.info("JAX configured at startup")
    except Exception as e:
        logger.warning(f"Failed to configure JAX: {e}")
        JAX_AVAILABLE = False