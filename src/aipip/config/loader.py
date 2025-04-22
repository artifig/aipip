# Logic for loading configuration will go here
from functools import lru_cache
from .models import AppConfig

@lru_cache()
def load_config() -> AppConfig:
    """Loads the application config using pydantic-settings.

    Uses lru_cache to ensure config is loaded only once.
    """
    # Pydantic-settings handles loading from .env and environment variables automatically
    return AppConfig() 