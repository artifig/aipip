# Logic for loading configuration will go here
from functools import lru_cache
from .models import Settings

@lru_cache()
def load_settings() -> Settings:
    """Loads the application settings using pydantic-settings.

    Uses lru_cache to ensure settings are loaded only once.
    """
    # Pydantic-settings handles loading from .env and environment variables automatically
    return Settings() 