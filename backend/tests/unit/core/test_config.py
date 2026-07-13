import os

from app.core.config import Settings, get_settings


def test_default_settings() -> None:
    get_settings.cache_clear()
    settings = Settings()
    assert settings.app_name == "SchemaLens"
    assert settings.backend_cors_origins == ["http://localhost:3000"]


def test_environment_variable_override() -> None:
    get_settings.cache_clear()
    os.environ["APP_NAME"] = "TestApp"
    settings = Settings()
    assert settings.app_name == "TestApp"
    os.environ.pop("APP_NAME")


def test_cors_origins_comma_separated() -> None:
    get_settings.cache_clear()
    os.environ["BACKEND_CORS_ORIGINS"] = "http://localhost:8000, http://example.com"
    settings = Settings()
    assert settings.backend_cors_origins == ["http://localhost:8000", "http://example.com"]
    os.environ.pop("BACKEND_CORS_ORIGINS")


def test_cors_origins_json() -> None:
    get_settings.cache_clear()
    os.environ["BACKEND_CORS_ORIGINS"] = '["http://localhost:8000", "http://example.com"]'
    settings = Settings()
    assert settings.backend_cors_origins == ["http://localhost:8000", "http://example.com"]
    os.environ.pop("BACKEND_CORS_ORIGINS")
