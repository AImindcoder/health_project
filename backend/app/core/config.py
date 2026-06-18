from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def backend_dir(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def models_dir(self) -> Path:
        return self.backend_dir / "models"

    @property
    def uploads_dir(self) -> Path:
        return self.backend_dir / "uploads"

    @property
    def risk_model_path(self) -> Path:
        return self.models_dir / "risk_model.pkl"

    @property
    def label_encoder_path(self) -> Path:
        return self.models_dir / "label_encoder.pkl"

    @property
    def feature_schema_path(self) -> Path:
        return self.models_dir / "feature_schema.json"

    @property
    def trend_encoder_path(self) -> Path:
        return self.models_dir / "trend_encoder.pkl"

    @property
    def groq_api_key_available(self) -> bool:
        return bool(self.groq_api_key) and self.groq_api_key != "gsk_YourKeyHere"


settings = Settings()
