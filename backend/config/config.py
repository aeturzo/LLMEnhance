from pydantic import BaseSettings

class Settings(BaseSettings):
    EMBEDDING_MODEL_NAME: str = "gpt4o-mini"
    EMBEDDING_VECTOR_DIM: int = 768

    class Config:
        env_file = ".env"

settings = Settings()
