# from pydantic_settings import BaseSettings , SettingsConfigDict

# class Setting (BaseSettings):
#     DB_HOST :str 
#     DB_PORT : str 
#     DB_USER : str
#     DB_PASSWORD : str
#     DB_NAME : str

#     model_config = SettingsConfigDict(env_file=".env",env_file_encoding="utf-8")

#     @property
#     def DB_URL(self):
#         return f"mysql+asyncmy://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

#     @property
#     def SYNC_DB_URL(self):
#         return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

# setting = Setting()

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION: str = "staff_faces"
    RECREATE_COLLECTION: bool = False
    ALLOW_DROP_ALL: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
