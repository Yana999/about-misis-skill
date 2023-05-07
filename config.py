from pydantic import BaseSettings, Field


class EnvSettings(BaseSettings):

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


class VOAConfig(EnvSettings):

    tmp_dir: str = Field(..., env="TMP_DIR")
    ml_root_path: str = Field(..., env="ML_ROOT_PATH")
    ml_config_path: str = Field(..., env="ML_CONFIG_FILE")
