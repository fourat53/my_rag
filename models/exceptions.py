import os


class MissingEnvVarError(Exception):
    def __init__(self, var_name: str, msg: str):
        self.var_name = var_name
        self.msg = msg
        super().__init__(self.msg)


def get_checked_env_var(var_name: str, logger) -> str:
    value = os.getenv(var_name)

    if not value:
        msg = f"❌ Environment variable {var_name} is not configured."
        logger.error(msg)
        raise MissingEnvVarError(var_name, msg)

    return value


# file loader exceptions
class FileNotFoundError(Exception):
    pass


class FileTypeError(Exception):
    pass


class DocumentLoadError(Exception):
    pass


# chat exceptions
class GeminiModelsError(Exception):
    pass


class GeminiResponseError(Exception):
    pass
