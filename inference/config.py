import os
import logging


class Config:
    """ Global config object, inherited by other config objects """
    SECRET_KEY = os.urandom(64)

    REGISTER_VIEW_BLUEPRINT = True
    REGISTER_API_BLUEPRINT = True



class ProductionConfig(Config):
    """ Production config object, inherits from Config """
    pass


class DevelopmentConfig(Config):
    """ Development config object, inherits from Config """
    SECRET_KEY = "cxqWa-FpfskCecPhNqXCI8T3EDXfHlfV7afnpfzFVg9mCnbqW5bWKH85DDYSI-Puv9QqdMO5iWVVPvTck-9JRA"


class TestingConfig(Config):
    """ Testing config object, inherits from Config """
    pass


def get_config():
    """ Returns a config object based on the FLASK_ENV environment variable """
    flask_env = os.getenv("FLASK_ENV", None)

    if not flask_env:
        logging.warning("No value found in environment for FLASK_ENV. Production Config Loaded for Odigo NLU")
        return ProductionConfig()
    elif flask_env == "production":
        logging.info("Production Config Loaded")
        return ProductionConfig()
    elif flask_env == "development":
        logging.info("Development Config Loaded")
        return DevelopmentConfig()
    elif flask_env == "testing":
        logging.info("Testing Config Loaded")
        return TestingConfig()
    else:
        logging.warning("Unknown value found in environment for FLASK_ENV. Production Config Loaded")
        return ProductionConfig()
