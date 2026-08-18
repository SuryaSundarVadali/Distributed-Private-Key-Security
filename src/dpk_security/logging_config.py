import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": (
                "%(asctime)s %(levelname)s "
                "%(name)s "
                "[node=%(node_id)s task=%(task_id)s] "
                "%(message)s"
            ),
        },
    },
    "filters": {
        "context": {
            "()": "dpk_security.logging_config.ContextFilter",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "filters": ["context"],
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "node_id"):
            record.node_id = "-"
        if not hasattr(record, "task_id"):
            record.task_id = "-"
        return True


def setup_logging() -> None:
    dictConfig(LOGGING_CONFIG)