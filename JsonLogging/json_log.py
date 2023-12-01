"""
Logging in JSON Format
"""

import json
import logging
import logging.config

ALLOWED_ATTRIBUTES = {
    "asctime": "s",
    "created": "f",
    "filename": "s",
    "funcName": "s",
    "levelname": "s",
    "levelno": "s",
    "lineno": "d",
    "message": "s",
    "module": "s",
    "msecs": "d",
    "name": "s",
    "pathname": "s",
    "process": "d",
    "processName": "s",
    "relativeCreated": "d",
    "thread": "d",
    "threadName": "s"
}
STYLE_GENERATOR = {
    "$":
        lambda attributes: ":".join(f"${{{attribute}}}"
                                    for attribute in attributes),
    "%":
        lambda attributes: ":".join(
            f"%({attribute}){ALLOWED_ATTRIBUTES[attribute]}"
            for attribute in attributes),
    "{":
        lambda attributes: ":".join(f"{{{attribute}}}"
                                    for attribute in attributes)
}


class JsonFormatter(logging.Formatter):
    """
    Custom logging formatter to log in JSON format

    Parameters
    ----------
    log_details : logging.config.ConvertingDict
        details of the log format
    timestamp_details : logging.config.ConvertingDict
        details of the timestamp format
    specification_style : str
        log format specification style

    Raises
    ------
    ValueError
        specification style is not alllowed, i.e. not one of $, % or {
    ValueError
        unknown attributes are provided to be logged
    """

    def __init__(self, log_details: logging.config.ConvertingDict,
                 timestamp_details: logging.config.ConvertingDict,
                 specification_style: str) -> None:
        self.log_details = log_details
        self.timestamp_details = timestamp_details
        self.specification_style = specification_style

        if self.specification_style not in STYLE_GENERATOR.keys():
            raise ValueError(
                f"Invalid Specification Style -> Allowed: {' '.join(STYLE_GENERATOR.keys())}"
            )

        if not set(self.log_details.values()).issubset(
                ALLOWED_ATTRIBUTES.keys()):
            raise ValueError(
                f"Invalid Log Format -> Allowed Attributes: {' '.join(ALLOWED_ATTRIBUTES.keys())}"
            )

        self.log_attritutes = list(self.log_details.values())
        self.log_style = STYLE_GENERATOR[specification_style](
            self.log_attritutes)
        self.timestamp_format = self.timestamp_details["format"]

        super(JsonFormatter, self).__init__(fmt=self.log_style,
                                            datefmt=self.timestamp_format,
                                            style=self.specification_style)

    def format(self, record: logging.LogRecord) -> str:
        """
        Returns the log in the desired JSON format

        Parameters
        ----------
        record : logging.LogRecord
            information that can be logged

        Returns
        -------
        str
            string in JSON format containing information to be logged
        """
        log_information = {}

        log_information["@message"] = record.getMessage()

        timestamp = self.formatTime(record, self.datefmt)
        if self.timestamp_details["milliseconds_required"]:
            timestamp = f"{timestamp}.{int(record.msecs):03d}"
        log_information["@timestamp"] = timestamp

        if record.exc_info:
            log_information["@exception"] = self.formatException(
                record.exc_info)
        elif record.exc_text:
            log_information["@exception"] = record.exc_text

        if record.stack_info:
            log_information["@stack"] = self.formatStack(record.stack_info)

        if "message" in self.log_details.values():
            record.message = record.getMessage()

        if "asctime" in self.log_details.values():
            record.asctime = self.formatTime(record, self.datefmt)

        for field_name, attribute in self.log_details.items():
            log_information[field_name] = getattr(record, attribute)

        return json.dumps(log_information)


with open("log_configurations.json", "r") as config_file_name:
    LOG_CONFIGS = json.load(config_file_name)

logging.config.dictConfig(LOG_CONFIGS)
