# harness/tool/exceptions.py
class ToolException(Exception):
    """Base class for all tool exceptions.
    Subclasses are caught by ToolExecutor and converted to
    LLM-readable error strings with actionable prefixes.
    """
    pass


class ParamError(ToolException):
    """Parameter is wrong (format, range, type).
    LLM should fix the argument and retry.
    """
    pass


class RetryableError(ToolException):
    """Transient failure (timeout, rate limit, flap).
    LLM may retry once.
    """
    pass


class MissingInfoError(ToolException):
    """A required value cannot be inferred by the LLM.
    LLM should ask the user for the missing information.
    """
    pass


class DataNotFoundError(ToolException):
    """The queried record does not exist.
    Not a parameter error — the parameters were valid.
    LLM should inform the user the data is absent.
    """
    pass


class ServiceUnavailableError(ToolException):
    """Downstream service is down or in maintenance.
    Retrying will not help. LLM should tell the user to try later.
    """
    pass


class DegradedError(ToolException):
    """Service is up but quality is reduced.
    LLM should continue with a fallback strategy.
    """
    pass


class PartialResultError(ToolException):
    """Only incomplete data was returned.
    LLM should produce output based on what is available.
    """
    pass