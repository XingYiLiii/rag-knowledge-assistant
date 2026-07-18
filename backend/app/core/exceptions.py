"""Safe, application-level exception types."""


class ApplicationError(Exception):
    """Base exception whose safe attributes can be returned to API clients."""

    def __init__(
        self,
        *,
        code: str = "APPLICATION_ERROR",
        message: str = "The request could not be processed.",
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
