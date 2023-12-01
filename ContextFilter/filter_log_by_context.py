"""Filter log records."""
import dataclasses
import inspect
import logging


@dataclasses.dataclass
class TaskContextFilter:
    """Implement log filter based on origin of call stack.

    Parameters
    ----------
    task : str
        name of orchestration function that must be called at root of stack
    task_argument : str
        name of identifier argument of ``task`` that must have a specific value
    task_identifier : int
        identifier of orchestration task that must be passed to ``task_argument``
    """

    task: str
    task_argument: str
    task_identifier: int

    def filter(self: "TaskContextFilter", record: logging.LogRecord) -> bool:  # noqa: A003
        """Check if log is originated from a specific function call with specific argument.

        Parameters
        ----------
        record : logging.LogRecord
            details of current log

        Returns
        -------
        bool
            indicator or log origin
        """
        del record

        current_stack = inspect.currentframe()

        while current_stack:
            if current_stack.f_code.co_name == self.task:
                current_stack_variables = inspect.getargvalues(current_stack).locals

                if current_stack_variables.get(self.task_argument) == self.task_identifier:
                    return True

            current_stack = current_stack.f_back

        return False
