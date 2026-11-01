import os
import logging
import time
from copy import deepcopy


class PlannerLogger:
    """
    A logger for the HTN planner that tracks and logs all planner actions.

    :param log_file: Path to the log file
    :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param console_output: Whether to also print logs to console
    """

    def __init__(self, log_file: str, log_level: int = logging.INFO, console_output: bool = False):
        """
        Initialize the planner logger.

        :param log_file: Path to the log file
        :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        :param console_output: Whether to also print logs to console
        """
        self.logger = logging.getLogger(f'htn_planner_{log_file}')  # Unique name per file
        self.logger.setLevel(log_level)

        # Avoid duplicate handlers if logger already exists
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')  # Overwrite log each run
        file_handler.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)

        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.info(f"HTN Planner Logger initialized")

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)

    def log_task(self, task):
        self.info(f"Task: {task}")

    def log_method_application(self, task, method, subtasks):
        self.info(f"Applying Method: {method.name} to task {task}")
        self.debug(f"  Resulting in subtasks: {subtasks}")

    def log_operator_application(self, task, operator, effects):
        self.info(f"Applying Operator: {operator.name} to task {task}")
        self.debug(f"  Effects: {effects}")

    def log_state_update(self, state):
        self.debug(f"State updated: {state}")

    def log_backtrack(self, from_task, to_task):
        self.info(f"Backtracking from {from_task} to {to_task}")

    def log_plan_step(self, plan_action):
        self.info(f"Plan Step: {plan_action}")

    def log_plan_complete(self, plan):
        self.info(f"Plan completed with {len(plan)} steps")
        for idx, action in enumerate(plan):
            self.info(f"  Step {idx + 1}: {action}")


# --- Trace Classes (from trace.py) ---

class TraceEntry:
    """Base class for all trace entries."""

    def __init__(self, entry_type, description):
        self.entry_type = entry_type
        self.description = description
        self.timestamp = time.time()


class MethodEntry(TraceEntry):
    """Records a method selection."""

    def __init__(self, task, method, reason):
        method_name = method.name if hasattr(method, 'name') else str(method)
        task_name = task[0] if isinstance(task, tuple) else str(task)
        super().__init__("method", f"Applied method {method_name} to task {task_name}")
        self.task = task
        self.method = method
        self.reason = reason


class OperatorEntry(TraceEntry):
    """Records an operator execution."""

    def __init__(self, task, operator, success, state_before, state_after=None):
        op_name = operator.name if hasattr(operator, 'name') else str(operator)
        task_name = task[0] if isinstance(task, tuple) else str(task)
        status = "successfully" if success else "unsuccessfully"
        super().__init__("operator", f"Executed operator {op_name} {status} for task {task_name}")
        self.task = task
        self.operator = operator
        self.success = success
        self.state_before = state_before
        self.state_after = state_after


class TaskEntry(TraceEntry):
    """Records a task decomposition."""

    def __init__(self, task, parent_task=None):
        task_name = task[0] if isinstance(task, tuple) else str(task)
        if parent_task:
            parent_name = parent_task[0] if isinstance(parent_task, tuple) else str(parent_task)
            desc = f"Decomposing task {task_name} as subtask of {parent_name}"
        else:
            desc = f"Decomposing root task {task_name}"
        super().__init__("task", desc)
        self.task = task
        self.parent_task = parent_task


class BacktrackEntry(TraceEntry):
    """Records a backtrack operation."""

    def __init__(self, from_task, to_task, reason):
        from_name = from_task[0] if isinstance(from_task, tuple) else str(from_task)
        if to_task is None:
            desc = f"Backtracking from {from_name}. No parent tasks: {reason}"
        else:
            to_name = to_task[0] if isinstance(to_task, tuple) else str(to_task)
            desc = f"Backtracking from {from_name} to {to_name}: {reason}"
        super().__init__("backtrack", desc)
        self.from_task = from_task
        self.to_task = to_task
        self.reason = reason


class Trace:
    """
    Maintains a comprehensive trace of the planning process including method decisions,
    operator executions, task decompositions, and backtracking operations.
    """

    def __init__(self, options: dict = None):
        self.entries = []
        self.sequence_counter = 0
        self.options = options if options else {}
        self.live = self.options.get("live", False)
        self.include_states = self.options.get("include_states", False)
        self.max_entries = self.options.get("max_entries", None)

    def add_method(self, task, method, reason="Method applicable"):
        """Record a method selection."""
        entry = MethodEntry(task, method, reason)
        self.add_trace_entry(entry)
        return entry

    def add_operator(self, task, operator, success, state_before, state_after=None):
        """Record an operator execution."""
        entry = OperatorEntry(task, operator, success, state_before, state_after)
        self.add_trace_entry(entry)
        self.sequence_counter += 1 if success else 0
        return entry

    def add_task(self, task, parent_task=None):
        """Record a task decomposition."""
        entry = TaskEntry(task, parent_task)
        self.add_trace_entry(entry)
        return entry

    def add_backtrack(self, from_task, to_task, reason):
        """Record a backtrack operation."""
        entry = BacktrackEntry(from_task, to_task, reason)
        self.add_trace_entry(entry)
        return entry

    def add_trace_entry(self, entry):
        """Add a trace entry."""
        self.entries.append(entry)
        if self.live:
            self.print_trace()

    def get_current_plan(self):
        """
        Get the current plan consisting of successful operator executions.
        Returns a list of operators in the order they were executed.
        """
        plan = []
        for entry in self.entries:
            if entry.entry_type == "operator" and entry.success:
                plan.append(entry.operator)
        return plan

    def get_current_trace(self):
        """
        Get the full trace with all entries.
        If include_states is False, state information is omitted to reduce verbosity.
        """
        if self.include_states:
            return self.entries

        # Create a copy without state information
        simplified_trace = []
        for entry in self.entries:
            entry_copy = deepcopy(entry)
            if hasattr(entry_copy, 'state_before'):
                entry_copy.state_before = "..."
            if hasattr(entry_copy, 'state_after'):
                entry_copy.state_after = "..."
            simplified_trace.append(entry_copy)

        return simplified_trace

    def print_plan(self):
        """
        Print the current plan.
        """
        plan = self.get_current_plan()

        print("\n+" + "-" * 49 + "+")
        print("|" + "                 CURRENT PLAN                    " + "|")
        print("+" + "-" * 49 + "+")

        if not plan:
            print("No actions in plan yet.")
            return

        for i, operator in enumerate(plan):
            op_name = operator.name if hasattr(operator, 'name') else str(operator)
            op_args = operator.args if hasattr(operator, 'args') else []
            print(f"Step {i + 1:02d}: {op_name}({', '.join(str(arg) for arg in op_args)})")

        print(f"\nTotal actions: {len(plan)}")
        print("-" * 52)

    def print_trace(self):
        """
        Prints the full trace of the plan.
        :return:
        """
        trace = self.get_current_trace()
        if self.live:
            trace = trace[-1:] if trace else []

        if self.max_entries:
            trace = trace[-self.max_entries:]

        print("\n+" + "-" * 49 + "+")
        print("|" + "                PLANNING TRACE                   " + "|")
        print("+" + "-" * 49 + "+")

        if not trace:
            print("No trace entries yet.")
            return

        # Define colors for different entry types (ANSI escape codes)
        colors = {
            "task": "\033[94m",  # Blue
            "method": "\033[92m",  # Green
            "operator": "\033[93m",  # Yellow
            "backtrack": "\033[91m"  # Red
        }
        reset_color = "\033[0m"

        for i, entry in enumerate(trace):
            # Format timestamp
            timestamp = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))

            # Start with line number and timestamp
            line = f"{i + 1:03d} [{timestamp}] "

            # Add colored type indicator
            type_color = colors.get(entry.entry_type, "")
            type_str = f"{type_color}[{entry.entry_type.upper():<9}]{reset_color}"  # Padded
            line += f"{type_str} "

            # Add description
            line += entry.description

            # Print line
            print(line)

            # Add additional info based on entry type
            if entry.entry_type == "method":
                print(f"     Reason: {entry.reason}")

            elif entry.entry_type == "operator" and self.include_states:
                print(f"     Success: {entry.success}")
                if self.include_states:
                    print(f"     State before: {entry.state_before}")
                    if entry.state_after:
                        print(f"     State after: {entry.state_after}")

            # Add separator between entries
            print("-" * 60)

        print(f"\nTotal entries: {len(trace)}")
        if self.max_entries and len(self.entries) > self.max_entries:
            print(f"Showing last {self.max_entries} of {len(self.entries)} entries.")
        print("-" * 52)
