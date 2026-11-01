from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import os

from .htn_elements.structure import Fact, Method, Operator, Task
from .htn_elements.planning import Cursor, WorkingMemory, is_variable
from .htn_elements.condition_learning import Filter, PatternMatch, substitute, unify
from .htn_logging import PlannerLogger, Trace


class HTN:
    """
    The Hierarchical Task Network class.
    """
    def __init__(self,
                env=None,
                domain: dict[str, list] = None,
                operators: list = None,
                log_file: str = 'htn_planner.log',
                log_level: int = logging.INFO,
                log_to_console: bool = False,
                trace_options: dict = {},
                preserve_tasks: bool = False):
        """
        Initializes the HTN planner.

        @param env: An environment object.
        @param domain: A dict mapping task name strings to lists of Method objects.
        @param operators: A list of primitive operators. Each element may be a
                          string (registered as an environment-action Operator) or
                          an Operator object.
        @param log_file: Path for the log file.
        @param log_level: Logging level.
        @param log_to_console: Whether to also log to console.
        @param live_trace: If True, print each trace entry as it is recorded.
        @param preserve_tasks: If True, task queue/stack persist across plan() calls.
        @return: None
        """

        # Environment Setup
        self.env = env

        # Network Setup
        self.network = self._parse_domain(domain if domain else {})
        self.operators = {}
        self.working_memory = WorkingMemory()

        self._add_operators(operators)

        # Planning Setup
        self.explanation_cache: Dict[Any, List[Tuple[List[str], List[Dict]]]] = defaultdict(list)
        self.task_queue = [] # Keeps track of tasks to complete during planning
        self.task_stack = [] # Stack used to keep track of tasks as they are decomposed
        self.cursor = Cursor() # Pointer to the current task being processed
        self.last_used_task: Optional[str] = None
        self.last_used_method: Optional[str] = None
        self.preserve_tasks = preserve_tasks

        # Learning Setup (not implemented in this version)
        self._method_counter = 0 # Used to generate method name skolems for new methods learned
        
        # Logging Setup
        self.trace_options = trace_options if trace_options else {} # Options for trace
        self.logger = PlannerLogger(log_file, log_level, log_to_console) # Logs learning and planning steps
        self.trace: Trace | None = None  # Keeps a trace of a planning run
    
    def _add_operators(self, operators: list[str, Operator]) -> None:
        for op in (operators or []):
            if isinstance(op, Operator):
                self.operators[op.name] = op
                task_name = op.name
            elif isinstance(op, str):
                self.operators[op] = Operator(
                    name=op, args=(), preconditions=[],
                    effects=lambda b: {}, is_env_action=True
                )
                task_name = op
            self.network[task_name] = Task(name=task_name, methods={task_name: self.operators[task_name]})
    
    def _parse_domain(self, domain: dict[str, list]) -> dict:
        """
        Converts the user-facing domain dict into the internal HTN network format.

        @param domain: A dict mapping task name strings to lists of Method objects.
        @return: A dict mapping task name strings to Task objects.
        """
        network = {}
        for task_name, methods in domain.items():
            task = Task(name=task_name, args=())
            for method in methods:
                # Separate Filter entries from ordinary pattern dicts
                filters = []
                clean_preconditions = []
                for precond in method.preconditions:
                    if 'filter' in precond and callable(precond['filter']):
                        filters.append(Filter(precond['filter']))
                    else:
                        clean_preconditions.append(precond)
                method.preconditions = clean_preconditions
                method.filters = filters
                task.add_method(method)
            network[task_name] = task
        return network

    def add_operator(self, operator: Operator):
        """
        Adds a predefined operator to the HTN.

        @param operator: The operator to add.
        @return: None
        """
        self.operators[operator.name] = operator

    def _resolve_args(self, args: list) -> list:
        """
        Resolves a list of task arguments by replacing any ?-prefixed
        variables with their concrete values from working memory.

        @param args: A list of task arguments (strings or constants).
        @return: A new list with variables replaced by their bound values.
        """
        resolved = []
        for arg in args:
            if is_variable(arg) and arg in self.working_memory.memory:
                mem_val = self.working_memory.memory[arg]
                resolved.append(
                    mem_val['value'] if isinstance(mem_val, dict) and 'value' in mem_val else mem_val
                )
            else:
                resolved.append(arg)
        return resolved


    def _execute_operator(self, task: tuple, op: Operator, input_args: list,
                          result_var: str) -> bool:
        """
        Executes an operator's effects, stores the result in working memory,
        records the trace, and pops the completed task from the queue.

        @param task: The original task tuple from the queue.
        @param op: The Operator object to execute.
        @param input_args: The resolved, ground input arguments for the operator.
        @param result_var: The variable name in which to store the operator's result.
        @return: True if the operator executed successfully.
        """
        bindings = {arg_name: arg_val for arg_name, arg_val in zip(op.args, input_args)}
        self.working_memory.memory[result_var] = op.effects(bindings)
        grounded_task = (op.name, *input_args, result_var)
        self.logger.info(f"Executing operator (grounded): {grounded_task}")
        self.trace.add_operator(task, op, True, None, None)
        self.task_queue.pop(0)
        return True

    def _try_decompose(self, task: tuple, task_name: str, task_args: list) -> bool:
        """
        Attempts to decompose a complex task by iterating through its methods
        starting from the cursor position. On success, pushes subtasks onto
        the front of the task queue and saves state for backtracking.

        @param task: The original task tuple from the queue.
        @param task_name: The name of the task to decompose.
        @param task_args: The arguments of the task.
        @return: True if a valid decomposition was found, False otherwise.
        """
        task_def = self.network[task_name]
        method_idx = self.cursor.method_idx
        methods = task_def.get_methods()

        while method_idx < len(methods):
            method = methods[method_idx]
            filters = getattr(method, 'filters', [])

            # Fetch fresh state from the environment and update working memory
            # state = self.env.get_state()
            # self.working_memory.set_state(state)

            for matched_facts, match_bindings in PatternMatch.match(method.preconditions,
                                                                    self.working_memory.state):
                final_bindings = unify(method.args, tuple(task_args), match_bindings)
                if final_bindings is None:
                    continue

                # Evaluate Filter guards with full bindings so method-arg variables
                # (e.g. ?tx, ?ty) are available alongside pattern-matched variables.
                if not all(f.evaluate(final_bindings) for f in filters):
                    continue

                self.logger.info(f"Applying method '{method.name}' to task '{task_name}'")
                reason = "Method preconditions met and unification successful."
                self.trace.add_method(task, method, reason)
                self.logger.debug(f"  Reason: {reason}")

                # Save current state for backtracking before modifying the task queue.
                self.cursor.method_idx = method_idx
                # Track the method used for caller inspection
                self.last_used_task = task_name
                self.last_used_method = method.name
                self.task_stack.append((task, deepcopy(self.working_memory.state), self.cursor.copy()))

                instantiated_subtasks = substitute(method.subtasks, final_bindings)
                self.task_queue.pop(0)
                # Add subtasks to front of queue to be processed first
                self.task_queue[0:0] = instantiated_subtasks
                self.logger.debug(f"Added subtasks: {instantiated_subtasks}")
                # We are are new level, reset Cursor
                self.cursor = Cursor(self.task_queue[0][0])
                return True

            method_idx += 1

        return False

    def _backtrack(self, task: tuple, task_name: str, reason: str) -> None:
        """
        Handles a failed decomposition by backtracking to the parent task.
        Pops the task stack, restores the previous state and cursor, and
        re-queues the parent task to try its next method.

        @param task: The task tuple that failed decomposition.
        @param task_name: The name of the task that failed.
        @return: None
        @raises FailedPlanException: If the task stack is empty (no parent to backtrack to).
        """
        to_task = self.task_stack[-1][0] if self.task_stack else None
        self.logger.warning(f"Backtracking: {reason}")
        self.trace.add_backtrack(task, to_task, reason)

        if not self.task_stack:
            self.logger.error(f"Plan failed: {reason} and no higher method to backtrack to.")
            raise FailedPlanException(
                f"Plan failed: Tried to backtrack after task '{task_name}', but no higher method to backtrack to.")

        # Method(s) unsuccessful, so pop from stack to backtrack to the parent.
        # TODO ACTUALLY NEED TO REMOVE ALL SUBTASKS PUT ON QUEUE FROM FAILED METHOD, NOT JUST THE ONE IN THE FRONT
        parent_task, self.working_memory.state, self.cursor = self.task_stack.pop()
        self.cursor.method_idx += 1
        self.task_queue.pop(0)  # Remove the task that couldn't be decomposed.
        self.task_queue.insert(0, parent_task)  # Re-queue parent to try its next method.

    def plan(self, tasks: [tuple, list[tuple], str]) -> None:
        """
        Generates and executes an action plan for the given tasks.
        Fetches state from self.env.get_state() on each iteration and
        executes environment actions directly via self.env.execute_action().
        The task_stack is cleared after each successful env-action because
        committed game actions cannot be undone.

        On failure, raises FailedPlanException.

        @param tasks: A single task tuple or a list of task tuples to plan for.
        @return: None
        @raises FailedPlanException: If the planner cannot find a valid plan.
        """
        self.trace = Trace(options=self.trace_options)
        self.logger.info(f"Starting new plan for tasks: {tasks}")

        if isinstance(tasks, str):
            tasks = [(tasks,)]

        if isinstance(tasks, tuple):
            tasks = [tasks]

        # Reset transient planning state to prevent accumulation across calls
        # if not self.preserve_tasks:
        #    self.task_queue.clear()
        #    self.task_stack.clear()
        #    self.working_memory.reset()

        self.task_queue.extend(tasks)

        while self.task_queue:
            task = self.task_queue[0]
            task_name, *task_args = task

            self.logger.info(f"Processing task: {task}")
            # Find parent task for trace
            parent_task_tuple = self.task_stack[-1][0] if self.task_stack else None
            self.trace.add_task(task, parent_task_tuple)

            # If the current task is different from the cursor's task, reset the cursor.
            if self.cursor.task_name != task_name or self.cursor.task_args != tuple(task_args):
                self.cursor = Cursor(task)

            # Fetch fresh state from the environment and update working memory
            state = self.env.get_state()
            # print(json.dumps(state, indent=2))
            self.working_memory.set_state(state)

            if task_name not in self.network:
                self.logger.error(f"Plan failed: Task '{task_name}' does not exist.")
                raise FailedPlanException(f"Plan failed for task '{task_name}'. Task does not exist in HTN.")

            # ── Primitive operator ──
            if task_name in self.operators:
                op = self.operators[task_name]
                resolved_task_args = self._resolve_args(task_args)

                # Env-actions execute directly on the environment
                if op.is_env_action:
                    self.logger.info(f"Attempting environment action: {op.name}")
                    state_before = deepcopy(self.working_memory.state)
                    self.trace.add_operator(task, op, True, state_before, deepcopy(self.working_memory.state))
                    #if not self.preserve_tasks:
                    #    self.task_queue.clear()
                    #    self.task_stack.clear()
                    op_success, message = self.env.execute_action((task_name, *resolved_task_args))
                    message = f"Operator execution {'successful' if op_success else 'unsuccessful'} " \
                              f"with status message: {'OK' if op_success else message}"
                    self.logger.info(message)
                    # Clear the task stack after a committed env-action: there is nothing
                    # meaningful to backtrack to once a game action has been sent.
                    if op_success:
                        self.task_stack.clear()
                else:

                    # Standard operator execution
                    self.logger.info(f"Attempting operator: {op.name}")
                    # Input args exclude the trailing result variable
                    input_task_args = resolved_task_args[:len(op.args)]
                    result_var = task_args[-1]
                    all_ground = not any(is_variable(a) for a in input_task_args)
                    op_success = False

                    if all_ground:
                        op_success = self._execute_operator(task, op, input_task_args, result_var)
                    else:
                        # Some args are still variables — bind via state pattern matching.
                        op_bindings = unify(op.args, tuple(input_task_args))
                        if op_bindings is not None:
                            ground_preconditions = substitute(op.preconditions, op_bindings)
                            for matched_facts, match_bindings in PatternMatch.match(ground_preconditions,
                                                                                    self.working_memory.state):
                                final_bindings = {**op_bindings, **match_bindings}
                                grounded_task = substitute(task, final_bindings)
                                grounded_task_args = grounded_task[1:-1]
                                result_var = grounded_task[-1]
                                op = self.operators[grounded_task[0]]
                                op_success, message = self._execute_operator(task, op, list(grounded_task_args), result_var)
                                message = f"Operator execution {'successful' if op_success else 'unsuccessful'} " \
                                          f"with status message: {'OK' if op_success else message}"
                                self.logger.info(message)
                                break
                # TODO ADD BACKTRACKING HERE
                if not op_success:
                    reason = f"Operator '{task_name}' preconditions not met or unification failed."
                    self.logger.info(reason)
                    self.trace.add_operator(task, op, False, None)
                    self._backtrack(task, task_name, reason=f"Operator ({op.name}) failed execution execution.")
                    continue
                # Remove env-action task from the queue after execution.
                # Note: non-env operators are removed inside _execute_operator itself,
                # so we only pop here for env-actions to avoid a double-pop.
                if op.is_env_action:
                    self.task_queue.pop(0)

            else:
                # ── Complex task decomposition ──
                if not self._try_decompose(task, task_name, task_args):
                    self._backtrack(task, task_name, reason= f"No applicable method found for task '{task_name}'")
                    continue

        self.logger.info("Plan execution complete.")
        if self.trace:
            self.trace.print_plan()  # Print the final plan

    def reset(self) -> None:
        """
        Clears all transient planning state so the planner starts fresh.
        Should be called when a level transition is detected to discard
        any stale locked-in task arguments from the previous level.

        @return: None
        """
        self.task_queue.clear()
        self.task_stack.clear()
        self.cursor = Cursor()
        self.working_memory.reset()
        self.last_used_task = None
        self.last_used_method = None

    def get_trace(self, include_states=False):
        """Returns the trace object from the last plan execution."""
        if self.trace:
            return self.trace.get_current_trace(include_states)
        return []

    def print_trace(self, include_states=False, max_entries=None):
        """Prints the trace from the last plan execution."""
        if self.trace:
            self.trace.print_trace(include_states, max_entries)
        else:
            print("No trace available. Run plan() first.")

    def print_plan_trace(self):
        """Prints the final plan from the last trace."""
        if self.trace:
            self.trace.print_plan()
        else:
            print("No trace available. Run plan() first.")

    def print_network(self):
        """Prints a textual representation of the HTN network."""
        print("=== HTN Network ===")
        if not self.network and not self.operators:
            print("  (empty)")
            return
        print(f"Operators ({len(self.operators)}): {list(self.operators.keys())}")
        for task_name, task in self.network.items():
            methods = task.get_methods()
            print(f"Task: {task_name} ({len(methods)} method(s))")
            for method in methods:
                ops = [s[0] for s in method.subtasks]
                print(f"  Method '{method.name}': {ops}")


class FailedPlanException(Exception):
    pass
