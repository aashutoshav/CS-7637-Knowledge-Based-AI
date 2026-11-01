from typing import Any, Generator, Optional, Tuple, List

class Cursor:
    def __init__(self, task: tuple = None, method_idx: int = 0, subtask_idx: int = 0):
        self.task = task
        self.task_name = task[0] if isinstance(task, tuple) else None
        self.task_args = task[1:] if isinstance(task, tuple) else ()
        self.method_idx = method_idx
        self.subtask_idx = subtask_idx

    def copy(self):
        return self.__class__((self.task_name, *self.task_args), self.method_idx, self.subtask_idx)

class WorkingMemory:
    def __init__(self):
        self.state = None
        self.facts = []
        self.memory = {}

    def reset(self) -> None:
        """
        Clears transient planning data (operator result bindings and
        accumulated facts) while preserving the state reference.  Should
        be called at the start of each new plan() invocation.

        @return: None
        """
        self.memory.clear()
        self.facts.clear()

    def set_state(self, state) -> None:
        self.state = state

    def add_fact(self, fact: 'Fact') -> None:
        self.facts.append(fact)

    def get_memory(self) -> list[dict]:
        return {"state": self.state, "facts": self.facts}

def is_variable(x):
    return isinstance(x, str) and x.startswith('?')