from typing import Any, Generator, Optional, Tuple, List
from .condition_learning import PatternMatch

class Task:
    """
    Represents a goal to be accomplished. It maintains a list of methods
    that can be used to achieve it.
    """

    def __init__(self, name: str, args: tuple = (), methods: dict[str, 'Method'] = None):
        self.name = name
        self.args = args
        self.methods = methods if methods else {}

    def add_method(self, method: 'Method'):
        """Adds a method to this task."""
        self.methods[method.name] = method

    def get_methods(self):
        return [m for m in self.methods.values()]

    def __hash__(self):
        """
        Computes a hash based on the Task's name and number of arguments.
        """
        return hash((self.name, len(self.args)))

    def to_dict(self):
        return {
            'name': self.name,
            'args': list(self.args),
            'methods': {name: m.to_dict() for name, m in self.methods.items()}
        }

    @classmethod
    def from_dict(cls, data):
        methods = {name: Method.from_dict(m_data) for name, m_data in data.get('methods', {}).items()}
        return cls(
            name=data['name'],
            args=tuple(data.get('args', [])),
            methods=methods
        )


class Method:
        def __init__(self, 
            name: str, 
            args: tuple, 
            preconditions: List[dict],
            subtasks: List[Tuple],
            task_name: str = None,
            nl_description: Optional[str] = None,
            positive_states: Optional[List[List[dict]]] = None):
            """
            A method decomposes a task into an ordered list of subtasks,
            guarded by a set of precondition patterns that must match the
            current state.

            @param name: The name of the method.
            @param args: The arguments of the method.
            @param task_name: The name of the task this method accomplishes.
            @param subtasks: A list of tuples representing subtask templates.
            @param preconditions: A list of pattern dicts (may contain
                ?-prefixed variables) used for pattern matching against
                the working memory state.
            @param nl_description: A natural-language description of when
                this method is applicable.  Defaults to None.
            @param positive_states: A list of prior positive example states
                (each is a list of relevant fact dicts extracted from the
                tutor state). Used by the specialization prompt so the LLM
                can ensure it does not exclude valid states.
            @return: None
            """
            self.name = name
            self.args = args
            self.task_name = task_name
            self.subtasks = subtasks
            self.preconditions = preconditions
            self.nl_description = nl_description
            self.positive_states = positive_states if positive_states is not None else []

        def match(self, state: list[dict]):
            """
            Return the first consistent bindings if the precondition patterns
            match the given state AND all attached Filter guards pass.

            Fact-pattern preconditions are matched first via PatternMatch.match.
            Each Filter in self.filters is then evaluated against the accumulated
            bindings; if any filter returns False the match is rejected.

            @param state: The current working memory state as a list of fact dicts.
            @return: The bindings dict on the first successful match, or None.
            """
            try:
                # print(f'Preconditions: {self.preconditions}')
                # print(f'State: {state}')
                for matched_facts, bindings in PatternMatch.match(self.preconditions, state):
                    filters = getattr(self, 'filters', [])
                    if all(f.evaluate(bindings) for f in filters):
                        return bindings
                return None
            except StopIteration:
                return None


        def to_dict(self):
            return {
                'name': self.name,
                'args': list(self.args),
                'task_name': self.task_name,
                'subtasks': [list(s) for s in self.subtasks],
                'preconditions': [dict(p) for p in self.preconditions],
                'nl_description': self.nl_description,
                'positive_states': self.positive_states
            }

        @classmethod
        def from_dict(cls, data):
            return cls(
                name=data['name'],
                args=tuple(data.get('args', [])),
                task_name=data['task_name'],
                subtasks=[tuple(s) for s in data.get('subtasks', [])],
                preconditions=[Fact(p) for p in data.get('preconditions', [])],
                nl_description=data.get('nl_description'),
                positive_states=data.get('positive_states', [])
            )


class Operator:
    def __init__(self, name: str, args: tuple, preconditions: list[dict], effects: callable,
                 task: str = None, is_env_action: bool = False):
        """
        Represents a primitive operator in the HTN.

        @param name: The name of the operator.
        @param args: The arguments of the operator.
        @param preconditions: A list of pattern dictionaries to be
                              matched against the working memory.
        @param effects: A function that takes a bindings dictionary and
                        returns a new fact dictionary or a final value.
        @param task: The name of the task this operator accomplishes.
                     Defaults to the operator name.
        @param is_env_action: If True, this operator acts directly on the
                              environment and returns its resolved args
                              immediately instead of storing a result in
                              working memory.
        @return: None
        """
        self.task = task if task is not None else name
        self.name = name
        self.args = args
        self.preconditions = preconditions
        self.effects = effects
        self.is_env_action = is_env_action


class Fact(dict):
    """
    A dictionary-like object.
    You can get, set, and delete items using dot notation (e.g., `fact.key`)
    in addition to the standard bracket notation (e.g., `fact['key']`).
    """

    def get_original(self):
        # Define the keys to exclude
        metadata_keys = {'parent_facts', 'parent_op', 'grounded_op_tuple'}

        # Use a dictionary comprehension to create a new dict without those keys
        return {key: value for key, value in self.items() if key not in metadata_keys}

    def __init__(self, *args, **kwargs):
        super(Fact, self).__init__(*args, **kwargs)


    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            # Raise an AttributeError if the key is not found.
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __eq__(self, other):
        """
        Two Fact objects are considered equal if their dictionary contents are equal.
        This ignores other attributes like parent_facts.
        """
        if not isinstance(other, dict):
            return NotImplemented
        # Compare the core dictionary contents.
        return dict(self) == dict(other)

    def __hash__(self):
        """
        Computes a hash based on the dictionary's contents.
        A frozenset of the items provides an immutable, hashable representation.
        """
        items = [i for i in list(self.items()) if i[0] not in ['parent_facts', 'parent_op', 'grounded_op_tuple']]
        return hash(frozenset(items))

    def __repr__(self):
        """Provides a clearer string representation of the object."""
        return f"Fact({super().__repr__()})"