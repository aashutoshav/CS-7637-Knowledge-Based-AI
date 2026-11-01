import itertools
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.exceptions import NotFittedError
from .planning import is_variable
# from vssm import VSSM  # Import the VSSM class


class PatternMatch:
    """
    Implements a unification algorithm to match patterns (with variables)
    against a state of concrete facts.

    Uses a backtracking search with per-pattern candidate filtering instead
    of generating all permutations, reducing complexity from O(n!) to
    approximately O(n^k) in the worst case (and near-linear in practice due
    to constant-key filtering).
    """

    @staticmethod
    def _is_variable(value):
        """Checks if a value is a variable (e.g., '?x').
        @param value: The value to check.
        @return: True if the value is a variable string, False otherwise.
        """
        return isinstance(value, str) and value.startswith('?')

    @classmethod
    def _candidates_for_pattern(cls, pattern, facts):
        """
        Pre-filters facts to only those that could possibly match *pattern*
        based on constant (non-variable) key/value constraints.

        @param pattern: A single precondition dict (may contain ?-variables).
        @param facts: The full list of fact dicts.
        @return: A list of (fact_index, fact_dict) pairs that pass the filter.
        """
        candidates = []
        for idx, fact in enumerate(facts):
            match = True
            for p_key, p_val in pattern.items():
                if p_key not in fact:
                    match = False
                    break
                if fact[p_key] is None:
                    match = False
                    break
                # Only check constant constraints here; variables are
                # resolved during the binding phase.
                if not cls._is_variable(p_val) and p_val != fact[p_key]:
                    match = False
                    break
            if match:
                candidates.append((idx, fact))
        return candidates

    @classmethod
    def _try_bind(cls, pattern, fact, bindings):
        """
        Attempts to unify a single pattern with a single fact given existing
        bindings.  Returns the extended bindings dict on success, or None if
        the pattern/fact pair is inconsistent.

        @param pattern: A precondition dict (may contain ?-variables).
        @param fact: A concrete fact dict.
        @param bindings: The current variable bindings dict.
        @return: A new bindings dict on success, or None on failure.
        """
        new_bindings = bindings.copy()
        for p_key, p_val in pattern.items():
            f_val = fact[p_key]  # key presence already guaranteed by _candidates_for_pattern
            if cls._is_variable(p_val):
                if p_val in new_bindings and new_bindings[p_val] != f_val:
                    return None
                new_bindings[p_val] = f_val
            # Constant mismatch already filtered by _candidates_for_pattern
        return new_bindings

    @classmethod
    def match(cls, patterns: list[dict], facts: list[dict]):
        """
        A generator that finds all valid ways to match a list of patterns
        against a list of facts. It yields tuples of (matched_facts, bindings).

        Uses backtracking with per-pattern candidate pre-filtering rather than
        itertools.permutations, avoiding factorial blowup on large fact lists.

        @param patterns: The operator's preconditions, may contain variables.
        @param facts: The current concrete facts in working memory.
        @return: Yields (matched_facts_tuple, bindings_dict) pairs.
        """
        if not patterns:
            yield (), {}
            return

        # Separate real patterns from filter-only declarations
        real_indices = []
        for i, p in enumerate(patterns):
            if 'filter' not in p:
                real_indices.append(i)

        # Pre-compute candidate facts for each real pattern
        candidate_lists = []
        for i in real_indices:
            candidate_lists.append(cls._candidates_for_pattern(patterns[i], facts))

        # If any real pattern has zero candidates, no match is possible
        if any(len(c) == 0 for c in candidate_lists):
            return

        # Backtracking search over candidate assignments
        # Stack frames: (pattern_position_in_real_indices, candidate_index, bindings, used_fact_indices, matched_facts)
        stack = [(0, 0, {}, set(), [])]

        while stack:
            pos, cand_idx, bindings, used, matched = stack.pop()

            if pos == len(real_indices):
                # All real patterns matched — build the full matched tuple
                # (including None placeholders for filter slots, to keep
                #  the tuple the same length as patterns for backward compat)
                full_matched = [None] * len(patterns)
                real_iter = iter(matched)
                for i in real_indices:
                    full_matched[i] = next(real_iter)
                yield tuple(full_matched), bindings
                continue

            candidates = candidate_lists[pos]
            # Push candidates in reverse so the first candidate is tried first
            for ci in range(len(candidates) - 1, cand_idx - 1, -1):
                fact_idx, fact = candidates[ci]
                if fact_idx in used:
                    continue
                new_bindings = cls._try_bind(patterns[real_indices[pos]], fact, bindings)
                if new_bindings is not None:
                    new_used = used | {fact_idx}
                    new_matched = matched + [fact]
                    stack.append((pos + 1, 0, new_bindings, new_used, new_matched))


class ConditionLearner(ABC):
    """
    Abstract Base Class for a condition learner. This defines the interface
    for any learning algorithm used for a method's preconditions.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, state: list[dict], label: bool) -> None:
        """Incorporate a new training instance into the learner."""
        pass

    @abstractmethod
    def predict(self, state: list[dict]) -> bool:
        """Determine if the given state satisfies the learned conditions."""
        pass

    @abstractmethod
    def get_rules(self) -> list:
        """Extract the learned conditions in a planner-readable format."""
        pass


def _flatten_state_dt(state: list[dict]) -> dict:
    """
    Helper function for DecisionTree to convert a list of dictionaries (state) into a single
    flat dictionary. It prepends the 'id' of each dictionary to its keys.
    Example: [{'id': 'f1', 'val': 3}] -> {'f1_id': 'f1', 'f1_val': 3}
    """
    flat_state = {}
    for fact in state:
        if 'id' not in fact:
            continue
        fact_id = fact['id']
        for key, value in fact.items():
            # The 'id' attribute is now included in the flattened state.
            flat_state[f"{fact_id}_{key}"] = value
    return flat_state


def _flatten_state_vssm(state: list[dict]) -> dict:
    """
    Helper function for VSSM to convert a list of dictionaries (state) into a single
    flat dictionary, creating keys like 'identifier.attribute'.
    """
    new_instance = {}
    for fact in state:
        if 'id' not in fact:
            continue
        identifier = fact['id']
        for attr, value in fact.items():
            new_instance[f'{identifier}.{attr}'] = value
    return new_instance


class DecisionTreeLearner(ConditionLearner):
    """
    A ConditionLearner that uses a Decision Tree classifier to learn preconditions.
    Handles categorical data by using one-hot encoding.
    """

    def __init__(self):
        super().__init__()
        self.tree = DecisionTreeClassifier()
        self.X_train_dicts = []
        self.y_train = []
        self.feature_names = []

    def fit(self, state: list[dict], label: bool) -> None:
        """
        Incorporates a new instance into the decision tree.
        The state is flattened, one-hot encoded, and the tree is retrained.
        """
        flat_state = _flatten_state_dt(state)
        self.X_train_dicts.append(flat_state)
        self.y_train.append(label)

        # Convert list of dicts to DataFrame
        df = pd.DataFrame(self.X_train_dicts)

        # FIX: Use one-hot encoding to handle categorical data (like 'heat', 'cool')
        df_encoded = pd.get_dummies(df)

        # Store the column names from the encoded training data
        self.feature_names = df_encoded.columns.tolist()

        self.tree.fit(df_encoded, self.y_train)
        print(f"DecisionTreeLearner: Retrained with {len(self.y_train)} examples.")

    def predict(self, state: list[dict]) -> bool:
        """
        Given a state, determines the class of the instance using the learned tree.
        Returns True if the method is applicable, False otherwise.
        """
        if not self.X_train_dicts:
            return False

        flat_state = _flatten_state_dt(state)

        # Create a DataFrame for the new instance
        instance_df = pd.DataFrame([flat_state])

        # FIX: One-hot encode the instance
        instance_df_encoded = pd.get_dummies(instance_df)

        # Reindex to ensure the instance has the exact same columns as the training data
        # This adds missing columns with a value of 0.
        instance_df_reindexed = instance_df_encoded.reindex(columns=self.feature_names, fill_value=0)

        try:
            prediction = self.tree.predict(instance_df_reindexed)
            return bool(prediction[0])
        except NotFittedError:
            return False
        except Exception as e:
            print(f"Error during prediction: {e}")
            return False

    def get_rules(self) -> list:
        """
        Extracts the rules from the trained decision tree.
        Returns a list of rules, where each rule is a list of conditions (dict).
        """
        if not hasattr(self.tree, 'tree_') or not self.feature_names or not self.X_train_dicts:
            return []

        # export_text provides a readable representation of the tree's rules
        text_rules = export_text(self.tree, feature_names=self.feature_names)

        # This is a simplified parser. A more robust solution might be needed for complex trees.
        rules = []
        for line in text_rules.split('\n'):
            if 'class:' in line:
                pred_class = int(line.split('class:')[-1].strip())
                # print(self.tree.classes_)
                # Find the predicted class for this leaf
                class_prediction = self.tree.classes_[pred_class]

                if class_prediction:  # Only care about paths that lead to a positive prediction
                    # Crude parsing of conditions from the text representation
                    conditions_str = line.strip().split('class:')[0].strip()
                    if conditions_str:
                        rule = []
                        # This parsing is basic and might need improvement for complex rules
                        # For now, it's a placeholder to show rule extraction is possible
                        # The core functionality relies on predict(), not this rule representation
                        rules.append([{'text_rule': conditions_str}])
        return rules


# class VSSMLearner(ConditionLearner):
#     """
#     A ConditionLearner that uses the Version Space Split/Merge (VSSM) algorithm.
#     """
#
#     def __init__(self):
#         super().__init__()
#         # self.vssm = VSSM(logging_enabled=False)
#
#     def fit(self, state: list[dict], label: bool) -> None:
#         """
#         Incorporates a new instance into the VSSM learner.
#         """
#         flat_state = _flatten_state_vssm(state)
#         self.vssm.ifit(flat_state, label)
#         print(f"VSSMLearner: Fitted with one {'positive' if label else 'negative'} example.")
#
#     def predict(self, state: list[dict]) -> bool:
#         """
#         Determines if the state is covered by the learned version spaces.
#         """
#         flat_state = _flatten_state_vssm(state)
#         return self.vssm.predict(flat_state)
#
#     def get_rules(self) -> list:
#         """
#         Extracts the learned concepts from the VSSM's S-sets and formats them
#         for the planner.
#         """
#         dnf = self.vssm.get_pyHTN_conds()
#         planner_rules = []
#         for conjunction in dnf:
#             rule = []
#             sorted_literals = sorted(conjunction, key=lambda x: x[1])
#             grouped_by_id = itertools.groupby(sorted_literals, key=lambda x: x[1])
#
#             for identifier, group in grouped_by_id:
#                 fact_pattern = {}
#                 for literal in group:
#                     attr_name = literal[0].split('.')[-1]
#                     if attr_name != 'id':
#                         fact_pattern[attr_name] = literal[2]
#
#                 if fact_pattern:
#                     fact_pattern['id'] = identifier
#                     rule.append(fact_pattern)
#
#             if rule:
#                 planner_rules.append(rule)
#
#         return planner_rules


class UnificationLearner(ConditionLearner):
    """
    A ConditionLearner that uses anti-unification to generalize a single
    conjunctive hypothesis.
    """

    def __init__(self):
        super().__init__()
        self.positive_examples = []
        self.negative_examples = []
        self.hypothesis = ([], [])
        self.var_counter = 0

    def _get_new_var(self):
        self.var_counter += 1
        return f'?v{self.var_counter - 1}'

    def _anti_unify_facts(self, fact1: dict, fact2: dict) -> (dict, int):
        generalized_fact = {}
        cost = 0
        all_keys = set(fact1.keys()) | set(fact2.keys())
        shared_keys = set(fact1.keys()) & set(fact2.keys())
        cost += len(all_keys - shared_keys)

        for key in shared_keys:
            if fact1[key] == fact2[key]:
                generalized_fact[key] = fact1[key]
            else:
                generalized_fact[key] = self._get_new_var()
                cost += 1

        if not generalized_fact:
            cost += 1

        return generalized_fact, cost

    def _anti_unify(self, hyp1: list[dict], hyp2: list[dict]) -> list[dict]:
        if not hyp1: return hyp2
        if not hyp2: return hyp1

        new_hypothesis = []
        for fact2 in hyp2:
            best_generalization = {}
            min_cost = float('inf')

            for fact1 in hyp1:
                generalized, cost = self._anti_unify_facts(fact1, fact2)
                if cost < min_cost:
                    min_cost = cost
                    best_generalization = generalized

            if best_generalization:
                new_hypothesis.append(best_generalization)

        return new_hypothesis

    def _specialize(self, neg_example: list[dict]):
        pos_patterns, neg_patterns = self.hypothesis
        current_pos_attrs = set()
        for fact in pos_patterns:
            current_pos_attrs.update(fact.keys())

        for neg_fact in neg_example:
            for key, value in neg_fact.items():
                if key not in current_pos_attrs:
                    new_neg_pattern = {key: value}
                    if new_neg_pattern not in neg_patterns:
                        self.hypothesis = (pos_patterns, neg_patterns + [new_neg_pattern])
                        print(f"UnificationLearner: Specialized by adding negative condition {new_neg_pattern}")
                        return

    def fit(self, state: list[dict], label: bool) -> None:
        if label:
            self.positive_examples.append(state)
            current_pos_patterns, _ = self.hypothesis
            new_pos_patterns = self._anti_unify(current_pos_patterns, state)
            self.hypothesis = (new_pos_patterns, self.hypothesis[1])
            print(f"UnificationLearner: Generalized hypothesis.")
        else:
            self.negative_examples.append(state)
            self._specialize(state)

    def predict(self, state: list[dict]) -> bool:
        pos_patterns, neg_patterns = self.hypothesis

        try:
            _, bindings = next(PatternMatch.match(pos_patterns, state))
            is_positive_match = True
        except StopIteration:
            is_positive_match = False

        if not is_positive_match:
            return False

        for neg_pattern in neg_patterns:
            try:
                next(PatternMatch.match([neg_pattern], state))
                return False
            except StopIteration:
                continue

        return True

    def get_rules(self) -> list:
        return [self.hypothesis[0]]


# --- Add after the Method class (around line 158) ---

class Filter:
    """
    Implements a filter on preconditions to exclude invalid states. 

    @param fn: A callable whose parameter names match variable names (without
        the leading '?') that will be bound during precondition matching.
    """
    def __init__(self, fn: callable):
        """
        @param fn: The guard function. Parameter names must match bound variable
            names (without the '?' prefix).
        @return: None
        """
        import inspect
        self.fn = fn
        self.params = list(inspect.signature(fn).parameters.keys())

    def evaluate(self, bindings: dict) -> bool:
        """
        Evaluates the filter given the current variable bindings.

        For each parameter name, the corresponding '?<name>' variable is looked
        up in *bindings*.  If any required variable is not yet bound, the filter
        returns False (cannot evaluate).

        @param bindings: A dict mapping '?<name>' strings to their ground values.
        @return: True if the guard condition holds, False otherwise.
        """
        kwargs = {}
        for param in self.params:
            var_key = f'?{param}'
            if var_key not in bindings:
                return False  # Variable not yet bound — cannot evaluate
            kwargs[param] = bindings[var_key]
        return bool(self.fn(**kwargs))


def unify(pattern1, pattern2, bindings=None):
    if bindings is None:
        bindings = {}

    if pattern1 == pattern2:
        return bindings

    if is_variable(pattern1):
        if pattern1 in bindings:
            return unify(bindings[pattern1], pattern2, bindings)
        else:
            bindings[pattern1] = pattern2
            return bindings

    if is_variable(pattern2):
        return unify(pattern2, pattern1, bindings)

    if isinstance(pattern1, (list, tuple)) and isinstance(pattern2, (list, tuple)):
        if len(pattern1) != len(pattern2):
            return None
        for p1, p2 in zip(pattern1, pattern2):
            bindings = unify(p1, p2, bindings)
            if bindings is None:
                return None
        return bindings

    return None


# In htn.py

def substitute(pattern, bindings):
    if not isinstance(pattern, (list, tuple, dict)):  # Modified this line slightly
        if is_variable(pattern):
            return bindings.get(pattern, pattern)
        return pattern


    # Added a case to handle dictionaries
    if isinstance(pattern, dict):
        new_dict = {}
        for key, value in pattern.items():
            # We substitute both keys and values if they are variables
            new_key = bindings.get(key, key) if is_variable(key) else key
            new_value = substitute(value, bindings)  # Recurse on the value
            new_dict[new_key] = new_value
        return new_dict

    # Original logic for lists and tuples
    result = []
    for item in pattern:
        # The recursive call now correctly handles all types, including dicts
        result.append(substitute(item, bindings))

    return type(pattern)(result)
