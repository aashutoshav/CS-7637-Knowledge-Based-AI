import random


class Rule(object):

    def __init__(self, conditions, seed):
        self.conditions = dict(conditions)
        self.seed = dict(seed)

    def matches(self, instance):
        for k, v in self.conditions.items():
            if instance.get(k) != v:
                return False
        return True


class LearningSystem(object):

    def __init__(self, k_memory=100):
        self.k_memory = k_memory
        self.H = []

    @staticmethod
    def _predict(instance, hypothesis):
        for rule in hypothesis:
            if rule.matches(instance):
                return 1
        return 0

    def _score(self, hypothesis, kset):
        if not kset:
            return 1.0
        correct = 0
        for inst, label in kset:
            if self._predict(inst, hypothesis) == label:
                correct += 1
        return correct / len(kset)

    @staticmethod
    def _minimal_generalization(rule, instance):
        new_conditions = {
            k: v for k, v in rule.conditions.items() if instance.get(k) == v
        }
        if not new_conditions:
            return None
        return Rule(new_conditions, rule.seed)

    @staticmethod
    def _minimal_specializations(rule, instance):
        candidates = []
        for k, v in rule.seed.items():
            if k not in rule.conditions and instance.get(k) != v:
                new_conditions = dict(rule.conditions)
                new_conditions[k] = v
                candidates.append(Rule(new_conditions, rule.seed))
        return candidates

    def _run_isc(self, train_set, train_labels, epochs):
        H = []

        first_pos_idx = None
        for idx, label in enumerate(train_labels):
            if label == 1:
                first_pos_idx = idx
                break

        if first_pos_idx is not None:
            first_pos = train_set[first_pos_idx]
            H.append(Rule(first_pos, first_pos))

        for _ in range(epochs):
            for i, (instance, label) in enumerate(zip(train_set, train_labels)):
                pred = self._predict(instance, H)
                if pred == label:
                    continue

                start_k = max(0, i - self.k_memory + 1)
                kset = list(zip(train_set[start_k:i + 1], train_labels[start_k:i + 1]))
                h_score = self._score(H, kset)

                best = None
                best_score = h_score

                if label == 1:
                    for term_idx, rule in enumerate(H):
                        revised = self._minimal_generalization(rule, instance)
                        if revised is None:
                            continue
                        candidate = list(H)
                        candidate[term_idx] = revised
                        score = self._score(candidate, kset)
                        if score > best_score:
                            best_score = score
                            best = candidate

                    new_term = Rule(instance, instance)
                    new_candidate = H + [new_term]
                    new_score = self._score(new_candidate, kset)
                    if new_score > best_score:
                        best_score = new_score
                        best = new_candidate
                else:
                    specialized = list(H)
                    changed = False
                    for term_idx, rule in enumerate(H):
                        if not rule.matches(instance):
                            continue
                        candidates = self._minimal_specializations(rule, instance)
                        if not candidates:
                            continue
                        best_g = None
                        best_g_score = -1.0
                        for g in candidates:
                            trial = list(specialized)
                            trial[term_idx] = g
                            score = self._score(trial, kset)
                            if score > best_g_score:
                                best_g_score = score
                                best_g = g
                        if best_g is not None:
                            specialized[term_idx] = best_g
                            changed = True
                    specialized_score = self._score(specialized, kset) if changed else -1.0

                    removed = [rule for rule in H if not rule.matches(instance)]
                    removed_score = self._score(removed, kset)

                    if removed_score > specialized_score and removed_score > best_score:
                        best_score = removed_score
                        best = removed
                    elif changed and specialized_score > best_score:
                        best_score = specialized_score
                        best = specialized

                if best is not None and best_score > h_score:
                    H = best

        return H

    def _full_accuracy(self, hypothesis, dataset, labels):
        if not dataset:
            return 0.0
        correct = 0
        for inst, lab in zip(dataset, labels):
            if self._predict(inst, hypothesis) == lab:
                correct += 1
        return correct / len(dataset)

    def fit(self, train_set, train_labels, epochs=3, restarts=7, seed=12345):
        rng = random.Random(seed)
        best_H = None
        best_acc = -1.0

        n = len(train_set)
        for r in range(restarts):
            order = list(range(n))
            if r > 0:
                rng.shuffle(order)
            shuffled_set = [train_set[i] for i in order]
            shuffled_labels = [train_labels[i] for i in order]

            H = self._run_isc(shuffled_set, shuffled_labels, epochs)
            acc = self._full_accuracy(H, train_set, train_labels)
            if acc > best_acc:
                best_acc = acc
                best_H = H

        self.H = best_H if best_H is not None else []

    def generate_rules(self, instance=None):
        rules = [rule for rule in self.H if len(rule.conditions) > 0]
        if instance is not None:
            rules = [rule for rule in rules if rule.matches(instance)]
        return [rule.conditions for rule in rules]


def define_model():
    return LearningSystem(k_memory=100)
