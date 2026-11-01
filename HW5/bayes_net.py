from itertools import product


class BayesNet:
    """
    Feel free to add additional internal methods to this class.
    """

    def __init__(self):
        """
        Each node maps to (parents, cpt). The cpt maps a tuple of parent values
        (in the order of `parents`) to P(node = True | parents).
        """
        self.nodes = {
            "Necronia flu": ((), {(): 0.8}),
            "Typhonus worm": ((), {(): 0.05}),
            "Scleraitis": ((), {(): 0.25}),
            "Hydrophobia": (
                ("Necronia flu",),
                {(True,): 0.95, (False,): 0.01},
            ),
            "Fatigue": (
                ("Hydrophobia",),
                {(True,): 0.9, (False,): 0.3},
            ),
            "Mania": (
                ("Hydrophobia", "Typhonus worm"),
                {
                    (True, True): 0.99,
                    (True, False): 0.4,
                    (False, True): 0.7,
                    (False, False): 0.05,
                },
            ),
            "Cysts": (
                ("Typhonus worm",),
                {(True,): 0.6, (False,): 0.5},
            ),
            "Tumors": (
                ("Cysts", "Scleraitis"),
                {
                    (True, True): 0.85,
                    (True, False): 0.7,
                    (False, True): 0.7,
                    (False, False): 0.15,
                },
            ),
            "Skin lesions": (
                ("Scleraitis",),
                {(True,): 0.55, (False,): 0.2},
            ),
            "Necrosis": (
                ("Skin lesions", "Scleraitis"),
                {
                    (True, True): 0.9,
                    (True, False): 0.25,
                    (False, True): 0.35,
                    (False, False): 0.1,
                },
            ),
        }
        self.names = list(self.nodes)
        self._lookup = {n.lower(): n for n in self.names}

    def _canonical(self, name: str) -> str:
        return self._lookup[name.strip().lower()]

    def _joint(self, assignment: dict) -> float:
        p = 1.0
        for node, (parents, cpt) in self.nodes.items():
            p_true = cpt[tuple(assignment[par] for par in parents)]
            p *= p_true if assignment[node] else 1.0 - p_true
        return p

    def _probability_of_evidence(self, evidence: dict) -> float:
        hidden = [n for n in self.names if n not in evidence]
        total = 0.0
        for values in product((True, False), repeat=len(hidden)):
            assignment = dict(evidence)
            assignment.update(zip(hidden, values))
            total += self._joint(assignment)
        return total

    def query(self, variable: str, evidence: dict[str, bool]) -> float:
        variable = self._canonical(variable)
        evidence = {self._canonical(k): bool(v) for k, v in evidence.items()}

        if variable in evidence:
            return 1.0 if evidence[variable] else 0.0

        numerator = self._probability_of_evidence({**evidence, variable: True})
        denominator = self._probability_of_evidence(evidence)
        return numerator / denominator


if __name__ == "__main__":

    net = BayesNet()
    print(net.query('Typhonus worm', {'Mania': True, 'Tumors': False}))
