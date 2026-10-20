import random

try:
	_popcount = int.bit_count
except AttributeError:
	def _popcount(value):
		return bin(value).count('1')


class Rule(object):

	def __init__(self, conditions, seed):
		self.conditions = dict(conditions)
		self.seed = dict(seed)

	def matches(self, instance, lenient=True):
		for key, value in self.conditions.items():
			if key in instance:
				if instance[key] != value:
					return False
			elif not lenient:
				return False
		return True

	def signature(self):
		return frozenset(self.conditions.items())


class Index(object):

	def __init__(self, instances, labels, lenient=True):
		self.lenient = lenient
		self.full = (1 << len(instances)) - 1
		present = {}
		equal = {}
		positive = 0
		for i, instance in enumerate(instances):
			bit = 1 << i
			if labels[i]:
				positive |= bit
			for key, value in instance.items():
				present[key] = present.get(key, 0) | bit
				pair = (key, value)
				equal[pair] = equal.get(pair, 0) | bit
		self.positive = positive
		self.negative = self.full & ~positive
		self.absent = {key: self.full & ~mask for key, mask in present.items()}
		self.equal = equal
		self.literals = {}

	def literal(self, key, value):
		pair = (key, value)
		mask = self.literals.get(pair)
		if mask is None:
			mask = self.equal.get(pair, 0)
			if self.lenient:
				mask |= self.absent.get(key, self.full)
			self.literals[pair] = mask
		return mask

	def cover(self, conditions):
		mask = self.full
		for key, value in conditions.items():
			mask &= self.literal(key, value)
		return mask

	def correct(self, cover, window=None):
		if window is None:
			return _popcount(cover & self.positive) + _popcount(self.negative & ~cover)
		return _popcount(cover & window & self.positive) + _popcount(window & self.negative & ~cover)


class LearningSystem(object):

	def __init__(self, k=100, runs=6, epochs=2, folds=3, fractions=(0.002, 0.005, 0.01, 0.02), seed=7):
		self.k = k
		self.runs = runs
		self.epochs = epochs
		self.folds = folds
		self.fractions = fractions
		self.seed = seed
		self.hypothesis = []
		self.lenient = True

	@staticmethod
	def _label(value):
		return 1 if value == 1 or str(value).strip() == '1' else 0

	@staticmethod
	def _or(masks):
		mask = 0
		for value in masks:
			mask |= value
		return mask

	@staticmethod
	def _union(index, rules):
		mask = 0
		for rule in rules:
			mask |= index.cover(rule.conditions)
		return mask

	@staticmethod
	def _generalize(rule, instance, lenient):
		conditions = {}
		for key, value in rule.conditions.items():
			if key in instance:
				if instance[key] == value:
					conditions[key] = value
			elif lenient:
				conditions[key] = value
		if not conditions:
			return None
		return Rule(conditions, rule.seed)

	@staticmethod
	def _specializations(rule, instance, lenient):
		options = []
		for key, value in rule.seed.items():
			if key in rule.conditions:
				continue
			if key in instance:
				if instance[key] == value:
					continue
			elif lenient:
				continue
			options.append((key, value))
		return options

	def _isc(self, instances, labels, lenient):
		index = Index(instances, labels, lenient)
		first = None
		for i in range(len(instances)):
			if labels[i] and instances[i]:
				first = i
				break
		if first is None:
			return []
		hypothesis = [Rule(instances[first], instances[first])]
		covers = [index.cover(hypothesis[0].conditions)]
		for _ in range(self.epochs):
			for i, instance in enumerate(instances):
				bit = 1 << i
				label = labels[i]
				union = self._or(covers)
				if bool(union & bit) == bool(label):
					continue
				low = max(0, i - self.k + 1)
				window = ((1 << (i + 1)) - 1) ^ ((1 << low) - 1)
				current = index.correct(union, window)
				best = None
				best_covers = None
				best_score = -1
				new = None
				new_covers = None
				if label:
					for j, rule in enumerate(hypothesis):
						revised = self._generalize(rule, instance, lenient)
						if revised is None:
							continue
						mask = index.cover(revised.conditions)
						score = index.correct(self._or(covers[:j] + covers[j + 1:]) | mask, window)
						if score > best_score:
							best = hypothesis[:j] + [revised] + hypothesis[j + 1:]
							best_covers = covers[:j] + [mask] + covers[j + 1:]
							best_score = score
					if instance:
						new = hypothesis + [Rule(instance, instance)]
						new_covers = covers + [index.cover(instance)]
				else:
					best = list(hypothesis)
					best_covers = list(covers)
					for j, rule in enumerate(hypothesis):
						if not covers[j] & bit:
							continue
						rest = self._or(best_covers[:j] + best_covers[j + 1:])
						choice = None
						choice_cover = 0
						choice_score = -1
						for key, value in self._specializations(rule, instance, lenient):
							mask = covers[j] & index.literal(key, value)
							score = index.correct(rest | mask, window)
							if score > choice_score:
								choice = (key, value)
								choice_cover = mask
								choice_score = score
						if choice is None:
							best[j] = None
						else:
							conditions = dict(rule.conditions)
							conditions[choice[0]] = choice[1]
							best[j] = Rule(conditions, rule.seed)
						best_covers[j] = choice_cover
					keep = [j for j in range(len(best)) if best[j] is not None]
					best = [best[j] for j in keep]
					best_covers = [best_covers[j] for j in keep]
					best_score = index.correct(self._or(best_covers), window)
					keep = [j for j in range(len(hypothesis)) if not covers[j] & bit]
					new = [hypothesis[j] for j in keep]
					new_covers = [covers[j] for j in keep]
				if new is not None:
					new_score = index.correct(self._or(new_covers), window)
					if new_score > best_score:
						best = new
						best_covers = new_covers
						best_score = new_score
				if best is not None and best_score > current:
					hypothesis = best
					covers = best_covers
		return hypothesis

	def _refine(self, index, rule, base):
		conditions = dict(rule.conditions)
		cover = index.cover(conditions)
		score = index.correct(base | cover)
		while True:
			move = None
			move_cover = None
			move_score = score
			if len(conditions) > 1:
				for key in conditions:
					trial = dict(conditions)
					del trial[key]
					mask = index.cover(trial)
					value = index.correct(base | mask)
					if value > move_score or (move is None and value == move_score):
						move = (key, None)
						move_cover = mask
						move_score = value
			for key, value in rule.seed.items():
				if key in conditions:
					continue
				mask = cover & index.literal(key, value)
				result = index.correct(base | mask)
				if result > move_score:
					move = (key, value)
					move_cover = mask
					move_score = result
			if move is None:
				break
			if move[1] is None:
				del conditions[move[0]]
			else:
				conditions[move[0]] = move[1]
			cover = move_cover
			score = move_score
		for key, value in rule.seed.items():
			if key in conditions:
				continue
			mask = cover & index.literal(key, value)
			result = index.correct(base | mask)
			if result >= score:
				conditions[key] = value
				cover = mask
				score = result
		return Rule(conditions, rule.seed), cover, score

	def _polish(self, index, rules, threshold):
		rules = list(rules)
		covers = [index.cover(rule.conditions) for rule in rules]
		for _ in range(8 * len(rules) + 8):
			changed = False
			order = sorted(range(len(rules)), key=lambda j: _popcount(covers[j] & index.positive))
			for j in order:
				rest = self._or(covers[:j] + covers[j + 1:])
				refined, cover, score = self._refine(index, rules[j], rest)
				if len(rules) > 1 and score - index.correct(rest) < threshold:
					del rules[j]
					del covers[j]
					changed = True
					break
				if refined.signature() != rules[j].signature():
					rules[j] = refined
					covers[j] = cover
					changed = True
					break
			if not changed:
				break
		return rules

	def _forward(self, index, pool, threshold):
		selected = []
		base = 0
		base_score = index.correct(0)
		while pool:
			best = None
			for rule in pool:
				refined, cover, score = self._refine(index, rule, base)
				if best is None or score > best[2]:
					best = (refined, cover, score)
			if best[2] - base_score < threshold:
				break
			selected.append(best[0])
			base |= best[1]
			base_score = best[2]
			pool = [rule for rule in pool if index.cover(rule.conditions) & ~base & index.positive]
		return selected

	def _learn(self, instances, labels, fractions, lenient):
		index = Index(instances, labels, lenient)
		rng = random.Random(self.seed)
		order = list(range(len(instances)))
		hypotheses = []
		pool = {}
		for run in range(self.runs):
			if run:
				rng.shuffle(order)
			hypothesis = self._isc([instances[i] for i in order], [labels[i] for i in order], lenient)
			if hypothesis:
				hypotheses.append(hypothesis)
			for rule in hypothesis:
				pool.setdefault(rule.signature(), rule)
		results = []
		for fraction in fractions:
			threshold = max(3, int(round(fraction * len(instances))))
			contenders = [self._polish(index, hypothesis, threshold) for hypothesis in hypotheses]
			forward = self._forward(index, list(pool.values()), threshold)
			if forward:
				contenders.append(self._polish(index, forward, threshold))
			best = []
			best_rank = None
			for rules in contenders:
				rank = (index.correct(self._union(index, rules)), -len(rules), -sum(len(rule.conditions) for rule in rules))
				if best_rank is None or rank > best_rank:
					best = rules
					best_rank = rank
			results.append(best)
		return results

	def _choose(self, instances, labels, lenient):
		rng = random.Random(self.seed + 1)
		groups = [[i for i in range(len(instances)) if labels[i] == target] for target in (0, 1)]
		fold = [0] * len(instances)
		for group in groups:
			rng.shuffle(group)
			for position, i in enumerate(group):
				fold[i] = position % self.folds
		totals = [0] * len(self.fractions)
		for f in range(self.folds):
			train = [i for i in range(len(instances)) if fold[i] != f]
			test = [i for i in range(len(instances)) if fold[i] == f]
			results = self._learn([instances[i] for i in train], [labels[i] for i in train], self.fractions, lenient)
			for j, rules in enumerate(results):
				for i in test:
					predicted = any(rule.matches(instances[i], lenient) for rule in rules)
					if predicted == bool(labels[i]):
						totals[j] += 1
		best = 0
		for j in range(1, len(self.fractions)):
			if totals[j] >= totals[best]:
				best = j
		return self.fractions[best], totals[best] / float(len(instances))

	def fit(self, train_set, train_labels):
		instances = [dict(instance) for instance in train_set]
		labels = [self._label(label) for label in train_labels]
		if len(instances) < 20 * self.folds:
			self.lenient = True
			self.hypothesis = self._learn(instances, labels, [self.fractions[-1]], True)[0]
			return
		lenient_fraction, lenient_score = self._choose(instances, labels, True)
		keys = set()
		for instance in instances:
			keys.update(instance)
		if any(len(instance) < len(keys) for instance in instances):
			strict_fraction, strict_score = self._choose(instances, labels, False)
		else:
			strict_fraction, strict_score = lenient_fraction, -1.0
		if strict_score > lenient_score:
			self.lenient = False
			self.hypothesis = self._learn(instances, labels, [strict_fraction], False)[0]
		else:
			self.lenient = True
			self.hypothesis = self._learn(instances, labels, [lenient_fraction], True)[0]

	def generate_rules(self, instance=None):
		return [dict(rule.conditions) for rule in self.hypothesis]


def define_model():
	return LearningSystem()
