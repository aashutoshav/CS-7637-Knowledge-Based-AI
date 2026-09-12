from collections import deque
from random import choice


class ARCPuzzle:
    """
    An eight puzzle class that can be used to test different search algorithms.
    When first created the puzzle is in the solved state.
    """

    def __init__(self, state):
        self.state = tuple((tuple(row) for row in state))

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        if isinstance(other, ARCPuzzle):
            return self.state == other.state
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        out = ""
        for row in self.state:
            out += str(row) + "\n"
        return out

    def copy(self):
        """
        Makes a deep copy of an ARCPuzzle object.
        """
        new = ARCPuzzle(self.state)
        return new

    def randomize(self, num_shuffles):
        """
        Randomizes an ARCPuzzle by executing a random action `num_suffles`
        times.
        """
        for i in range(num_shuffles):
            actions = [a for a in self.legalActions()]
            a = choice([a for a in self.legalActions()])
            # print(actions, a)
            self.executeAction(a)

        return self

    def tophalf(self, grid):
        """ upper half """
        return grid[:len(grid) // 2]


    def rot90(self, grid):
        """ clockwise rotation by 90 degrees """
        return tuple(zip(*grid[::-1]))


    def hmirror(self, grid):
        """ mirroring along horizontal """
        return grid[::-1]

    def vmirror(self, grid):
        """ mirroring along horizontal """
        return tuple(reversed(grid))

    def lshift(self, grid):
        return tuple([tuple([e for e in row if e != 0] + [0]*row.count(0))
                      for row in grid])

    def compress(self, grid):
        """ removes frontiers """
        ri = [i for i, r in enumerate(grid) if len(set(r)) == 1]
        ci = [j for j, c in enumerate(zip(*grid)) if len(set(c)) == 1]
        return tuple([tuple([v for j, v in enumerate(r) if j not in ci])
                      for i, r in enumerate(grid) if i not in ri])

    def mapcolor(self, grid, a, b):
        return tuple(tuple(b if e == a else e for e in row) for row in grid)

    def trim(self, grid):
        """ removes border """
        return tuple(r[1:-1] for r in grid[1:-1])

    def executeAction(self, action):
        """
        Executes an action to the ARCPuzzle object.

        :param action: the action to execute
        :type action: "up", "left", "right", or "down"
        """
        if action == 'tophalf':
            self.state = self.tophalf(self.state)
        elif action == 'rot90':
            self.state = self.rot90(self.state)
        elif action == 'hmirror':
            self.state = self.hmirror(self.state)
        elif action == 'vmirror':
            self.state = self.vmirror(self.state)
        elif action == 'lshift':
            self.state = self.lshift(self.state)
        elif action == 'compress':
            self.state = self.compress(self.state)
        elif action[:8] == 'mapcolor':
            args = action[9:-1].split(',')
            self.state = self.mapcolor(self.state, int(args[0]), int(args[1]))

    def legalActions(self):
        """
        Returns an iterator to the legal actions that can be executed in the
        current state.
        """
        for action in ['tophalf', 'rot90', 'hmirror', 'vmirror', 'lshift', 'compress']:
            yield action

        for a in set(e for row in self.state for e in row):
            for b in range(10):
                if a == b:
                    continue
                yield f'mapcolor({a},{b})'


def arc_planning(input_grid, output_grid) -> list[str]:
    start = ARCPuzzle(input_grid)
    goal = ARCPuzzle(output_grid)

    if start == goal:
        return []

    frontier = deque([start])
    parents = {start: (None, None)} # {state: (prev_state, action that produced present state)}

    while frontier:
        current = frontier.popleft()

        for action in current.legalActions():
            successor = current.copy()
            successor.executeAction(action)

            if successor == current or successor in parents:
                continue

            parents[successor] = (current, action)

            if successor == goal:
                plan = []
                state = successor
                while parents[state][0] is not None:
                    previous, action_used = parents[state]
                    plan.append(action_used)
                    state = previous
                return list(reversed(plan))

            frontier.append(successor)

    return []

if __name__ == "__main__":

    initial = ARCPuzzle([[7, 0, 7], [7, 0, 7], [7, 7, 0]])
    initial_copy = initial.copy()
    goal = initial.copy()
    goal.randomize(3)

    print("Puzzle being solved:")
    print(initial)
    print("Goal =====>")
    print(goal)
    print("-------------------")

    ans = arc_planning(initial.state, goal.state)
    print(ans)
    print(initial_copy)
    for action in ans:
        initial_copy.executeAction(action)
        print(initial_copy)
