from py_search.base import Problem
from py_search.base import Node
from py_search.informed import best_first_search


class OBJECT_COST:
    BLANK_CELL = 0
    WALL = float('inf') # walls are impassable; no successor is generated for wall cells
    CHALLENGE = 1       # monsters, traps, stones, rocks (low cost so agent passes through toward goal)
    SHRINE = 0          # shrine objects (attractive)
    GOAL = 0            # tower/goal objects (attractive)
    #REVISIT_PENALTY = 0.5


class MovePlanner:
    def __init__(self):
        pass

    def get_next_move(self, src, dest, scene, game_data, visited=None, max_steps=None):
        """ Plans the next move from src to dest using A* search.
        @param src: Tuple (x, y) of the agent's current position.
        @param dest: Tuple (x, y) of the target position.
        @param scene: List of entity dicts on the board.
        @param game_data: Dict with board metadata (boardWidth, boardHeight).
        @param visited: Optional Counter mapping (x, y) -> visit count for revisit penalties.
        @param max_steps: Optional int capping the maximum path length (e.g. action points).
        @return: Tuple of action strings forming the path, or None if no path found.
        """
        new_state = self._get_new_state(scene, game_data, visited, max_steps)

        problem = DiceAdventureRouteProblem(src, goal=dest, extra=new_state)
        try:
            sol = next(best_first_search(problem))
            path = sol.path()
            cost = sol.cost()
            if path:
                return path
            else:
                return None

        except StopIteration:
            return None

    @staticmethod
    def _get_new_state(scene, game_data, visited=None, max_steps=None):
        """ Builds the search state from the scene and game data.
        @param scene: List of entity dicts on the board.
        @param game_data: Dict with board metadata.
        @param visited: Optional Counter of visited positions.
        @param max_steps: Optional int capping maximum path length.
        @return: Dict containing scene, gameData, positions lookup, and optional extra data.
        """
        new_state = {
            "scene": scene,
            "gameData": game_data,
            "positions": {},
        }
        if visited is not None:
            new_state["visited"] = visited
        if max_steps is not None:
            new_state["max_steps"] = max_steps
        for obj in new_state["scene"]:
            x, y = obj.get("x"), obj.get("y")
            if x is not None and y is not None:
                if (x, y) not in new_state["positions"]:
                    new_state["positions"][(x, y)] = [obj]
                else:
                    new_state["positions"][(x, y)].append(obj)
        return new_state


class DiceAdventureRouteProblem(Problem):

    def node_value(self, node):
        """
        Returns f(n) = g(n) + h(n) for A* search.
        g(n) is the accumulated path cost stored in the node.
        h(n) is the Manhattan distance heuristic to the goal.
        @param node: The current search node.
        @return: The estimated total cost f(n) used to prioritize the search.
        """
        goal = self.goal.state
        h = abs(node.state[0] - goal[0]) + abs(node.state[1] - goal[1])
        return node.cost() + h

    def successors(self, node):
        """
        Yields valid successor nodes for A* search. Invalid moves (out of
        bounds) are skipped. Cells containing a Wall entity are skipped entirely
        — no path may pass through them. Positions already in the current path
        are excluded to prevent backtracking. Step cost includes a base cost of
        1, a penalty for hazards (monsters, stones, traps, rocks), and an
        optional revisit penalty for previously visited cells.
        @param node: The current search node containing state, cost, and extra data.
        @return: Generator of successor Node objects with accumulated path cost.
        """
        curr_pos = node.state
        actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        action_map = {(-1, 0): "left", (1, 0): "right", (0, 1): "up", (0, -1): "down"}
        positions = node.extra["positions"]
        hazard_types = {"monster", "stone", "trap", "rock"}

        for action in actions:
            new_pos = (curr_pos[0] + action[0], curr_pos[1] + action[1])

            # Skip if new position is out of bounds
            if new_pos[0] < 0 or new_pos[0] >= node.extra["gameData"]["boardWidth"] or \
                    new_pos[1] < 0 or new_pos[1] >= node.extra["gameData"]["boardHeight"]:
                continue

            # Skip if new position has a wall
            if new_pos in positions:
                if any(str(obj.get("entityType")).lower() == "wall" for obj in positions[new_pos]):
                    continue

            # Base step cost
            total_cost = 1

            # Add costs/discounts based on entity types at the new position
            if new_pos in positions:
                for obj in positions[new_pos]:
                    entity_type = str(obj.get("entityType", "")).lower()
                    if entity_type in {"monster", "stone", "trap", "rock"}:
                        total_cost += OBJECT_COST.CHALLENGE
                    # elif entity_type == "shrine":
                    #     total_cost += OBJECT_COST.SHRINE
                    # elif entity_type == "goal":
                    #     total_cost += OBJECT_COST.GOAL

            # Add revisit penalty scaled by visit count
            # visited = node.extra.get("visited")
            # if visited is not None and new_pos in visited:
            #     total_cost += OBJECT_COST.REVISIT_PENALTY * visited[new_pos]

            path_cost = node.cost() + total_cost
            yield Node(new_pos, node, action_map[action], path_cost, extra=node.extra)

    def goal_test(self, state_node, goal_node=None):
        """ Returns True if the state_node is at the goal position, or if
        the path has reached the max_steps cap (treating current pos as goal).
        @param state_node: The current search node.
        @param goal_node: Optional explicit goal node.
        @return: True if goal condition is met.
        """
        if goal_node is None:
            goal = self.goal
        else:
            goal = goal_node.state
        pos = state_node.state
        # Primary goal: reached destination
        if pos[0] == goal[0] and pos[1] == goal[1]:
            return True
        # Secondary goal: path length has hit the action point cap
        max_steps = state_node.extra.get("max_steps")
        if max_steps is not None and state_node.depth() >= max_steps:
            return True
        return False

