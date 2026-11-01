import random
from functools import reduce
from random import randint
from traceback import format_exc

from environment.htn.htn_elements import Fact, Method, Task
from environment.htn.htn import HTN, FailedPlanException
from environment.dice_adventure_env import GameOverException, LevelChangedException
from utils import get_timestamp_string


class DiceAdventureAgent:
    """
    Provides a uniform interface to connect agents to Dice Adventure environment.
    Developers must implement the take_action() function.
    - init():         Initialize any needed variables.
    - take_action():  Determines which action to take given a state (dict) and list of actions. Note that your
                      agent does not need to use the list of actions, it is just provided for convenience.
    """

    def __init__(self, character_name: str, character_id: str, env):
        """
        Initialize any needed variables. The character_name and character_id arguments specify the name and ID
        of the character this agent will play as.

        @param character_name: The character the agent will play as.
        @param character_id: The character ID corresponding to the character.
        @param env: A DiceAdventureEnv instance this agent will interact with.
        @return: None

        Player code index:
        {
            "Dwarf" : "C11",
            "Giant" : "C21",
            "Human" : "C31"
        }
        """
        self.character_name = character_name  # The character's human-readable name
        self.character_id = character_id  # The character's ID used in the game state
        self.env = env
        self.htn_planner = HTN(
            env=env,
            domain=self.get_domain(),
            operators=env.get_env_actions(),
            preserve_tasks=True,
            trace_options={"live": False, "include_states": True}
        )

    def play(self) -> None:
        """
        Invokes the HTN planner to decompose and execute the 'play' task until the game ends.
        On a level transition, the HTN task queue is cleared and planning restarts
        from 'play' so stale locked-in exploration targets from the previous level
        do not carry over.

        @return: None
        @raises GameOverException: Propagated when the game server reports
                                   GAME_OVER status.
        """
        while True:

            try:
                self.htn_planner.reset()
                self.htn_planner.plan("play")
            except LevelChangedException as e:
                pass

    def get_domain(self) -> dict:
        """
        Implements an HTN domain consisting of tasks, methods, and operators.

        Strategy: skip pinning (pins cost AP and the A* pin operator walks the character),
        then each planning round head for (1) this character's shrine if known, (2) the tower
        once the shrine is reached, (3) the nearest unexplored cell otherwise. Routes come from a
        hazard-aware Dijkstra; the HTN 'move' operator is given the waypoint reachable this turn.

        @return: A dict mapping task name strings to lists of Method objects.
        """
        import heapq
        from environment.move_planner import MovePlanner

        me = self.character_id
        planner = self  
        def die_values(die):
            """'D6+2' -> list of equally likely totals; None if unparseable."""
            try:
                n, m = str(die).upper().lstrip("D").split("+")
                n, m = int(n), int(m)
                return [m] if n == 0 else [v + m for v in range(1, n + 1)]
            except ValueError:
                return None

        def win_prob(mine, theirs):
            """Probability our roll >= the challenge roll (ties go to the player)."""
            a, b = die_values(mine), die_values(theirs)
            if not a or not b:
                return 0.5
            return sum(x >= y for x in a for y in b) / (len(a) * len(b))

        die_key = {"Rock": "stoneDie", "Stone": "stoneDie", "Trap": "trapDie", "monster": "monsterDie"}
        cache = {"state": None, "decision": None, "ping_state": None, "ping": None}
        mem = {"level": None}
        pin_for = {"C11": "a", "C21": "b", "C31": "c", "tower": "d"}

        def level_memory(game):
            if mem["level"] != game["currLevel"]:
                mem.clear()
                mem.update(level=game["currLevel"], seen=set(), walls=set(), tower=None, blocked=set(),
                           last=None, pins={}, pinned=set())
            return mem

        def read_pins(state):
            """Remembers teammates' pins (they are cleared every round) as type -> location."""
            for o in state:
                if o.get("entityType") == "Pin" and o.get("placedBy") != me and o.get("x") is not None \
                        and o.get("y") is not None:
                    mem["pins"][str(o.get("objKey", "P?"))[-1].lower()] = (o["x"], o["y"])

        def decide(state):
            """Returns (mode, waypoint) for the current state; cached per state object."""
            if cache["state"] is state:
                return cache["decision"]
            cache["state"] = state
            try:
                cache["decision"] = _decide(state)
            except Exception:
                cache["decision"] = ("stay", None)
            return cache["decision"]

        def _decide(state):
            game = state[0]
            width, height = game["boardWidth"], game["boardHeight"]
            player = next((o for o in state if o.get("id") == me), None)
            if player is None or "x" not in player:
                return ("stay", None)
            pos = (player["x"], player["y"])
            ap = max(1, int(player.get("actionPoints", 1)))
            sight = int(player.get("sightRange", 1))

            level_memory(game)
            read_pins(state)
            walls, seen, cost, shrine, tower = mem["walls"], mem["seen"], {}, None, mem["tower"]
            impassable = set()  
            for sx in range(max(0, pos[0] - sight), min(width, pos[0] + sight + 1)):
                for sy in range(max(0, pos[1] - sight), min(height, pos[1] + sight + 1)):
                    seen.add((sx, sy))
            for o in state:
                x, y = o.get("x"), o.get("y")
                if x is None or y is None or "sight_status" not in o:
                    continue
                et = o.get("entityType")
                if o["sight_status"] != "unexplored" and et not in ("Cell", "Pin", "Character"):
                    seen.add((x, y))
                if et == "Wall":
                    walls.add((x, y))
                elif "challengeDie" in o:
                    p = win_prob(player.get(die_key.get(et, "stoneDie")), o["challengeDie"])
                    if p == 0:
                        impassable.add((x, y))
                    elif et == "Trap" or et == "monster":
                        cost[(x, y)] = max(cost.get((x, y), 0), 1 + 8 * (1 - p))
                    else:
                        cost[(x, y)] = max(cost.get((x, y), 0), 1 + 3 * (1 / p - 1))
                elif et == "Shrine" and o.get("character") == me and not o.get("reached"):
                    shrine = (x, y)
                elif et in ("Goal", "Tower") or o.get("objKey") == "**":
                    tower = mem["tower"] = (x, y)
            my_shrine_done = any(o.get("entityType") == "Shrine" and o.get("character") == me
                                 and o.get("reached") for o in state)
            shrine_seen = any(o.get("entityType") == "Shrine" and o.get("character") == me for o in state)
            pinned_shrine = mem["pins"].get(pin_for[me]) if me in pin_for else None
            if pinned_shrine and pinned_shrine in seen and not shrine_seen:
                mem["pins"].pop(pin_for[me])
                pinned_shrine = None
            pinned_tower = mem["pins"].get(pin_for["tower"])
            if pinned_tower and pinned_tower in seen and pinned_tower != tower:
                mem["pins"].pop(pin_for["tower"])
                pinned_tower = None

            def route(is_goal):
                """Dijkstra over non-wall cells; returns path (list of cells, excl. start) to the cheapest goal."""
                dist, prev, heap = {pos: 0}, {}, [(0, pos)]
                while heap:
                    d, cur = heapq.heappop(heap)
                    if d > dist.get(cur, 1e9):
                        continue
                    if cur != pos and is_goal(cur):
                        path = [cur]
                        while path[-1] in prev and prev[path[-1]] != pos:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nxt = (cur[0] + dx, cur[1] + dy)
                        if not (0 <= nxt[0] < width and 0 <= nxt[1] < height) or nxt in walls \
                                or nxt in impassable:
                            continue
                        nd = d + 1 + cost.get(nxt, 0)
                        if nd < dist.get(nxt, 1e9):
                            dist[nxt], prev[nxt] = nd, cur
                            heapq.heappush(heap, (nd, nxt))
                return None

            def explore_route():
                """Path to the reachable cell revealing the most unseen cells per unit of travel cost."""
                dist, prev, heap = {pos: 0}, {}, [(0, pos)]
                while heap:
                    d, cur = heapq.heappop(heap)
                    if d > dist.get(cur, 1e9):
                        continue
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nxt = (cur[0] + dx, cur[1] + dy)
                        if not (0 <= nxt[0] < width and 0 <= nxt[1] < height) or nxt in walls \
                                or nxt in impassable:
                            continue
                        nd = d + 1 + cost.get(nxt, 0)
                        if nd < dist.get(nxt, 1e9):
                            dist[nxt], prev[nxt] = nd, cur
                            heapq.heappush(heap, (nd, nxt))
                best, best_score = None, 0
                for cell, d in dist.items():
                    if cell == pos or cell in mem["blocked"]:
                        continue
                    gain = sum((cx, cy) not in seen
                               for cx in range(max(0, cell[0] - sight), min(width, cell[0] + sight + 1))
                               for cy in range(max(0, cell[1] - sight), min(height, cell[1] + sight + 1)))
                    score = gain / (d + 1)
                    if score > best_score:
                        best, best_score = cell, score
                if best is None:
                    return None
                path = [best]
                while prev[path[-1]] != pos:
                    path.append(prev[path[-1]])
                return path[::-1]

            steps = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

            def waypoint(path):
                """Furthest cell on our path (within AP) that the env's own A* 'move' reaches along our path.
                Without this, 'move' re-plans through hazards our route deliberately avoids."""
                for i in range(min(ap, len(path)) - 1, 0, -1):
                    actions = MovePlanner().get_next_move(src=pos, dest=path[i], scene=state[1:],
                                                          game_data=game, max_steps=ap) or ()
                    cells, cur = [], pos
                    for a in actions:
                        cur = (cur[0] + steps[a][0], cur[1] + steps[a][1])
                        cells.append(cur)
                    if cells == path[:i + 1]:
                        return path[i]
                return path[0]

            last = mem["last"]
            if last and last[0] == game["currRound"] - 1 and last[1] == pos and last[2] == "explore":
                mem["blocked"].add(last[3])

            if shrine and not my_shrine_done:
                if pos == shrine:
                    return ("stay", pos)
                path = route(lambda c: c == shrine)
                if path:
                    return ("shrine", waypoint(path))
            elif pinned_shrine and not my_shrine_done and not shrine_seen:
                path = route(lambda c: c == pinned_shrine)
                if path:
                    return ("pin", waypoint(path))
            if my_shrine_done and not tower and pinned_tower:
                path = route(lambda c: c == pinned_tower)
                if path:
                    return ("pin", waypoint(path))
            if my_shrine_done and tower:
                if pos == tower:
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nb = (pos[0] + dx, pos[1] + dy)
                        if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in walls:
                            return ("tower", nb)
                    return ("stay", pos)
                path = route(lambda c: c == tower)
                if path:
                    return ("tower", waypoint(path))
            path = explore_route()
            if path:
                mem["last"] = (game["currRound"], pos, "explore", path[-1])
                return ("explore", waypoint(path))
            return ("stay", pos)

        def choose_ping(state):
            """(pin type, x, y) for one teammate shrine or the tower we can see and have not pinned yet."""
            if cache["ping_state"] is state:
                return cache["ping"]
            cache["ping_state"], cache["ping"] = state, None
            try:
                cache["ping"] = _choose_ping(state)
            except Exception:
                cache["ping"] = None
            return cache["ping"]

        def _choose_ping(state):
            game = state[0]
            level_memory(game)
            player = next((o for o in state if o.get("id") == me), None)
            if player is None or "x" not in player or int(player.get("actionPoints", 0)) < 2:
                return None
            if mem.get("ping_round") == game["currRound"]:
                return None
            pos = (player["x"], player["y"])
            for o in state:
                et = o.get("entityType")
                if o.get("x") is None or o.get("y") is None:
                    continue
                if et == "Shrine" and o.get("character") != me and not o.get("reached"):
                    kind = pin_for.get(o.get("character"))
                elif et == "Goal":
                    kind = pin_for["tower"]
                else:
                    continue
                target = (o["x"], o["y"])
                if kind is None or (kind, target) in mem["pinned"]:
                    continue
                actions = MovePlanner().get_next_move(src=pos, dest=target, scene=state[1:], game_data=game,
                                                      max_steps=999)
                if target != pos and not actions:
                    continue
                mem["pinned"].add((kind, target))
                mem["ping_round"] = game["currRound"]
                return (kind,) + target
            return None

        def ping_is(kind, gx, gy):
            return choose_ping(planner.htn_planner.working_memory.state) == (kind, gx, gy)

        def ping_method(kind, label):
            return Method(
                name="Pin" + label,
                args=(),
                preconditions=[
                    {"currentPhase": "Player_Pinning"},
                    {"id": me, "ready": False, "dead": False},
                    {"entityType": "Cell", "x": "?gx", "y": "?gy"},
                    {"filter": lambda gx, gy: ping_is(kind, gx, gy)},
                ],
                subtasks=[("pin", kind, "?gx", "?gy"), ("play",)],
            )

        def mode_is(mode, gx, gy):
            state = planner.htn_planner.working_memory.state
            m, wp = decide(state)
            return m == mode and wp == (gx, gy)

        domain = {
            "play": [
                ping_method("a", "DwarfShrine"),
                ping_method("b", "GiantShrine"),
                ping_method("c", "HumanShrine"),
                ping_method("d", "Tower"),
                Method(
                    name="EndPinning",
                    args=(),
                    preconditions=[
                        {"currentPhase": "Player_Pinning"},
                        {"id": me, "ready": False},
                    ],
                    subtasks=[("end_pinning",), ("play",)],
                ),
                Method(
                    name="GoToShrine",
                    args=(),
                    preconditions=[
                        {"currentPhase": "Player_Planning"},
                        {"id": me, "ready": False, "dead": False},
                        {"entityType": "Shrine", "character": me, "reached": False},
                        {"entityType": "Cell", "x": "?gx", "y": "?gy"},
                        {"filter": lambda gx, gy: mode_is("shrine", gx, gy)},
                    ],
                    subtasks=[("move", "?gx", "?gy"), ("play",)],
                ),
                Method(
                    name="GoToTower",
                    args=(),
                    preconditions=[
                        {"currentPhase": "Player_Planning"},
                        {"id": me, "ready": False, "dead": False},
                        {"entityType": "Shrine", "character": me, "reached": True},
                        {"entityType": "Goal"},
                        {"entityType": "Cell", "x": "?gx", "y": "?gy"},
                        {"filter": lambda gx, gy: mode_is("tower", gx, gy)},
                    ],
                    subtasks=[("move", "?gx", "?gy"), ("play",)],
                ),
                Method(
                    name="FollowTeammatePin",
                    args=(),
                    preconditions=[
                        {"currentPhase": "Player_Planning"},
                        {"id": me, "ready": False, "dead": False},
                        {"entityType": "Cell", "x": "?gx", "y": "?gy"},
                        {"filter": lambda gx, gy: mode_is("pin", gx, gy)},
                    ],
                    subtasks=[("move", "?gx", "?gy"), ("play",)],
                ),
                Method(
                    name="Explore",
                    args=(),
                    preconditions=[
                        {"currentPhase": "Player_Planning"},
                        {"id": me, "ready": False, "dead": False},
                        {"entityType": "Cell", "x": "?gx", "y": "?gy"},
                        {"filter": lambda gx, gy: mode_is("explore", gx, gy)},
                    ],
                    subtasks=[("move", "?gx", "?gy"), ("play",)],
                ),
                Method(
                    name="HoldPosition",
                    args=(),
                    preconditions=[
                        {"currentPhase": "Player_Planning"},
                        {"id": me, "ready": False, "dead": False, "x": "?px", "y": "?py"},
                    ],
                    subtasks=[("move", "?px", "?py"), ("play",)],
                ),
                Method(
                    name="SubmitWhileDead",
                    args=(),
                    preconditions=[
                        {"currentPhase": "Player_Planning"},
                        {"id": me, "ready": False, "dead": True},
                    ],
                    # 'end_pinning' just submits; a dead character has nothing to move
                    subtasks=[("end_pinning",), ("play",)],
                ),
                Method(
                    name="WaitForTeammates",
                    args=(),
                    preconditions=[
                        {"id": me},
                    ],
                    subtasks=[("play",)],
                ),
            ]
        }

        return domain
