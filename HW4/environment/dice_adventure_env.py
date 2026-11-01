from collections import Counter
from json import loads
import subprocess
import copy
import json
import random
from time import sleep
from threading import Thread
from typing import Any, Tuple, Union, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from agent import DiceAdventureAgent
from environment.htn.htn_elements.structure import Operator
from environment.move_planner import MovePlanner
from utils import augment_game_state
from environment.unity_socket import UnityWebSocket
from utils import get_timestamp_string


class DiceAdventureEnv:
    """
    Implements a custom gym environment for the Dice Adventure Unity game.
    """
    def __init__(self, 
                 port: str, 
                 player_name: str,
                 game_speed: float = 1.0, 
                 game_launch_wait_time: float = 5,
                 universal_sleep_time: float = 0.5):
        """
        Init function for Dice Adventure gym environment.
        :param player:      The player that will be used to train the agent. Defaults to the Dwarf.
        :param port:        The custom port that will be used to connect the agent to the game instance.
                            You should choose a port that is currently not in use by your system.
        """
        # Game setup
        self.port = port
        self.player_name = player_name
        self.socket_url = "ws://localhost:{}/hmt/{}"
        self.sockets = {'Dwarf': None, 'Giant': None, 'Human': None}
        self.player_names = ["Dwarf", "Giant", "Human"]
        self.game_actions = ['pinga', 'pingb', 'pingc', 'pingd', 'undo', 'wait', 'submit', 'up', 'down', 'left', 'right', 'move', 'random']
        # Game Tracker
        self.game_tracker = GameTracker() # Tracks gameplay and scoring
        # Game play
        self.move_planner = MovePlanner() # An A* search planner used to execute move actions for the agent's character
        # Options
        self.universal_sleep_time = universal_sleep_time
        # Level change tracking
        self._current_level = None

    def execute_action(self, operator_execution: tuple,  player_name: str = None) -> None:
        """
        Executes the given action for the given player. For move actions,
        A* is called with the action point cap so it never generates a path
        longer than the player can execute. All moves are then sent to the
        game followed by an auto-submit. For random actions, any directional
        choice that would take the character out of board bounds is replaced
        with 'wait' before being sent to the game server.
        :param player_name:         The player the action is being applied to
        :param operator_execution:  The action to take in the form of an HTN operator execution (operator name + args)
        :return:                    The resulting state after taking the given action
        """
        player_name = player_name if player_name is not None else self.player_name

        action = operator_execution[0]
        args = operator_execution[1:]

        if action == 'move':

            scene = self.get_state(player_name)

            player_obj = find_player(scene, player_id=self.game_tracker.player_name_id_map[player_name])
            assert player_obj is not None
            src = player_obj.get('x', None), player_obj.get('y', None)

            # src = get_player_position_cursor(*args[:2], action_plan=player_obj['actionPlan'])
            dest = tuple(args[:2])
            max_actions = int(player_obj.get('actionPoints', 0))
            current_level = scene[0]['currLevel']
            visited = self.game_tracker.get_visited(current_level, player_name)

            # A* plans a path capped at max_actions steps
            full_path = self.move_planner.get_next_move(
                src=src, dest=dest,
                scene=scene[1:], game_data=scene[0],
                visited=visited,
                max_steps=max_actions
            )
            planned_actions = list(full_path) if full_path else []

            # Execute each planned action, then auto-submit
            for game_action in planned_actions:
                resp = self._get_socket(player_name).execute_action(game_action)
                sleep(self.universal_sleep_time)

            # Submit the plan
            game_action = 'submit'
            # resp = self._get_socket(player_name).execute_action('submit')
            # sleep(self.universal_sleep_time)
            # return resp['status'] == 'OK', resp['message']

        elif action == 'pin':

            pin_type = args[0]

            if pin_type not in {'a', 'b', 'c', 'd'}:
                raise Exception("Ping type does not exist, only pinga, pingb, pingc, and pingd are available.")

            scene = self.get_state(player_name)
            player_obj = find_player(scene, player_id=self.game_tracker.player_name_id_map[player_name])
            assert player_obj is not None
            src = player_obj.get('x', None), player_obj.get('y', None)
            # src = get_player_position_cursor(*args[:2], action_plan=player_obj['actionPlan'])
            dest = tuple(args[1:3])
            # max_actions = int(player_obj.get('actionPoints', 0))
            max_actions = 999
            current_level = scene[0]['currLevel']
            # visited = self.game_tracker.get_visited(current_level, player_name)

            # A* plans a path capped at max_actions steps
            full_path = self.move_planner.get_next_move(
                src=src, dest=dest,
                scene=scene[1:], game_data=scene[0],
                # visited=visited,
                max_steps=max_actions
            )
            planned_actions = list(full_path) if full_path else []

            # Execute each planned action, then auto-submit
            for game_action in planned_actions:
                resp = self._get_socket(player_name).execute_action(game_action)
                sleep(self.universal_sleep_time)

            resp = self._get_socket(player_name).execute_action('ping' + pin_type)
            sleep(self.universal_sleep_time)

            # Submit the plan
            # game_action = 'submit'
            game_action = None
            # resp = self._get_socket(player_name).execute_action('submit')
            # sleep(self.universal_sleep_time)
            # return resp['status'] == 'OK', resp['message']

        elif action == 'end_pinning':
            game_action = 'submit'

        else:
            raise Exception("Invalid action", game_action)
            # game_action = action

        if game_action:
            resp = self._get_socket(player_name).execute_action(game_action)
            sleep(self.universal_sleep_time)
        # self._wait_for_round_to_complete()

        # print(f'Response: {resp} for character: ({player_name})')
        return resp['status'] == 'OK', resp['message']

    def get_env_actions(self) -> list:
        """
        Returns a list of Operator objects representing all environment
        actions available to the given player.

        @return: A list of Operator objects for environment actions.
        """
        # Mapping of action name -> argument variable names
        action_args = {
            'move':   ('?dest_x', '?dest_y'),
            'pin':   ('?pin_type', '?dest_x', '?dest_y'), # pin type is 'a', 'b', 'c', or 'd'
            'end_pinning': (),
            # 'random': (),
            # 'submit': (),
            # 'wait':   (),
            # 'undo':   (),
            # 'pinga':  ('?px', '?py'),
            # 'pingb':  ('?px', '?py'),
            # 'pingc':  ('?px', '?py'),
            # 'pingd':  ('?px', '?py'),
        }
        operators = []
        for action_name, args in action_args.items():
            operators.append(Operator(
                name=action_name,
                args=args,
                preconditions=[],
                effects=lambda b: {},
                is_env_action=True
            ))
        return operators
    
    def get_player_id(self, player_name: str = None) -> str:
        return self.game_tracker.player_name_id_map[player_name if player_name is not None else self.player_name]
    
    def get_score(self):
        return self.game_tracker.get_score()

    def get_state(self, player_name: str = None):
        """
        Gets the current state of the game.

        The state is always given from the perspective of the given player, which defines how much of the environment (level)
        the agent can currently "see". The following describes how the Fog-Of-War mechanic limits the environment
        view for agents.

            In the Unity game, you can see a visibility mask for each character.
            Black positions have not been observed. Gray positions have been observed but are not
            currently in the player's view. This function returns all objects in the current sight range (view) of
            the player plus objects in positions that the player has seen before. Note that any object that can
            move (such as monsters and other players) are only returned when they are in the player's current
            view (i.e. not obscured by black or gray squares), but static objects such as walls, stones, and traps,
            and shrines are returned if they've been previously observed.
        """
        player_name = player_name if player_name is not None else self.player_name
        while True:
            raw_state = self._get_socket(player_name).get_state()
            if ('content' not in raw_state or 'gameData' not in
                raw_state['content'] or 'currentPhase' not in
                raw_state['content']['gameData']):
                break
            if raw_state['content']['gameData']['currentPhase'] in ['Player_Pinning', 'Player_Planning']:
                break
            sleep(self.universal_sleep_time)
        
        # Detect game-over before attempting to augment
        if raw_state is None or raw_state.get('status') == 'GAME_OVER':
            raise GameOverException("Game Over")
        
        state = augment_game_state(raw_state, self.game_tracker.player_name_id_map[player_name])
        sleep(self.universal_sleep_time)

        # Update the game tracker for scoring
        self.game_tracker.update(state, self.player_name)

        if self.game_tracker.exceeded_max_rounds():
            raise Exception("Max Rounds Exceeded")

        # Detect level transitions and signal the agent to reset its task queue
        new_level = state[0].get('currLevel')
        if self._current_level is not None and new_level != self._current_level:
            self._current_level = new_level
            raise LevelChangedException(f"Level changed to {new_level}")
        self._current_level = new_level

        # print(f'***State for character: ({player_name}): \n{state}')

        return state

    def register(self, player_name: str  = None):
        """ Registers the agent with the game server. Retries on connection
        refused errors (Windows: ConnectionRefusedError, Mac: OSError) until
        the connection succeeds.
        @param player_name: The player name to register. Defaults to self.player_name.
        @return: None
        """
        player_name = player_name if player_name is not None else self.player_name
        print(f"{get_timestamp_string()} Registering agent for character: {player_name}.")
        # Retry until the game server accepts the connection
        while True:
            try:
                self._get_socket(player_name).register(player_name)
                break
            except (ConnectionRefusedError, OSError) as e:
                sleep(self.universal_sleep_time * 5)
        print(f"{get_timestamp_string()} Agent for character ({player_name}) registered.")
        sleep(self.universal_sleep_time)

    def _get_socket(self, player_name):
        if self.sockets[player_name] is None:
            self.sockets[player_name] = UnityWebSocket(self.socket_url.format(self.port, player_name.lower()))
        return self.sockets[player_name]
    
    def _wait_for_round_to_complete(self):
        """
        Waits for the current round to complete (or level to advance)
        before allowing agents to continue playing.

        The method first spins until the game returns to a player-actionable
        phase (Pinning or Planning).  It then checks whether the player is
        standing on the tower tile — but only if the game just left a
        resolution phase, since a level transition cannot occur during
        pinning.  A timeout prevents infinite hangs.

        @param state: The current augmented game state (list of fact dicts).
        @param player_name: The human-readable player name (e.g. 'Giant').
        @return: None
        """
        state = self.get_state()

        initial_phase = state[0]['currentPhase']
        was_resolving = initial_phase not in ['Player_Pinning', 'Player_Planning']

        # Wait for the game to return to a player-actionable phase
        while state[0]['currentPhase'] not in ['Player_Pinning', 'Player_Planning']:
            sleep(self.universal_sleep_time*5)
            state = self.get_state()

        # Only check for a level transition if the game was previously in a
        # resolution phase (e.g. Monster_Movement, Combat).  During pinning
        # the player may be standing on the tower but no level change can occur.
        if not was_resolving:
            return

        tower = None
        for obj in state:
            if obj.get('entityType') == 'Goal':
                tower = obj
        if tower is None:
            return

        current_level = state[0]['currLevel']
        player_id = self.game_tracker.player_name_id_map[self.player_name]
        max_wait = 30  # seconds
        elapsed = 0
        for obj in state:
            if obj.get('id') == player_id:
                if obj['x'] == tower['x'] and obj['y'] == tower['y']:
                    while current_level == self.get_state()[0]['currLevel']:
                        sleep(self.universal_sleep_time*5)
                        elapsed += 0.5
                        if elapsed >= max_wait:
                            print(f"[WARN] Timed out waiting for level change for {self.player_name}.")
                            return
                break


class GameOverException(Exception):
    """Raised when the game server reports GAME_OVER status."""
    pass


class LevelChangedException(Exception):
    """Raised when a level transition is detected in get_state().
    Signals the agent to reset its HTN task queue and re-plan from
    the beginning for the new level.
    """
    pass



def _launch_game(game_executable_filepath, port, game_speed, game_launch_wait_time):
    wait_time = 5
    print(f'{get_timestamp_string()} Launching game...')
    game_thread = Thread(target=_launch_game_thread, args=(game_executable_filepath, port, game_speed))
    game_thread.start()

    sleep(game_launch_wait_time)
    print(f'{get_timestamp_string()} Game launched.')


def _launch_game_thread(game_executable_filepath, port, game_speed):
    """
    Launches the game executable in a new process with a visible window.
    @param game_executable_filepath: `str` Path to the game executable
    @param port: `str` The port the game should listen on
    @return: `None`
    """
    command = [game_executable_filepath,
                "-localMode",
                "-hmtsocketurl", "ws://localhost",
                "-hmtsocketport", "{}".format(port),
                "-stepTime", "{}".format(game_speed),
                "-logFile", "DiceAdventure/playerlog.log"]
    import sys
    if sys.platform == 'win32':
        subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(command)

class GameTracker:
    def __init__(self):
        self.levels = {}
        self.player_name_id_map = {'Dwarf': 'C11', 'Giant': 'C21', 'Human': 'C31'}
        # For leaderboard scoring
        self.decay_factor = 0.95
        self.scoring_bonus = 10
        # Constraints
        self.max_rounds = 250 
        # Visited position tracking (per character per level)
        self.visited = {}          # (level, player_name) -> Counter of (x, y) -> visit count
        self.pending_plan = {}     # player_name -> {"start_pos": (x,y), "actions": [...]}
        self.last_round = {}       # player_name -> last observed round number

    def exceeded_max_rounds(self):
        # print('ROUNDS', int(sum([self.levels[level]['num_rounds'] for level in self.levels])))
        return int(sum([self.levels[level]['num_rounds'] for level in self.levels])) > self.max_rounds
        
    def get_score(self) -> float:
        score = 0
        for level in self.levels:
            level_score = 100 * pow(self.decay_factor, int(self.levels[level]["num_rounds"]))
            for player_id in self.player_name_id_map.values():
                try:
                    level_score += (self.scoring_bonus * (int(self.levels[level][player_id]["health"])
                                                / int(self.levels[level][player_id]["lives"])) \
                                                * (not self.levels[level][player_id]["lives"] < 3))
                except ZeroDivisionError:
                    pass 
            score += level_score
        return score
    
    def update(self, state: list[dict], player_name: str) -> None:
        """ Updates the game tracker with the latest state for the given player.
        Handles level tracking, scoring data, and visited-position reconciliation.
        The pending action plan is stored on every call. Reconciliation of visited
        positions only runs when the round has increased during the planning phase.
        @param state: The augmented game state (list of fact dicts).
        @param player_name: The human-readable player name (e.g. 'Dwarf').
        @return: None
        """
        game_data = self._get_game_data(state)
        current_level = game_data['currLevel']
        current_round = game_data['currRound']
        current_phase = game_data['currentPhase']
        
        if current_level not in self.levels:
            self.levels[current_level] = self._setup_level_tracker()
        
        self.levels[current_level]['num_rounds'] = current_round

        player_id = self.player_name_id_map[player_name]
        player_data = self._get_player_data(state, player_id)
        if player_data is None:
            return
        self.levels[current_level][player_id]['health'] = player_data['health']
        self.levels[current_level][player_id]['lives'] = player_data['lives']

        # Reconcile visited positions when round has increased during planning phase
        prev_round = self.last_round.get(player_name, 0)
        if current_phase == 'Player_Planning' and current_round > prev_round:
            if player_name in self.pending_plan and self.pending_plan[player_name] is not None:
                actual_pos = (player_data['x'], player_data['y'])
                self._reconcile_visited(current_level, player_name, actual_pos)
            # Only update last_round in Planning so the Pinning→Planning transition
            # is still detectable on the next round, ensuring reconciliation fires.
            self.last_round[player_name] = current_round

        # Always store the current pending plan (captures the growing action plan)
        action_plan_raw = player_data.get('actionPlan', '')
        if isinstance(action_plan_raw, str):
            actions = [a.strip() for a in action_plan_raw.split(',') if a.strip()] if action_plan_raw else []
        else:
            actions = list(action_plan_raw)
        self.pending_plan[player_name] = {
            "start_pos": (player_data['x'], player_data['y']),
            "actions": actions
        }

    def _reconcile_visited(self, level, player_name: str, actual_pos: tuple) -> None:
        """ Walks the stored action plan from the start position and marks
        each intermediate cell as visited, stopping when the walked position
        matches the agent's actual observed position or the plan is exhausted.
        @param level: The current level identifier.
        @param player_name: The player name key for the pending plan.
        @param actual_pos: Tuple (x, y) of the player's actual current position.
        @return: None
        """
        plan = self.pending_plan[player_name]
        x, y = plan["start_pos"]
        key = (level, player_name)
        if key not in self.visited:
            self.visited[key] = Counter()

        # Mark the starting position
        self.visited[key][(x, y)] += 1

        # Walk each action and mark intermediate positions
        for action in plan["actions"]:
            action = action.lower()
            if action == "up":
                y += 1
            elif action == "down":
                y -= 1
            elif action == "left":
                x -= 1
            elif action == "right":
                x += 1
            elif action == "wait":
                pass

            self.visited[key][(x, y)] += 1

            # Stop if we've reached the agent's actual position
            if (x, y) == actual_pos:
                break

        self.pending_plan[player_name] = None

    def get_visited(self, level, player_name: str) -> Counter:
        """ Returns the visited position Counter for a given level and player.
        @param level: The level identifier.
        @param player_name: The player name.
        @return: Counter mapping (x, y) -> visit count, or empty Counter if none.
        """
        return self.visited.get((level, player_name), Counter())

    @staticmethod
    def _setup_level_tracker() -> dict:
        return {'num_rounds': 0, 'C11': {'health': 0, 'lives': 0}, 'C21': {'health': 0, 'lives': 0}, 'C31': {'health': 0, 'lives': 0}}

    @staticmethod
    def _get_game_data(state: list[dict]) -> dict:
        for obj in state:
            if obj['id'] == 'gameData':
                return obj

    @staticmethod
    def _get_player_data(state: list[dict], player_id: str) -> dict:
        for obj in state:
            if obj['id'] == player_id:
                return obj
    

#############
# UTILITIES #
#############

def get_player_position_cursor(x, y, action_plan):
    if isinstance(action_plan, str):
        action_plan = [a.strip() for a in action_plan.split(',')]
    for action in action_plan:
        action = action.lower()
        if action == "up":
            y += 1
        elif action == "down":
            y -= 1
        elif action == "left":
            x -= 1
        elif action == "right":
            x += 1
        elif action == "wait":
            pass
    return x, y


def find_player(scene, player_id):
    for obj in scene:
        if obj.get('id') == player_id:
            return obj

