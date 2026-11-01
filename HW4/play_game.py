import sys
import psutil
from time import sleep
from threading import Thread
from traceback import format_exc
from environment.dice_adventure_env import DiceAdventureEnv, _launch_game
from agent import DiceAdventureAgent
from utils import get_timestamp_string
from environment.htn.htn import FailedPlanException
GAME_SPEED = 0.0
GAME_LAUNCH_WAIT_TIME = 0


def _run_agent(agent: DiceAdventureAgent, results: list) -> None:
    """
    Thread target that runs a single agent's game loop. Always appends
    the agent's game tracker to the shared results list, even on failure.

    @param agent: The DiceAdventureAgent to run.
    @param results: Shared list to append the game tracker to.
    @return: None
    """
    try:
        agent.env.register()
        agent.play()
    except Exception as e:
        if "GAME_OVER" in str(e):
            error_message = "Game Complete!"
        else:
            error_message = format_exc()
        print(f'{get_timestamp_string()} {error_message}')
    finally:
        results.append(agent.env.game_tracker)


def main():
    # Launch the game executable
    if sys.platform == 'win32':
        game_filepath = "DiceAdventure/Windows/DiceAdventure.exe"
    else:
        game_filepath = "DiceAdventure/MacOS/DiceAdventure.app/Contents/MacOS/DiceAdventure"
    _launch_game(game_filepath, '4649', GAME_SPEED, GAME_LAUNCH_WAIT_TIME)


    temp_env = DiceAdventureEnv(None, None)
    # Create one agent per player, each with its own env reference
    agents = []
    for name in temp_env.player_names:
        agent = DiceAdventureAgent(name, temp_env.get_player_id(name), env=DiceAdventureEnv(port='4649', player_name=name))
        agents.append(agent)

    # Shared list to collect game trackers from all threads
    game_tracker_results = []

    # Spawn a background thread per agent
    threads = []
    for agent in agents:
        t = Thread(target=_run_agent, 
                   args=(agent, game_tracker_results), 
                   daemon=True)
        t.start()
        threads.append(t)

    # Wait for all agent threads to finish
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print(f'{get_timestamp_string()} Interrupted by user.')

    # Merge all game trackers into one and calculate the final score
    if game_tracker_results:
        merged_tracker = game_tracker_results[0]
        for other_tracker in game_tracker_results[1:]:
            for level, level_data in other_tracker.levels.items():
                if level not in merged_tracker.levels:
                    merged_tracker.levels[level] = level_data
                else:
                    # Merge per-player data from other trackers into the merged one
                    for player_id in merged_tracker.player_name_id_map.values():
                        if player_id in level_data and level_data[player_id].get('health', 0) > 0:
                            merged_tracker.levels[level][player_id] = level_data[player_id]
        final_score = merged_tracker.get_score()
        print(f'{get_timestamp_string()} Homework Score: {round((100/6) * min(6, len(merged_tracker.levels)), 2)} | Leaderboard Score: {round(final_score, 2)}')
    
    # Kill game process
    PROCNAME = "DiceAdventure"
    for proc in psutil.process_iter():
        if PROCNAME in proc.name():
            proc.kill()



if __name__ == "__main__":
    main()

