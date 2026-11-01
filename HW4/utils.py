from datetime import datetime
from random import shuffle


def get_timestamp_string():
    return f'[{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}]'


def augment_game_state(state: dict, player_id: str) -> list[dict]:
    """
    Entry point that augments the raw game state with additional cell/sight info.
    @param state: The full game state dictionary from the environment.
    @param player: The player id.
    @return: A flat list of scene object dictionaries.
    """
    game_data_obj = state['content']['gameData']
    game_data_obj['id'] = 'gameData'

    scene = state['content']['scene']
    scene = _add_sight_status(scene, game_data_obj, player_id)
    scene = _fill_missing_cells(scene, game_data_obj)
    scene = _convert_lists_to_strings(scene)
    scene = _check_if_shrine_reached(scene, player_id)
    shuffle(scene)
    scene = [game_data_obj] + scene
    return scene


def _add_sight_status(scene: list[dict], game_data_obj: dict, player_id: str) -> list[dict]:
    """
    Adds 'sight_status' to each object in the scene based on the player's sight range.
    Objects within the player's sight range are marked 'visible'; others 'hidden'.
    Objects with no x/y coordinates (e.g., gameData) are skipped.
    @param scene: List of scene object dictionaries.
    @param game_data_obj: The gameData dictionary containing player position and sight range.
    @return: The scene list with 'sight_status' added to each positional object.
    """
    player_obj = next((obj for obj in scene if obj.get('id') == player_id), None)
    if player_obj is None:
        return scene

    x_lower = player_obj['x'] - player_obj['sightRange']
    x_upper = player_obj['x'] + player_obj['sightRange']
    y_lower = player_obj['y'] - player_obj['sightRange']
    y_upper = player_obj['y'] + player_obj['sightRange']

    for obj in scene:
        x, y = obj.get('x'), obj.get('y')
        if x is None or y is None:
            continue
        obj['sight_status'] = 'visible' if x_lower <= x <= x_upper and y_lower <= y <= y_upper else 'hidden'

    return scene


def _fill_missing_cells(scene: list[dict], game_data_obj: dict) -> list[dict]:
    """
    Appends placeholder Cell objects for any board positions not present in the scene.
    Missing cells are marked with sight_status='unexplored'.
    @param scene: List of scene object dictionaries.
    @param game_data_obj: The gameData dictionary containing boardWidth and boardHeight.
    @return: The scene list with unexplored Cell entries appended for all missing positions.
    """
    all_cells = {(i, j) for i in range(game_data_obj['boardWidth']) for j in range(game_data_obj['boardHeight'])}
    scene_cells = {(obj.get('x'), obj.get('y'), obj.get('sight_status')) for obj in scene}
    unexplored = all_cells - scene_cells

    unexplored_cells = [
        {'id': f'C-{pos[0]}-{pos[1]}', 'entityType': 'Cell', 'objKey': '  ', 'sight_status': 'unexplored', 'x': pos[0], 'y': pos[1]}
        for i, pos in enumerate(unexplored)
    ]

    explored_cells = [
        {'id': f'C-{pos[0]}-{pos[1]}', 'entityType': 'Cell', 'objKey': '  ', 'sight_status': pos[2], 'x': pos[0], 'y': pos[1]}
        for i, pos in enumerate(scene_cells)
    ]

    cells = explored_cells + unexplored_cells
    # shuffle(cells)
    scene.extend(cells)

    return scene


def _convert_lists_to_strings(scene: list[dict]) -> list[dict]:
    """
    Converts any list values in scene object dictionaries to comma-joined strings.
    This is just a requirement of the HTN planner implementation.
    @param scene: List of scene object dictionaries.
    @return: The scene list with all list values replaced by comma-separated strings.
    """
    return [{k: ','.join(v) if isinstance(v, list) else v for k, v in obj.items()} for obj in scene]


def _check_if_shrine_reached(scene: list[dict], player_id: str) -> list[dict]:
    """
    Correction logic: if character is co-located with shrine on grid, make sure the shrine is
    set as having been reached.
    The shrine's 'character' field stores the character ID (e.g. 'C21'), so we
    compare it directly against player_id rather than mapping to a display name.
    @param scene: List of scene object dictionaries.
    @param player_id: The character ID of the active player (e.g. 'C21').
    @return: The scene list with the shrine's reached flag corrected if applicable.
    """
    player = None
    shrine = None
    for obj in scene:
        if obj.get('id') == player_id:
            player = obj
        elif obj.get('entityType') == 'Shrine' and obj.get('character') == player_id:
            shrine = obj

        if player and shrine:
            break

    # Shrine has not been seen yet, nothing to do
    if shrine is None:
        return scene

    # Shrine or player position not yet revealed (fog of war) — nothing to compare
    if shrine.get('x') is None or player is None or player.get('x') is None:
        return scene

    if shrine['x'] == player['x'] and shrine['y'] == player['y']:
        shrine['reached'] = True
    return scene

