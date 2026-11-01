from json import loads
from json.decoder import JSONDecodeError
import re
from time import sleep
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError, ConnectionClosed
from websockets.sync.client import connect
from utils import get_timestamp_string


class UnityWebSocket:
    def __init__(self, url):
        self.url = url
        self.connection = None
        self.connect()

    def connect(self):
        while True:
            try:
                self.connection = connect(self.url, open_timeout=None, close_timeout=None)
                break
            except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed):
                sleep(1)
        
    def execute_action(self, action):
        # Command to send to Game env
        action_command = {"command": "execute_action", "action": action}
        # Planning phase: no inputs
        # Pinging phase: no inputs
        resp = self.send(str(action_command))
        return self._fix_json_string_response(resp)

    def register(self, agent_id):
        # Command to send to Game env
        action_command = {"command": "register", "agent_id": agent_id}
        # Planning phase: no inputs
        # Pinging phase: no inputs
        return self.send(str(action_command))

    def get_state(self):
        """
        Requests the current fog-of-war state from the Unity game.

        @return: The parsed state dict, or None on unrecoverable errors.
        """
        import re
        state = self.send('{"command":"get_fow_state"}')
        return self._fix_json_string_response(state)

    def send(self, message, num_retries=5):
        for _ in range(num_retries):
            try:
                self.connection.send(message)
                return self.connection.recv()
            except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed):
                sleep(0.5)
                self.connect()

    @staticmethod
    def _fix_json_string_response(response: str) -> dict:
        """
        Attempts to parse a JSON response string. If parsing fails
        (typically due to a malformed "content" value from the game
        server), strips the "content" attribute and everything after
        it, then closes the JSON object.

        @param response: The raw JSON string from the server.
        @return: The parsed dict, or None on unrecoverable errors.
        """
        try:
            return loads(response)
        except JSONDecodeError:
            # Strip the malformed "content" field and everything after it
            match = re.search(r',\s*"content"\s*:', response)
            if match:
                try:
                    fixed = response[:match.start()] + '}'
                    return loads(fixed)
                except JSONDecodeError:
                    pass

            print(f'{get_timestamp_string()} Error decoding json in response: {response}')