from pynput import keyboard

class KeyboardManager:
    def __init__(self):
        """
        Initialize the KeyboardManager, setting up listeners and key states.
        """
        self.pressed_keys = set()  # Store currently pressed keys
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )

    def _on_press(self, key):
        """
        Internal method to handle key press events.
        """
        try:
            self.pressed_keys.add(key.char)  # Add character keys
        except AttributeError:
            self.pressed_keys.add(key)  # Add special keys

    def _on_release(self, key):
        """
        Internal method to handle key release events.
        """
        try:
            self.pressed_keys.discard(key.char)  # Remove character keys
        except AttributeError:
            self.pressed_keys.discard(key)  # Remove special keys

    def is_key_pressed(self, key):
        """
        Check if a specific key is currently pressed.
        :param key: The key to check (e.g., 'd' or keyboard.Key.esc).
        :return: True if the key is pressed, False otherwise.
        """
        return key in self.pressed_keys

    def start(self):
        """
        Start the keyboard listener.
        """
        self.listener.start()

    def stop(self):
        """
        Stop the keyboard listener.
        """
        self.listener.stop()
