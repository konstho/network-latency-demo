"""
ESP32 serial communication with a proper simulation mode.

- pyserial is imported lazily so the module works even if pyserial isn't installed
- if port is None, or pyserial is missing, or the port can't be opened,
  we silently fall back to printing what would have been sent
- message format: "<x>,<y>\n" where x and y are integers in [0, 100]
"""

import time

try:
    import serial
    HAVE_SERIAL = True
except ImportError:
    serial = None
    HAVE_SERIAL = False


class ESP32Comm:
    def __init__(self, port=None, baudrate=115200, print_simulated=True):
        self.port = port
        self.baudrate = baudrate
        self.connection = None
        self.connected = False
        self.print_simulated = print_simulated
        self._last_print_t = 0.0

    def connect(self):
        if not self.port:
            print("[ESP32] no port given — simulation mode")
            return
        if not HAVE_SERIAL:
            print("[ESP32] pyserial not installed — simulation mode")
            return
        try:
            self.connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2.0)   # ESP32 resets on connect; give it time
            self.connected = True
            print(f"[ESP32] connected on {self.port} @ {self.baudrate}")
        except Exception as e:
            print(f"[ESP32] failed to open {self.port}: {e} — simulation mode")
            self.connected = False

    def send(self, x, y):
        x = max(0, min(100, int(x)))
        y = max(0, min(100, int(y)))
        if not self.connected:
            # throttle the prints to once per 100ms so the terminal stays readable
            if self.print_simulated:
                now = time.time()
                if now - self._last_print_t > 0.1:
                    print(f"[ESP32-sim] X={x:3d} Y={y:3d}")
                    self._last_print_t = now
            return
        try:
            self.connection.write(f"{x},{y}\n".encode())
        except Exception as e:
            print(f"[ESP32] send error: {e}")
            self.connected = False

    def disconnect(self):
        if self.connection:
            try:
                self.connection.close()
            except Exception:
                pass
        self.connected = False
