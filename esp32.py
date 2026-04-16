import serial
import time

class ESP32Comm:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.connection = None
        self.connected = False

    def connect(self):
        try:
            self.connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.connected = True
            print(f"Connected to ESP32 on {self.port}")
        except Exception as e:
            print(f"Failed to connect to ESP32: {e}")
            self.connected = False

    def send(self, x, y):
        if not self.connected:
            print("ESP32 not connected — simulating")
            print(f"Would send: X={x}, Y={y}")
            return
        try:
            message = f"{x},{y}\n"
            self.connection.write(message.encode())
        except Exception as e:
            print(f"Send error: {e}")

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connected = False
            print("Disconnected from ESP32")
