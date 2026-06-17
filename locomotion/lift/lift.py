from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import serial
import serial.tools.list_ports


DEFAULT_BAUD = 9600
DEFAULT_MIN_POSITION_MM = 0
DEFAULT_MAX_POSITION_MM = 300
DEFAULT_PORT_CANDIDATES = (
    "/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_EKDMo151406-if00-port0",
    "/dev/ttyUSB1",
)


@dataclass(frozen=True)
class LiftPacket:
    position_mm: int
    raw: bytes


@dataclass(frozen=True)
class LiftConfig:
    port: str | None = None
    baud: int = DEFAULT_BAUD
    min_position_mm: int = DEFAULT_MIN_POSITION_MM
    max_position_mm: int = DEFAULT_MAX_POSITION_MM
    serial_timeout_s: float = 0.1
    feedback_timeout_s: float = 1.0


class Lift:
    """TYC serial-protocol lift controller.

    Protocol:
      - Feedback: 8E 8E <hundreds> <tens_ones> FF
      - Command:  CE CE <cmd> <hundreds> <tens_ones> FF
      - Commands: 0 stop, 1 up limit, 2 down limit, 3 goto position.
    """

    def __init__(self, config: LiftConfig | None = None, **kwargs) -> None:
        if config is None:
            config = LiftConfig()
        self.config = replace(config, **kwargs)
        self.port = self.config.port or choose_default_port()
        if not self.port:
            raise RuntimeError(
                "No lift serial port found. Run `bash locomotion/scripts/lift.sh list` "
                "or pass --port explicitly."
            )

    @staticmethod
    def list_ports() -> list[tuple[str, str, str]]:
        return [
            (port.device, port.description, port.hwid)
            for port in serial.tools.list_ports.comports()
        ]

    def read_status(self, timeout_s: float | None = None) -> list[LiftPacket]:
        deadline = time.monotonic() + (
            self.config.feedback_timeout_s if timeout_s is None else float(timeout_s)
        )
        buffer = bytearray()
        packets: list[LiftPacket] = []
        with self._open_serial() as ser:
            while time.monotonic() < deadline:
                data = ser.read(max(1, ser.in_waiting))
                if data:
                    buffer.extend(data)
                    packets.extend(parse_position_packets(buffer))
                else:
                    time.sleep(0.02)
        return packets

    def read_first_position(self, timeout_s: float | None = None) -> LiftPacket | None:
        packets = self.read_status(timeout_s=timeout_s)
        return packets[-1] if packets else None

    def sniff_raw(self, timeout_s: float) -> list[bytes]:
        deadline = time.monotonic() + float(timeout_s)
        chunks: list[bytes] = []
        with self._open_serial() as ser:
            while time.monotonic() < deadline:
                data = ser.read(max(1, ser.in_waiting))
                if data:
                    chunks.append(data)
                else:
                    time.sleep(0.02)
        return chunks

    def stop(self) -> bytes:
        packet = command_packet(
            0,
            min_position_mm=self.config.min_position_mm,
            max_position_mm=self.config.max_position_mm,
        )
        self._send(packet)
        return packet

    def up(self, check_feedback: bool = True) -> bytes:
        if check_feedback:
            self._require_feedback()
        packet = command_packet(
            1,
            min_position_mm=self.config.min_position_mm,
            max_position_mm=self.config.max_position_mm,
        )
        self._send(packet)
        return packet

    def down(self, check_feedback: bool = True) -> bytes:
        if check_feedback:
            self._require_feedback()
        packet = command_packet(
            2,
            min_position_mm=self.config.min_position_mm,
            max_position_mm=self.config.max_position_mm,
        )
        self._send(packet)
        return packet

    def goto(self, position_mm: int, check_feedback: bool = True) -> bytes:
        if check_feedback:
            self._require_feedback()
        packet = command_packet(
            3,
            position_mm=position_mm,
            min_position_mm=self.config.min_position_mm,
            max_position_mm=self.config.max_position_mm,
        )
        self._send(packet)
        return packet

    def _open_serial(self) -> serial.Serial:
        return serial.Serial(
            port=self.port,
            baudrate=self.config.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.config.serial_timeout_s,
        )

    def _send(self, packet: bytes) -> None:
        with self._open_serial() as ser:
            ser.write(packet)
            ser.flush()

    def _require_feedback(self) -> None:
        packet = self.read_first_position(timeout_s=self.config.feedback_timeout_s)
        if packet is None:
            raise RuntimeError(
                f"No lift position feedback received from {self.port}; refusing motion."
            )


def choose_default_port() -> str | None:
    for port in DEFAULT_PORT_CANDIDATES:
        if Path(port).exists():
            return port
    return None


def command_packet(
    command: int,
    position_mm: int = 0,
    min_position_mm: int = DEFAULT_MIN_POSITION_MM,
    max_position_mm: int = DEFAULT_MAX_POSITION_MM,
) -> bytes:
    if not 0 <= command <= 3:
        raise ValueError("command must be in range 0..3")
    if not min_position_mm <= position_mm <= max_position_mm:
        raise ValueError(
            f"position must be in range {min_position_mm}..{max_position_mm} mm"
        )
    return bytes(
        [
            0xCE,
            0xCE,
            command,
            position_mm // 100,
            position_mm % 100,
            0xFF,
        ]
    )


def parse_position_packets(buffer: bytearray) -> list[LiftPacket]:
    packets: list[LiftPacket] = []
    i = 0
    while i <= len(buffer) - 5:
        if buffer[i] == 0x8E and buffer[i + 1] == 0x8E and buffer[i + 4] == 0xFF:
            raw = bytes(buffer[i : i + 5])
            hundreds = raw[2]
            tens_ones = raw[3]
            if hundreds <= 9 and tens_ones <= 99:
                packets.append(LiftPacket(hundreds * 100 + tens_ones, raw))
            i += 5
            continue
        i += 1
    if i:
        del buffer[:i]
    return packets


def format_packet(packet: bytes) -> str:
    return " ".join(f"{b:02X}" for b in packet)
