"""Protocol view definitions for M1, M2, and M12."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolView:
    protocol: str
    m1: str
    m2: str
    m12: str


PROTOCOLS = {
    "vt": ProtocolView(protocol="vt", m1="vision", m2="text", m12="vision_text"),
    "va": ProtocolView(protocol="va", m1="vision", m2="audio", m12="vision_audio"),
    "it": ProtocolView(protocol="it", m1="image", m2="text", m12="image_text"),
}
