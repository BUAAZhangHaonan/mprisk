"""Lightweight schema markers used by configs and manifests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemaInfo:
    name: str
    version: str


def parse_schema(schema: str) -> SchemaInfo:
    parts = schema.rsplit("_v", 1)
    if len(parts) != 2:
        return SchemaInfo(name=schema, version="unknown")
    return SchemaInfo(name=parts[0], version=parts[1])
