"""Filesystem event watcher (inotify-backed)."""

from __future__ import annotations

import ctypes
import math
import os
import select
import struct
import time
from collections.abc import Sequence
from pathlib import Path


class InotifyArtifactWatcher:
    """Block on relevant artifact changes without time-based polling."""

    _EVENT = struct.Struct("iIII")
    _MASK = 0x00000002 | 0x00000004 | 0x00000008 | 0x00000080 | 0x00000100

    def __init__(self, paths: Sequence[Path], process_pid: int | None = None) -> None:
        self.targets = {path.expanduser().resolve() for path in paths}
        for target in self.targets:
            target.parent.mkdir(parents=True, exist_ok=True)
        libc = ctypes.CDLL(None, use_errno=True)
        self._close = libc.close
        self.fd = int(libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK))
        if self.fd < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        self.watch_dirs: dict[int, Path] = {}
        for parent in sorted({path.parent for path in self.targets}):
            wd = int(libc.inotify_add_watch(self.fd, os.fsencode(parent), self._MASK))
            if wd < 0:
                errno = ctypes.get_errno()
                self.close()
                raise OSError(errno, os.strerror(errno), parent)
            self.watch_dirs[wd] = parent
        self.process_fd = -1
        if process_pid is not None and hasattr(os, "pidfd_open"):
            try:
                self.process_fd = os.pidfd_open(process_pid)
            except ProcessLookupError:
                self.process_fd = -1

    def wait(self, timeout_seconds: float | None = None) -> None:
        poller = select.poll()
        poller.register(self.fd, select.POLLIN | select.POLLERR | select.POLLHUP)
        if self.process_fd >= 0:
            poller.register(
                self.process_fd,
                select.POLLIN | select.POLLERR | select.POLLHUP,
            )
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            timeout_ms = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                timeout_ms = max(1, math.ceil(remaining * 1000))
            events = poller.poll(timeout_ms)
            if not events:
                return
            if any(fd == self.process_fd for fd, _ in events):
                return
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                continue
            offset = 0
            while offset < len(data):
                wd, _, _, name_length = self._EVENT.unpack_from(data, offset)
                offset += self._EVENT.size
                raw_name = data[offset : offset + name_length]
                offset += name_length
                name = os.fsdecode(raw_name.split(b"\0", 1)[0])
                watch_dir = self.watch_dirs.get(wd)
                if watch_dir is None:
                    continue
                if name and (watch_dir / name).resolve() in self.targets:
                    return

    def close(self) -> None:
        if getattr(self, "process_fd", -1) >= 0:
            os.close(self.process_fd)
            self.process_fd = -1
        if getattr(self, "fd", -1) >= 0:
            self._close(self.fd)
            self.fd = -1
