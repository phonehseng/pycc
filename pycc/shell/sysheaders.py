"""Put the toolchain's system headers into ``/usr/include`` so users can read them.

The compiler never reads ``/usr/include`` from the virtual filesystem: ``#include
<stdio.h>`` is answered by ``SystemHeaders`` directly. The files
written here are a read-only *copy* of the same texts, so ``ls /usr/include`` and
``cat /usr/include/device.h`` show exactly what the compiler sees.
"""
from __future__ import annotations

import posixpath
from collections.abc import Mapping

from pycc.interp.toolchain import SYSTEM_INCLUDE_DIR, SystemHeaders
from pycc.vfs.filesystem import VirtualFS


def install_system_headers(vfs: VirtualFS,
                           headers: SystemHeaders | Mapping[str, str]) -> list[str]:
    """Write every header read-only under ``/usr/include``; returns the paths written.

    ``headers`` is the toolchain's ``SystemHeaders`` or a plain name -> text mapping
    (``SystemHeaders.as_dict()``). Calling it again (for example after the host
    added device headers) replaces the old texts. The
    folders stay read-only for the user afterwards.
    """
    if isinstance(headers, Mapping):
        headers = SystemHeaders(dict(headers))
    written: list[str] = []
    for name in headers.names():
        text = headers.get(name)
        if text is None:
            continue
        path = f"{SYSTEM_INCLUDE_DIR}/{name}"
        _write_protected(vfs, path, text)
        written.append(path)
    if vfs.is_dir(SYSTEM_INCLUDE_DIR):
        vfs.set_read_only(SYSTEM_INCLUDE_DIR)
    return written


def _write_protected(vfs: VirtualFS, path: str, text: str) -> None:
    """Write ``text`` at ``path`` and mark it read-only, even over an old read-only copy.

    Folders on the way are created and made writable just long enough to add the file:
    the read-only flag protects the headers from the user, not from the toolchain.
    """
    folder = posixpath.dirname(path)
    _ensure_folder(vfs, folder)
    vfs.set_read_only(folder, False)
    if vfs.is_file(path):
        vfs.set_read_only(path, False)
    vfs.write_file(path, text)
    vfs.set_read_only(path)
    vfs.set_read_only(folder)


def _ensure_folder(vfs: VirtualFS, folder: str) -> None:
    """Create ``folder`` and its parents, unlocking read-only parents while doing so."""
    if vfs.is_dir(folder):
        return
    parent = posixpath.dirname(folder)
    if parent != folder:
        _ensure_folder(vfs, parent)
    locked = vfs.stat(parent).read_only
    if locked:
        vfs.set_read_only(parent, False)
    vfs.mkdir(folder)
    if locked:
        vfs.set_read_only(parent)


__all__ = ["install_system_headers"]
