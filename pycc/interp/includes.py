"""Finding the file behind an ``#include``, the way gcc does.

The preprocessor asks an ``IncludeResolver`` for every ``#include``; this module is the
one the compile driver uses. It follows gcc's search order:

* ``#include "x.h"``: the folder of the file that contains the ``#include``, then each
  ``-I`` folder in the order given, then the system headers.
* ``#include <x.h>``: the system headers first, then each ``-I`` folder.

System headers never come from the user's filesystem: they are served by the
toolchain's ``SystemHeaders`` and shown as ``/usr/include/<name>``.
Project files are read through ``SourceFiles`` relative to the ``cc`` working
directory, but the path we hand back - and that every diagnostic shows - is the path
as the user would write it: ``"util.h"``, ``"include/parts.h"``.
"""
from __future__ import annotations

import posixpath
from collections.abc import Iterable

from pycc.interp.hostio import SourceFiles, join_path
from pycc.interp.toolchain import SYSTEM_INCLUDE_DIR, SystemHeaders


class IncludeResolverImpl:
    """An ``interp.preprocessor.IncludeResolver`` over project files and system headers."""

    def __init__(self, files: SourceFiles, cwd: str, include_dirs: Iterable[str],
                 system_headers: SystemHeaders) -> None:
        self.files = files
        self.cwd = cwd or "/"
        self.include_dirs = [directory for directory in include_dirs if directory]
        self.system_headers = system_headers

    def resolve(self, name: str, quoted: bool, including_file: str) -> tuple[str, str] | None:
        """Return ``(path to show, text)`` for ``#include "name"`` / ``<name>``, or None."""
        if name.startswith("/"):
            return self._lookup_project(name)   # an absolute path is shown as written
        if quoted:
            return (self._lookup_project(posixpath.join(posixpath.dirname(including_file), name))
                    or self._search_include_dirs(name)
                    or self._lookup_system(name))
        return self._lookup_system(name) or self._search_include_dirs(name)

    # --- the three places a header can live ---------------------------------------------
    def _lookup_system(self, name: str) -> tuple[str, str] | None:
        """A simulated system header such as ``stdio.h`` or ``device.h``."""
        text = self.system_headers.get(posixpath.normpath(name))
        if text is None:
            return None
        return self.system_headers.path_of(posixpath.normpath(name)), text

    def _search_include_dirs(self, name: str) -> tuple[str, str] | None:
        """Try each ``-I`` folder in the order it was given on the command line."""
        for directory in self.include_dirs:
            found = self._lookup_project(posixpath.join(directory, name))
            if found is not None:
                return found
        return None

    def _lookup_project(self, shown_path: str) -> tuple[str, str] | None:
        """A file of the user's project, named relative to the cc working directory."""
        shown = posixpath.normpath(shown_path)
        if shown.startswith(SYSTEM_INCLUDE_DIR + "/"):
            # A system header including a sibling with quotes ("machine.h" from
            # /usr/include/device.h): the compiler never reads /usr/include from the
            # user's files, so answer from the system headers instead.
            return self._lookup_system(shown[len(SYSTEM_INCLUDE_DIR) + 1:])
        text = self._read(join_path(self.cwd, shown))
        return (shown, text) if text is not None else None

    def _read(self, full_path: str) -> str | None:
        """Read a project file, treating any failure as "not there"."""
        if not self.files.is_file(full_path):
            return None
        try:
            return self.files.read_text(full_path)
        except Exception:  # noqa: BLE001 - each SourceFiles implementation raises its own errors
            return None
