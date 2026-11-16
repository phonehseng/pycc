"""Device functions: C functions whose calls are answered by the embedding host.

A program can call functions that are not written in C or in the standard library,
for example to drive a simulated robot, an LED panel or a test double. Each such call
pauses the interpreter with a ``DeviceCall`` event; the host performs it and sends a
reply back, and the program carries on with the returned value.

``register_device_function`` adds one such function to a ``BuiltinRegistry``.
On the host side, ``pycc.shell.headless`` answers ``DeviceCall`` events with any object
that has a ``perform(name, args)`` method (see ``Device`` below); without one, the call
traps with ``invalid_argument``.
"""
from __future__ import annotations

from typing import Any, Protocol

from pycc.core.concepts import Concept
from pycc.core.events import DeviceCall
from pycc.interp.builtins import BuiltinRegistry, BuiltinSpec


class Device(Protocol):
    """What the headless runner needs from a device. Only ``perform`` is required.

    ``perform`` returns an object with a ``value`` attribute (or raises an exception
    with ``code`` and ``to_diagnostic()`` to trap the program). Optional attributes:
    ``tick`` (read by ``time()``), ``warnings`` (a list of ``Diagnostic`` the device
    appends to) and ``log_lines``.
    """

    def perform(self, name: str, args: list[Any]) -> Any: ...


def reply_value(machine: Any, reply: Any) -> Any:
    """Turn the host's reply into the C return value, showing warnings and traps."""
    for warning in reply.warnings:
        machine.warn(warning.code, warning.message, warning.friendly, warning.concept)
    if reply.trap is not None:
        machine.trap(reply.trap.code, reply.trap.message, reply.trap.friendly,
                     reply.trap.concept)
    return machine.make_int(int(reply.value or 0))


def register_device_function(registry: BuiltinRegistry, name: str, prototype: str,
                             header: str = "device.h", doc: str = "") -> BuiltinSpec:
    """Register ``prototype`` as a device function whose calls go to the host.

    Integer and pointer arguments are passed to the device as plain integers; the
    reply's ``value`` becomes the C return value (an ``int``).
    """
    def impl(machine: Any, args: list) -> Any:
        reply = yield DeviceCall(name, [int(arg.raw) for arg in args])
        return reply_value(machine, reply)

    impl.__name__ = f"device_{name}"
    return registry.register(name, prototype, impl, concept=Concept.DEVICE_API,
                             header=header, doc=doc)


__all__ = ["Device", "register_device_function", "reply_value"]
