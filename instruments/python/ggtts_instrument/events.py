"""Read a `ggtts-trace/2` or previous v1 stream: one JSON object per line, the header first, `end` last.

`read_trace` reads a whole recorded trace. `iter_events` yields events as they arrive, from a pipe or a file being
written. Both refuse what is not a trace: a line that is not an object, an event without `ev` or `t_ms`, a first line
that is not the header, or a format this reader does not know. A stream that stops without `end` is truncated:
`Trace.complete` is False, and an instrument reports `truncated`, never a short render.
"""

import json
import sys

FORMAT = "ggtts-trace/2"
# The formats this reader accepts: the current one first, then at most the one before it.
READS = (FORMAT, "ggtts-trace/1")


class TraceError(ValueError):
    """The input is not a trace this reader accepts."""


def _parse_line(n, raw, first):
    try:
        ev = json.loads(raw)
    except ValueError as e:
        raise TraceError("line %d: not JSON: %s" % (n, e))
    if not isinstance(ev, dict):
        raise TraceError("line %d: not an object" % n)
    if not isinstance(ev.get("ev"), str) or not isinstance(ev.get("t_ms"), int) or isinstance(ev.get("t_ms"), bool):
        raise TraceError("line %d: an event needs `ev` and an integer `t_ms`" % n)
    if first:
        if ev["ev"] != "trace":
            raise TraceError("line %d: the first line is not the header" % n)
        if ev.get("format") not in READS:
            raise TraceError("not a trace this reads (%s): %r" % (", ".join(READS), ev.get("format")))
    elif ev["ev"] == "trace":
        raise TraceError("line %d: a second header" % n)
    return ev


def iter_events(source):
    """Yield the events of a trace as they arrive, the header first.

    `source` is a path, "-" for standard input, or an open text file. Stops at `end`, or when the input ends; the
    caller knows the stream was complete if the last event it saw was `end`.
    """
    if source == "-":
        f, close = sys.stdin, False
    elif hasattr(source, "readline"):
        f, close = source, False
    else:
        f, close = open(source, encoding="utf-8"), True
    try:
        n, first = 0, True
        for raw in f:
            n += 1
            if not raw.strip():
                continue
            ev = _parse_line(n, raw, first)
            first = False
            yield ev
            if ev["ev"] == "end":
                return
        if first:
            raise TraceError("empty trace")
    finally:
        if close:
            f.close()


class Trace:
    """A whole trace: its header, its events after the header, and whether it ended with `end`."""

    def __init__(self, header, events, complete):
        self.header = header
        self.events = events
        self.complete = complete

    def of(self, *kinds):
        """The events of the given kinds, in stream order."""
        return [e for e in self.events if e["ev"] in kinds]

    def draws(self):
        """Each draw as (chunk_begin, [steps], chunk_end or None), in stream order."""
        out, cur = [], None
        for e in self.events:
            if e["ev"] == "chunk_begin":
                cur = (e, [], None)
                out.append(cur)
            elif e["ev"] == "step" and cur is not None:
                cur[1].append(e)
            elif e["ev"] == "chunk_end" and cur is not None:
                out[-1] = (cur[0], cur[1], e)
                cur = None
        return out


def read_trace(source):
    """Read a whole trace. Raises `TraceError` if it is not one; a trace without `end` reads with complete False."""
    events = list(iter_events(source))
    return Trace(events[0], events[1:], events[-1]["ev"] == "end")
