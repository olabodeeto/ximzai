"""Calendar and Reminders via EventKit — read locally from the Mac's own
Calendar/Reminders databases (whatever accounts you've added there)."""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, time, timedelta
from typing import Any

from ximz.config import ToolsConfig
from ximz.tools import Tool

log = logging.getLogger(__name__)

_FULL_ACCESS = 3  # EKAuthorizationStatusFullAccess (== legacy "Authorized")
_MAX_ITEMS = 15
_FETCH_TIMEOUT_S = 10


def _nsdate(dt: datetime):  # type: ignore[no-untyped-def]
    from Foundation import NSDate

    return NSDate.dateWithTimeIntervalSince1970_(dt.timestamp())


def _pydate(nsdate) -> datetime:  # type: ignore[no-untyped-def]
    return datetime.fromtimestamp(nsdate.timeIntervalSince1970())


def spoken_time(dt: datetime) -> str:
    return dt.strftime("%-I:%M %p").replace(":00 ", " ")


def parse_day(day: str, today: date) -> date:
    if day == "today":
        return today
    if day == "tomorrow":
        return today + timedelta(days=1)
    return date.fromisoformat(day)


def parse_due(due: str, now: datetime) -> datetime:
    dt = datetime.fromisoformat(due.strip().replace(" ", "T"))
    if dt < now - timedelta(minutes=1):
        raise ValueError(f"{due} is in the past; it is now {now:%Y-%m-%d %H:%M}")
    return dt


class EventKitStore:
    """Thin wrapper around EKEventStore with access requests and timeouts."""

    def __init__(self, cfg: ToolsConfig) -> None:
        from EventKit import EKEventStore

        self._cfg = cfg
        self._store = EKEventStore.alloc().init()

    def ensure_access(self, entity: int) -> bool:
        from EventKit import EKEntityTypeEvent, EKEventStore

        if EKEventStore.authorizationStatusForEntityType_(entity) == _FULL_ACCESS:
            return True
        done, granted = threading.Event(), [False]

        def handler(ok: bool, error: Any) -> None:
            granted[0] = bool(ok)
            done.set()

        if entity == EKEntityTypeEvent:
            self._store.requestFullAccessToEventsWithCompletion_(handler)
        else:
            self._store.requestFullAccessToRemindersWithCompletion_(handler)
        done.wait(self._cfg.permission_timeout_s)
        if granted[0]:
            self._store.reset()
        return granted[0]

    # -- calendar ---------------------------------------------------------

    def events(self, day: str, days: int) -> str:
        from EventKit import EKEntityTypeEvent

        if not self.ensure_access(EKEntityTypeEvent):
            return ("I don't have permission to read the calendar. The user can allow it in System "
                    "Settings, Privacy and Security, Calendars.")
        start_day = parse_day(day, date.today())
        start = datetime.combine(start_day, time.min)
        end = start + timedelta(days=days)
        pred = self._store.predicateForEventsWithStartDate_endDate_calendars_(_nsdate(start), _nsdate(end), None)
        events = sorted(self._store.eventsMatchingPredicate_(pred) or [], key=lambda e: e.startDate().timeIntervalSince1970())
        span = "today" if day == "today" and days == 1 else f"from {start_day:%A %d %B} for {days} day(s)"
        if not events:
            return f"No calendar events {span}."
        lines = []
        for ev in events[:_MAX_ITEMS]:
            s = _pydate(ev.startDate())
            when = "all day" if ev.isAllDay() else f"{spoken_time(s)} to {spoken_time(_pydate(ev.endDate()))}"
            prefix = f"{s:%A} " if days > 1 else ""
            place = f" at {ev.location()}" if ev.location() else ""
            lines.append(f"{prefix}{when}: {ev.title()}{place}")
        more = f" (and {len(events) - _MAX_ITEMS} more)" if len(events) > _MAX_ITEMS else ""
        return f"Events {span}{more}:\n" + "\n".join(lines)

    # -- reminders --------------------------------------------------------

    def reminders(self) -> str:
        from EventKit import EKEntityTypeReminder
        from Foundation import NSCalendar

        if not self.ensure_access(EKEntityTypeReminder):
            return ("I don't have permission to read reminders. The user can allow it in System "
                    "Settings, Privacy and Security, Reminders.")
        pred = self._store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(None, None, None)
        done, found = threading.Event(), [[]]

        def handler(items: Any) -> None:
            found[0] = list(items or [])
            done.set()

        self._store.fetchRemindersMatchingPredicate_completion_(pred, handler)
        if not done.wait(_FETCH_TIMEOUT_S):
            return "Reminders took too long to load."
        cal = NSCalendar.currentCalendar()

        def due(r: Any) -> datetime | None:
            comps = r.dueDateComponents()
            d = cal.dateFromComponents_(comps) if comps else None
            return _pydate(d) if d else None

        items = sorted(found[0], key=lambda r: (due(r) is None, due(r) or datetime.max))
        if not items:
            return "No open reminders."
        lines = []
        for r in items[:_MAX_ITEMS]:
            d = due(r)
            lines.append(f"{r.title()}" + (f" (due {d:%A %d %B}, {spoken_time(d)})" if d else ""))
        return f"{len(items)} open reminder(s):\n" + "\n".join(lines)

    def add_reminder(self, title: str, due: str | None) -> str:
        from EventKit import EKAlarm, EKEntityTypeReminder, EKReminder
        from Foundation import (NSCalendar, NSCalendarUnitDay, NSCalendarUnitHour, NSCalendarUnitMinute,
                                NSCalendarUnitMonth, NSCalendarUnitYear)

        if not self.ensure_access(EKEntityTypeReminder):
            return "I don't have permission to create reminders."
        reminder = EKReminder.reminderWithEventStore_(self._store)
        reminder.setTitle_(title.strip())
        reminder.setCalendar_(self._store.defaultCalendarForNewReminders())
        when = ""
        if due:
            dt = parse_due(due, datetime.now())
            units = NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay | NSCalendarUnitHour | NSCalendarUnitMinute
            reminder.setDueDateComponents_(NSCalendar.currentCalendar().components_fromDate_(units, _nsdate(dt)))
            reminder.addAlarm_(EKAlarm.alarmWithAbsoluteDate_(_nsdate(dt)))
            when = f" for {dt:%A %d %B} at {spoken_time(dt)}"
        ok, error = self._store.saveReminder_commit_error_(reminder, True, None)
        if not ok:
            return f"Error: couldn't save the reminder: {error}"
        return f"Reminder set: {title.strip()}{when}."


def calendar_tools(cfg: ToolsConfig, store: EventKitStore | None = None) -> list[Tool]:
    ek = store or EventKitStore(cfg)
    return [
        Tool(
            name="calendar_events",
            description="List the user's calendar events for a day (and optionally following days).",
            parameters={
                "type": "object",
                "properties": {
                    "day": {"type": "string", "maxLength": 10,
                            "description": "'today', 'tomorrow', or a date as YYYY-MM-DD."},
                    "days": {"type": "integer", "description": f"Number of days, 1-{cfg.calendar_max_days}."},
                },
                "required": [],
            },
            risk="safe",
            func=lambda day="today", days=1: ek.events(day, max(1, min(days, cfg.calendar_max_days))),
            triggers=(r"\b(calendar|schedule|meetings?|agenda|events?)\b", r"\bmy day\b", r"\bam i (free|busy)\b"),
        ),
        Tool(
            name="reminders",
            description="List the user's open (not completed) reminders.",
            parameters={"type": "object", "properties": {}},
            risk="safe",
            func=ek.reminders,
            triggers=(r"\b(my|open|any) reminders\b", r"\bto-?do\b"),
        ),
        Tool(
            name="add_reminder",
            description="Create a reminder in the Reminders app, optionally with a due date and time.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 2, "maxLength": 200},
                    "due": {"type": "string", "maxLength": 25,
                            "description": "Local date and time as YYYY-MM-DD HH:MM, e.g. 2026-11-05 18:00."},
                },
                "required": ["title"],
            },
            risk="safe",
            func=lambda title, due=None: ek.add_reminder(title, due),
            triggers=(r"\bremind me\b", r"\b(set|add|create) (a )?reminder\b"),
        ),
    ]
