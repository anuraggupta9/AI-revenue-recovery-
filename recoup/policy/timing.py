"""Time arithmetic for recovery decisions, in Indian Standard Time.

IST is a fixed +05:30 offset with no daylight saving, so it is defined here
directly rather than through zoneinfo. That avoids a tzdata dependency on
Windows and, more usefully, makes the offset visible in the code that depends on
it — a quiet-hours rule that silently shifts by five and a half hours because the
process runs in UTC is a compliance failure, not a formatting bug.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

# Contact window. Outside it we may still retry silently, but we will not reach
# out to a customer.
CONTACT_OPENS = time(9, 0)
CONTACT_CLOSES = time(19, 0)

# Salary-cycle landing zone. Balance-related failures are retried here rather
# than on a fixed +24/48/72h ladder, which is the main timing claim this project
# makes.
SALARY_DAYS = (1, 2)
SALARY_HOUR = 10


def to_ist(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError("refusing to localise a naive datetime; pass tz-aware input")
    return moment.astimezone(IST)


def ist_stamp(moment: datetime) -> str:
    """The one way a moment is shown to a human.

    Every rule that mentions a time goes through here. Before it existed, the rule
    details formatted `%Z` on whatever tzinfo the datetime happened to carry, so a
    single audit trail read "not due until 2026-07-20 15:01 UTC" on one line and
    "until 2026-07-21 15:00 IST" three lines later — the same policy, described in
    two timezones, in the document a merchant would use to check it. The rules are
    written against IST, so the explanation is too.
    """
    return f"{to_ist(moment):%Y-%m-%d %H:%M} IST"


def is_within_contact_hours(moment: datetime) -> bool:
    local = to_ist(moment).time()
    return CONTACT_OPENS <= local < CONTACT_CLOSES


def next_contact_window(moment: datetime) -> datetime:
    """The soonest instant at which contacting the customer is permitted."""
    local = to_ist(moment)
    if local.time() < CONTACT_OPENS:
        return local.replace(
            hour=CONTACT_OPENS.hour, minute=0, second=0, microsecond=0
        )
    if local.time() >= CONTACT_CLOSES:
        tomorrow = local + timedelta(days=1)
        return tomorrow.replace(hour=CONTACT_OPENS.hour, minute=0, second=0, microsecond=0)
    return local


def next_salary_window(moment: datetime, *, not_before: datetime | None = None) -> datetime:
    """Next 1st or 2nd of a month at 10:00 IST, at or after `moment`.

    `not_before` lets a caller compose this with a cool-off: an
    insufficient-funds failure waits out the cool-off *and* lands on the salary
    window, whichever is later.
    """
    floor = to_ist(moment)
    if not_before is not None:
        floor = max(floor, to_ist(not_before))

    candidate = floor.replace(hour=SALARY_HOUR, minute=0, second=0, microsecond=0)
    for _ in range(70):  # two months of days is plenty; bounded to avoid a spin
        if candidate.day in SALARY_DAYS and candidate >= floor:
            return candidate
        candidate = (candidate + timedelta(days=1)).replace(
            hour=SALARY_HOUR, minute=0, second=0, microsecond=0
        )
    raise RuntimeError("no salary window found within 70 days — check the calendar logic")


def latest(*moments: datetime | None) -> datetime | None:
    """Latest of several deferral times, ignoring None.

    Deferrals compose by taking the maximum, not the minimum: if quiet hours say
    wait until 09:00 and a cool-off says wait 72 hours, both constraints must
    hold, so the later one governs. Getting this backwards would let an action
    fire inside a window it was supposed to respect.
    """
    known = [m for m in moments if m is not None]
    return max(known) if known else None
