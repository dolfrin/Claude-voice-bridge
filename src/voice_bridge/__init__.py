"""Claude Voice Bridge package."""

import os

__version__ = "0.1.0"

# Take python-telegram-bot's upcoming behaviour now (durations such as
# RetryAfter.retry_after as datetime.timedelta) instead of its deprecation
# path; the retry helper already handles both.
os.environ.setdefault("PTB_TIMEDELTA", "1")
