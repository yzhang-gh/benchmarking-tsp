from datetime import datetime


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DotDict ' + dict.__repr__(self) + '>'


def info(text):
    return f"\033[94m{text}\033[0m"


def bold(text):
    return f"\033[1m{text}\033[0m"


def human_readable_time(seconds):
    if seconds < 60:
        return f"{seconds:5.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:5.2f}m"
    else:
        return f"{seconds / 3600:5.2f}h"


def datetime_str():
    return f"{datetime.now():%Y%m%d_%H%M%S}"
