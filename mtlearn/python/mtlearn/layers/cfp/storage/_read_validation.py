from collections import OrderedDict
import os
from pathlib import Path
import stat


def file_signature(value):
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Prepared entry must be a regular file.")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def descriptor_path(handle):
    for root in ("/proc/self/fd", "/dev/fd"):
        path = Path(root) / str(handle.fileno())
        if path.exists():
            return path
    return None


class ReadValidationCache:
    def __init__(self, max_entries):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.checksums = self.checksum_bytes = self.hits = self.invalidations = 0

    def matches(self, key, signature):
        if self.entries.get(key) != signature:
            return False
        self.entries.move_to_end(key)
        self.hits += 1
        return True

    def remember(self, key, signature):
        if self.max_entries == 0:
            return
        self.entries[key] = signature
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

    def clear(self):
        self.entries.clear()
        self.invalidations += 1


class ReadSessionLock:
    def __init__(self, path):
        if os.name != "posix":
            raise NotImplementedError("Session validation currently requires POSIX shared file locks; use validation='always'.")
        import fcntl
        self.handle = None
        try:
            self.handle = open(path, "rb")
            fcntl.flock(self.handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            self.close()
            raise RuntimeError("Cannot establish an immutable read session: a writer/worker is active or its lock is unavailable.") from exc

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
