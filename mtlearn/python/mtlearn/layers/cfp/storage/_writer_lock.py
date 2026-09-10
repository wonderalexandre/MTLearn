"""OS-owned single-coordinator lock, released automatically after process exit."""
import os


class _WriterLock:
    def __init__(self, path):
        self.handle = open(path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.handle.tell() == 0:
                    self.handle.write(b"\0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError("A DiskStore writer is already active; use readonly=True for readers.") from exc

    def close(self):
        if not self.handle.closed:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()


def fsync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class _QuotaWriter:
    def __init__(self, handle, available):
        self.handle, self.available = handle, available
        self.exceeded = False

    def write(self, data):
        if self.handle.tell() + len(data) > self.available:
            self.exceeded = True
            raise OSError("DiskStore quota exceeded; completed entries remain reusable. Increase max_disk_bytes to resume.")
        return self.handle.write(data)

    def flush(self):
        return self.handle.flush()

    def tell(self):
        return self.handle.tell()
