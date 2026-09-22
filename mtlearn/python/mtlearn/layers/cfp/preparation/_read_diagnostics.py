import os
import sys
import time

try:
    import psutil
except ImportError:
    psutil = None


def process_memory():
    result = {"pid": os.getpid(), "rss_bytes": None, "lifetime_peak_rss_bytes": None,
              "cpu_seconds": time.process_time()}
    if psutil is not None:
        try:
            result["rss_bytes"] = psutil.Process().memory_info().rss
        except (OSError, psutil.Error):
            pass
    try:
        import resource
        result["lifetime_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    except (ImportError, OSError):
        pass
    return result


def available_memory():
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except (OSError, psutil.Error):
            pass
    return None
