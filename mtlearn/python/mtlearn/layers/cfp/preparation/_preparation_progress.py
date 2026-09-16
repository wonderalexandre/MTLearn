def emit(progress, manifest, count, total, *, status="running", position=None, sample_id=None):
    if progress is not None:
        progress({"phase": "prepare", "status": status, "manifest": manifest,
                  "sample_count": count, "total_samples": total,
                  "position": position, "sample_id": sample_id})
