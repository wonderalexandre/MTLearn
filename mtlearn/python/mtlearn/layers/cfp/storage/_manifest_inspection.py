import json

from ..preparation._identity import preprocessor_from_config


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _cancelled(cancel):
    return cancel is not None and (cancel() if callable(cancel) else cancel.is_set())


def inspect_manifest(store, name, *, offset, limit):
    _integer("offset", offset)
    _integer("limit", limit, 1)
    if limit > 1000:
        raise ValueError("limit cannot exceed 1000 samples per page.")
    generation = store._read_generation()
    header = store._manifest_record(name, complete=False)
    config = json.loads(header["config"])
    compatibility_error = None
    try:
        preprocessor_from_config(config)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        compatibility_error = str(exc)
    bound = store._db.execute("SELECT COUNT(*) FROM samples WHERE manifest=?", (name,)).fetchone()[0]
    entry_counts = list(store._db.execute("""SELECT COALESCE(e.state,'missing') AS state,
        COUNT(*) AS count,COALESCE(SUM(e.size_bytes),0) AS bytes
        FROM (SELECT DISTINCT entry_key FROM contributions WHERE manifest=?) c
        LEFT JOIN entries e ON e.key=c.entry_key GROUP BY e.state""", (name,)))
    rows = store._db.execute("SELECT * FROM samples WHERE manifest=? AND position>=? ORDER BY position LIMIT ?",
                             (name, offset, limit)).fetchall()
    samples = []
    for row in rows:
        entries = [dict(entry) for entry in store._db.execute("""SELECT c.channel,c.tree_key,
            c.entry_key,COALESCE(e.state,'missing') AS state,e.size_bytes,e.error
            FROM contributions c LEFT JOIN entries e ON e.key=c.entry_key
            WHERE c.manifest=? AND c.position=? ORDER BY c.channel,c.tree_key""", (name, row["position"]))]
        samples.append({"position": row["position"], "sample_id": row["sample_id"],
                        "shape": json.loads(row["shape"]), "bindings": json.loads(row["bindings"]),
                        "entries": entries})
    next_offset = None
    if rows:
        following = rows[-1]["position"] + 1
        if store._db.execute("SELECT 1 FROM samples WHERE manifest=? AND position>=? LIMIT 1",
                              (name, following)).fetchone() is not None:
            next_offset = following
    if store._read_generation() != generation:
        raise RuntimeError("Manifest database changed during inspection; retry the page.")
    return {"name": name, "state": header["state"], "sample_count": header["sample_count"],
            "bound_samples": bound, "source_version": header["source_version"],
            "preprocessing_version": header["preprocessing_version"], "split": header["split"],
            "config": config, "compatible": compatibility_error is None,
            "compatibility_error": compatibility_error, "unique_entries": sum(row["count"] for row in entry_counts),
            "registered_disk_bytes": sum(row["bytes"] for row in entry_counts),
            "entry_states": {row["state"]: row["count"] for row in entry_counts},
            "offset": offset, "limit": limit, "samples": samples, "next_offset": next_offset}


def validate_manifest(store, name, *, offset, limit, progress, cancel, max_errors):
    from .disk_store import _ENTRY_ERRORS
    _integer("offset", offset)
    _integer("max_errors", max_errors)
    if limit is not None:
        _integer("limit", limit, 1)
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None.")
    if cancel is not None and not callable(cancel) and not callable(getattr(cancel, "is_set", None)):
        raise TypeError("cancel must be callable, an event, or None.")
    generation = store._read_generation()
    header = store._manifest_record(name, complete=False)
    if offset > header["sample_count"]:
        raise ValueError("offset exceeds the manifest sample count.")
    end = header["sample_count"] if limit is None else min(header["sample_count"], offset + limit)
    report = {"name": name, "manifest_state": header["state"], "status": "running",
              "offset": offset, "scope_samples": end-offset, "checked_samples": 0,
              "checked_entries": 0, "valid_entries": 0, "error_count": 0, "errors": [],
              "errors_truncated": False, "fully_validated": False}
    generation_changed = False

    def error(position, sample_id, key, message):
        report["error_count"] += 1
        store._memory.clear()
        store._validation.clear()
        if len(report["errors"]) < max_errors:
            report["errors"].append({"position": position, "sample_id": sample_id,
                                     "entry_key": key, "message": message})
        else:
            report["errors_truncated"] = True

    def notify(position=None, sample_id=None):
        if progress is not None:
            progress({key: report[key] for key in ("name", "status", "scope_samples", "checked_samples",
                "checked_entries", "valid_entries", "error_count")} | {"position": position, "sample_id": sample_id})

    def changed():
        nonlocal generation_changed
        if generation_changed:
            return True
        if store._read_generation() != generation:
            error(None, None, None, "Manifest database changed during validation; start validation again.")
            generation_changed = True
            return True
        return False

    notify()
    try:
        preprocessor = preprocessor_from_config(json.loads(header["config"]))
    except _ENTRY_ERRORS as exc:
        error(None, None, None, str(exc))
        report["status"] = "invalid"
        notify()
        return report
    if offset == 0 and end == header["sample_count"]:
        counts = store._db.execute("SELECT COUNT(*),MIN(position),MAX(position) FROM samples WHERE manifest=?",
                                   (name,)).fetchone()
        if counts[0] != header["sample_count"] or (counts[0] and (counts[1] != 0 or counts[2] != counts[0]-1)):
            error(None, None, None, "Manifest sample positions/count are incomplete or inconsistent.")
    cancelled = False
    for position in range(offset, end):
        if _cancelled(cancel):
            cancelled = True
            break
        if changed():
            break
        row = store._db.execute("SELECT * FROM samples WHERE manifest=? AND position=?", (name, position)).fetchone()
        sample_id = None if row is None else row["sample_id"]
        try:
            if row is None:
                raise ValueError("Manifest sample is missing.")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("Manifest sample ID must be a nonempty string.")
            bindings, shape = json.loads(row["bindings"]), json.loads(row["shape"])
            if (not isinstance(shape, list) or len(shape) != 3 or
                    any(type(n) is not int or n <= 0 for n in shape) or
                    not isinstance(bindings, list) or len(bindings) != shape[0] or
                    any(not isinstance(trees, dict) or set(trees) != set(preprocessor.tree_specs) for trees in bindings)):
                raise ValueError("Sample shape/tree/channel bindings do not match the manifest contract.")
            expected = {(channel, tree_key, key) for channel, trees in enumerate(bindings)
                        for tree_key, key in trees.items()}
            actual = {tuple(item) for item in store._db.execute(
                "SELECT channel,tree_key,entry_key FROM contributions WHERE manifest=? AND position=?", (name, position))}
            if expected != actual:
                raise ValueError("Sample bindings and statistical contributions disagree.")
        except _ENTRY_ERRORS as exc:
            error(position, sample_id, None, str(exc))
            report["checked_samples"] += 1
            notify(position, sample_id)
            continue
        for channel, trees in enumerate(bindings):
            for tree_key, key in trees.items():
                if _cancelled(cancel):
                    cancelled = True
                    break
                prepared = None
                try:
                    store._entry_path(key)
                    entry = store._db.execute("SELECT * FROM entries WHERE key=?", (key,)).fetchone()
                    report["checked_entries"] += 1
                    if entry is None or entry["state"] != "valid":
                        raise ValueError("Prepared entry is missing or incomplete.")
                    prepared, _ = store._read_entry(entry, force=True, full=True)
                    if (prepared.image_shape != tuple(shape[1:]) or
                            prepared.tree_spec != preprocessor.tree_specs[tree_key] or
                            prepared.feature_spec != preprocessor.features[tree_key] or
                            any(str(value.dtype) != "torch." + preprocessor.attribute_dtype.name
                                for value in prepared.raw_attributes.values())):
                        raise ValueError("Prepared entry does not match the sample/manifest contract.")
                    report["valid_entries"] += 1
                except _ENTRY_ERRORS as exc:
                    error(position, sample_id, key, str(exc))
                finally:
                    prepared = None
            if cancelled:
                break
        if cancelled:
            break
        report["checked_samples"] += 1
        notify(position, sample_id)
    if not cancelled:
        changed()
    report["status"] = ("cancelled" if cancelled else "invalid" if report["error_count"] else
                        "incomplete" if header["state"] != "complete" else "valid")
    report["fully_validated"] = (report["status"] == "valid" and offset == 0 and
                                report["checked_samples"] == header["sample_count"])
    notify()
    return report
