"""Recoverable local CPU tensor storage with a single SQLite coordinator."""
import json
import os
import pickle
from pathlib import Path
import re
import uuid
from collections import OrderedDict

import torch

from .memory_store import MemoryStore
from ._disk_format import pack, unpack
from ._manifest import connect
from ._writer_lock import _WriterLock, _QuotaWriter, fsync_directory
from ._read_validation import ReadSessionLock, ReadValidationCache, descriptor_path, file_signature
from ..preparation._identity import canonical_json, digest_file, identity_key, validate_identity

_ENTRY_ERRORS = (OSError, ValueError, RuntimeError, KeyError, EOFError, TypeError, AttributeError, pickle.UnpicklingError)
_KEY = re.compile(r"[0-9a-f]{64}\Z")
_WORKER_LEASE = re.compile(r"\.worker\.[0-9a-f]{32}\.lock\Z")
_TEMP = re.compile(r"\.[0-9a-f]{64}\.[0-9a-f]{32}\.tmp\Z")


class DiskStore:
    """Versioned local SSD entries with bounded RAM and restricted CPU loading.

    Set a tensor-file quota explicitly for a writer. ``max_ram_bytes=0`` is the
    default. Quota counts published files and temporary tensor files; SQLite,
    lock files, filesystem allocation and OS page cache are reported separately.
    One writer may coexist with readonly readers. Use each handle sequentially
    in the process that opened it, and close it (or use a context manager).
    """

    retains_entries = True
    persistent = True

    def __init__(self, path, *, max_disk_bytes=None, max_ram_bytes=0, readonly=False, mmap=True,
                 validation="always", immutable=False, max_validation_entries=1024, min_free_disk_bytes=0):
        if max_disk_bytes is not None and (type(max_disk_bytes) is not int or max_disk_bytes < 0):
            raise ValueError("max_disk_bytes must be a nonnegative integer.")
        if not readonly and max_disk_bytes is None:
            raise ValueError("A writer requires an explicit max_disk_bytes quota.")
        if type(min_free_disk_bytes) is not int or min_free_disk_bytes < 0:
            raise ValueError("min_free_disk_bytes must be a nonnegative integer.")
        if validation not in ("always", "session"):
            raise ValueError("validation must be 'always' or 'session'.")
        if type(immutable) is not bool:
            raise ValueError("immutable must be a boolean.")
        if validation == "session" and (not readonly or not immutable):
            raise ValueError("validation='session' requires readonly=True and immutable=True.")
        if type(max_validation_entries) is not int or max_validation_entries < 0:
            raise ValueError("max_validation_entries must be a nonnegative integer.")
        self.path = Path(path).expanduser().resolve()
        self.readonly, self.mmap = bool(readonly), bool(mmap)
        self.max_disk_bytes = max_disk_bytes
        self.min_free_disk_bytes = min_free_disk_bytes
        self.validation = validation
        self.immutable = immutable
        self._memory = MemoryStore(max_ram_bytes)
        self._pid = os.getpid()
        self._db = self._lock = self._session_lock = None
        self._validation = ReadValidationCache(max_validation_entries)
        self._manifest_rows = OrderedDict()
        self._generation = self._database_identity = None
        self._hits = self._misses = self._disk_hits = self._writes = self._invalid = 0
        self._logical_load_bytes = 0
        self._files = self.path / "entries"
        try:
            if not readonly:
                self._files.mkdir(parents=True, exist_ok=True)
                self._lock = _WriterLock(self.path / ".writer.lock")
                self._check_worker_leases()
            if validation == "session":
                self._session_lock = ReadSessionLock(self.path / ".writer.lock")
                with os.scandir(self.path) as paths:
                    for entry in paths:
                        if _WORKER_LEASE.fullmatch(entry.name):
                            lease = ReadSessionLock(entry.path)
                            lease.close()
            self._db = connect(self.path / "manifest.sqlite3", readonly=readonly)
            self._database_contract = json.loads(self._db.execute(
                "SELECT value FROM metadata WHERE key='contract'").fetchone()[0])
            value = (self.path / "manifest.sqlite3").stat()
            self._database_identity = (value.st_dev, value.st_ino)
            if not readonly:
                self.recover()
        except BaseException:
            self.close()
            raise

    def _check_worker_leases(self):
        # A coordinator can die while spawned CPU processes are still writing
        # unique temporaries. Delay recovery until their OS-owned leases expire.
        # Workers require a live parent handshake after acquiring their lease.
        with os.scandir(self.path) as paths:
            for entry in paths:
                if _WORKER_LEASE.fullmatch(entry.name):
                    try:
                        lease = _WriterLock(entry.path)
                    except RuntimeError as exc:
                        raise RuntimeError("A CFP preparation worker is still active after its coordinator stopped; "
                                           "wait for it to exit before reopening the writer.") from exc
                    lease.close()
                    Path(entry.path).unlink(missing_ok=True)

    def _check(self, *, write=False):
        if os.getpid() != self._pid:
            raise RuntimeError("Open a separate DiskStore handle in each process.")
        if self._db is None:
            raise RuntimeError("DiskStore is closed.")
        if write and self.readonly:
            raise RuntimeError("DiskStore is readonly; reopen with a writer quota to prepare or repair entries.")

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        self._memory.clear()
        self._validation.clear()
        self._manifest_rows.clear()
        if self._lock is not None:
            self._lock.close()
            self._lock = None
        if self._session_lock is not None:
            self._session_lock.close()
            self._session_lock = None

    def __del__(self):
        if getattr(self, "_pid", None) == os.getpid():
            self.close()

    def _entry_path(self, key):
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Entry keys must be SHA-256 hexadecimal strings.")
        return self._files / f"{key}.pt"

    def _load_file(self, path, *, full=False):
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=self.mmap)
        prepared, summary = unpack(data, full=full)
        return data["identity"], prepared, summary

    def _failure(self, key, error):
        self._invalid += 1
        if not self.readonly:
            with self._db:
                self._db.execute("UPDATE entries SET state='failed',error=? WHERE key=?", (str(error), key))
        # A stale resident entry must not mask an explicitly detected disk error.
        self._memory.clear()
        self._validation.clear()

    def _read_generation(self):
        self._check()
        try:
            value = (self.path / "manifest.sqlite3").stat()
            replaced = (value.st_dev, value.st_ino) != self._database_identity
        except FileNotFoundError:
            replaced = True
        if replaced:
            self._memory.clear()
            self._validation.clear()
            self._manifest_rows.clear()
            raise RuntimeError("Manifest database was replaced or removed; reopen DiskStore.")
        generation = (self._db.execute("PRAGMA data_version").fetchone()[0], self._db.total_changes)
        if generation != self._generation:
            self._manifest_rows.clear()
            if self._generation is not None:
                self._validation.clear()
                if self.validation == "session":
                    self._memory.clear()
            contract = self._db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
            if contract is None or json.loads(contract[0]) != self._database_contract:
                self._memory.clear()
                raise ValueError("DiskStore format/backend contract changed; reopen with a compatible store.")
            self._generation = generation
        return generation

    def _manifest_record(self, name, *, complete=True):
        self._read_generation()
        row = self._manifest_rows.get(name)
        if row is None:
            row = self._db.execute("SELECT * FROM manifests WHERE name=?", (name,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown prepared manifest {name!r}.")
            if self._validation.max_entries:
                self._manifest_rows[name] = row
                while len(self._manifest_rows) > self._validation.max_entries:
                    self._manifest_rows.popitem(last=False)
        else:
            self._manifest_rows.move_to_end(name)
        if complete and row["state"] != "complete":
            raise RuntimeError(f"Manifest {name!r} is {row['state']}; resume preparation before consumption.")
        return row

    def _read_entry(self, row, *, force=False, full=False):
        key = row["key"]
        path = self._entry_path(key)
        with path.open("rb") as handle:
            signature = file_signature(os.fstat(handle.fileno()))
            if signature[2] != row["size_bytes"]:
                raise ValueError("Persistent file checksum/size mismatch.")
            token = (signature, row["sha256"], row["identity"], row["summary"])
            reused = self.validation == "session" and not force and self._validation.matches(key, token)
            stable_path = descriptor_path(handle)
            if self.validation == "session" and stable_path is None:
                raise RuntimeError("Session validation requires loading through an open file descriptor.")
            if not reused:
                self._validation.checksums += 1
                self._validation.checksum_bytes += signature[2]
                if digest_file(stable_path or path) != row["sha256"]:
                    raise ValueError("Persistent file checksum/size mismatch.")
            cached = self._memory.get(key) if reused and not full else None
            if cached is None:
                handle.seek(0)
                self._logical_load_bytes += signature[2]
                identity, prepared, summary = self._load_file(stable_path or path, full=full)
                if identity_key(identity) != key or canonical_json(identity) != row["identity"]:
                    raise ValueError("Persistent entry identity mismatch.")
                if canonical_json(summary) != row["summary"]:
                    raise ValueError("Persistent entry summary mismatch.")
            else:
                prepared = cached
            if (file_signature(os.fstat(handle.fileno())) != signature or
                    file_signature(path.stat()) != signature):
                raise ValueError("Prepared file changed during validation/loading; retry with an immutable entry.")
            if self.validation == "session":
                self._validation.remember(key, token)
            return prepared, cached is not None

    def _register_valid(self, key, identity, path, summary, expected_digest):
        with self._db:
            self._db.execute("""INSERT INTO entries VALUES (?,?, 'valid',?,?,?,NULL)
                ON CONFLICT(key) DO UPDATE SET identity=excluded.identity,state='valid',
                size_bytes=excluded.size_bytes,sha256=excluded.sha256,summary=excluded.summary,error=NULL""",
                (key, canonical_json(identity), path.stat().st_size, expected_digest, canonical_json(summary)))

    def _recover_file(self, key, path):
        expected_digest = digest_file(path)
        identity, prepared, summary = self._load_file(path, full=True)
        if identity_key(identity) != key:
            raise ValueError("Orphan file identity does not match its filename.")
        self._register_valid(key, identity, path, summary, expected_digest)
        return prepared

    def recover(self):
        """Recover fully published orphans; discard owned partial temporary files.

        Valid registered files are checked lazily on reads. Recovery visits one
        entry at a time and never retains a dataset of mapped tensors.
        """
        self._check(write=True)
        counts = {"recovered": 0, "partial_files_removed": 0, "failed": 0}
        with os.scandir(self._files) as paths:
            for entry in paths:
                if _TEMP.fullmatch(entry.name) and entry.is_file():
                    os.unlink(entry.path)
                    counts["partial_files_removed"] += 1
                elif entry.name.endswith(".pt") and _KEY.fullmatch(entry.name[:-3]):
                    key = entry.name[:-3]
                    row = self._db.execute("SELECT state FROM entries WHERE key=?", (key,)).fetchone()
                    if row is None or row["state"] != "valid":
                        try:
                            prepared = self._recover_file(key, Path(entry.path))
                            del prepared
                            counts["recovered"] += 1
                        except _ENTRY_ERRORS as exc:
                            self._failure(key, exc)
                            counts["failed"] += 1
        with self._db:
            self._db.execute("UPDATE entries SET state='failed',error='Interrupted preparation' WHERE state='preparing'")
        return counts

    def get(self, key):
        """Read a content key, verifying file checksum before restricted loading."""
        self._read_generation()
        path = self._entry_path(key)
        row = self._db.execute("SELECT * FROM entries WHERE key=?", (key,)).fetchone()
        if row is None or row["state"] != "valid":
            self._misses += 1
            return None
        cached = self._memory.get(key) if self.validation == "always" else None
        if cached is not None:
            self._hits += 1
            return cached
        if not path.is_file():
            self._failure(key, "Missing tensor file")
            self._misses += 1
            return None
        try:
            prepared, ram_hit = self._read_entry(row)
        except _ENTRY_ERRORS as exc:
            self._failure(key, exc)
            raise ValueError(f"Invalid prepared entry {key}: {exc}") from exc
        self._hits += 1
        if not ram_hit:
            self._disk_hits += 1
            return self._memory.put(key, prepared)
        return prepared

    def get_or_prepare(self, identity, factory):
        """Resolve canonical content, or prepare/repair it without touching model stats."""
        key = validate_identity(identity)
        try:
            prepared = self.get(key)
        except ValueError:
            if self.readonly:
                raise
            prepared = None
        if prepared is not None:
            return prepared
        self._check(write=True)
        with self._db:
            self._db.execute("""INSERT INTO entries(key,identity,state) VALUES (?,?,'preparing')
                ON CONFLICT(key) DO UPDATE SET state='preparing',error=NULL""", (key, canonical_json(identity)))
        try:
            return self.put(identity, factory())
        except BaseException as exc:
            self._failure(key, exc)
            raise

    def put(self, identity, prepared):
        """Publish a complete entry atomically; never delete a valid entry for quota."""
        self._check(write=True)
        key = validate_identity(identity)
        path = self._entry_path(key)
        row = self._db.execute("SELECT state FROM entries WHERE key=?", (key,)).fetchone()
        if row is not None and row["state"] == "valid":
            existing = self.get(key)
            if existing is not None:
                return existing
        data = pack(identity, prepared)
        with self._db:
            self._db.execute("""INSERT INTO entries(key,identity,state) VALUES (?,?,'preparing')
                ON CONFLICT(key) DO UPDATE SET state='preparing',error=NULL""", (key, canonical_json(identity)))
        temporary = self._files / f".{key}.{uuid.uuid4().hex}.tmp"
        quota_writer = None
        try:
            remaining = self.max_disk_bytes - self._tensor_file_bytes()
            with temporary.open("xb") as handle:
                quota_writer = _QuotaWriter(handle, remaining, min_free_disk_bytes=self.min_free_disk_bytes,
                                            disk_path=self.path)
                torch.save(data, quota_writer)
                handle.flush()
                os.fsync(handle.fileno())
            expected_digest = digest_file(temporary)
            verified_identity, verified, verified_summary = self._load_file(temporary, full=True)
            if verified_identity != identity or verified_summary != data["summary"]:
                raise ValueError("Serialized preparation does not match the completed task.")
            del verified
            # File checksum in SQLite plus embedded checksums make both ordinary
            # reads and publication-before-registration recovery verifiable.
            os.replace(temporary, path)
            fsync_directory(self._files)
            self._register_valid(key, identity, path, data["summary"], expected_digest)
            self._writes += 1
        except BaseException as exc:
            self._failure(key, exc)
            if quota_writer is not None and getattr(quota_writer, "exceeded", False):
                raise OSError("DiskStore quota exceeded; increase max_disk_bytes and resume. Completed entries were preserved.") from exc
            if quota_writer is not None and quota_writer.reserve_exceeded:
                raise OSError("DiskStore free-space reserve reached; free disk space and resume. Completed entries were preserved.") from exc
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return self._memory.put(key, prepared)

    def _tensor_file_bytes(self):
        with os.scandir(self._files) as entries:
            return sum(entry.stat().st_size for entry in entries if entry.is_file())

    def clear(self):
        """Release only the RAM cache. Persistent entries and active handles survive."""
        self._memory.clear()

    def counters(self):
        self._check()
        ram = self._memory.info()
        return {**ram, "ram_hits": ram["hits"], "hits": self._hits, "misses": self._misses,
                "disk_hits": self._disk_hits, "writes": self._writes, "invalid_entries": self._invalid,
                "validation": self.validation, "checksum_reads": self._validation.checksums,
                "logical_load_bytes": self._logical_load_bytes,
                "checksum_bytes": self._validation.checksum_bytes, "validation_hits": self._validation.hits,
                "validation_entries": len(self._validation.entries),
                "validation_invalidations": self._validation.invalidations,
                "max_validation_entries": self._validation.max_entries,
                "manifest_cache_entries": len(self._manifest_rows)}

    def inspect_manifest(self, name, *, offset=0, limit=100):
        from ._manifest_inspection import inspect_manifest
        return inspect_manifest(self, name, offset=offset, limit=limit)

    def validate_manifest(self, name, *, offset=0, limit=None, progress=None, cancel=None, max_errors=100):
        from ._manifest_inspection import validate_manifest
        return validate_manifest(self, name, offset=offset, limit=limit, progress=progress,
                                 cancel=cancel, max_errors=max_errors)

    def info(self):
        self._check()
        ram = self._memory.info()
        counts = dict(self._db.execute("SELECT state,COUNT(*) FROM entries GROUP BY state"))
        metadata_bytes = sum(p.stat().st_size for p in self.path.glob("manifest.sqlite3*") if p.is_file())
        return {**ram, "ram_hits": ram["hits"], "hits": self._hits, "misses": self._misses,
                "disk_hits": self._disk_hits, "writes": self._writes, "invalid_entries": self._invalid,
                "disk_entries": counts.get("valid", 0), "entry_states": counts,
                "disk_bytes": self._tensor_file_bytes(), "manifest_bytes": metadata_bytes,
                "max_disk_bytes": self.max_disk_bytes, "readonly": self.readonly}

    def statistics(self, manifest, contract):
        """Recombine persisted scalar summaries in manifest order, without trees."""
        from ..preparation._persistent_preparation import statistics_from_manifest
        return statistics_from_manifest(self, manifest, contract)
