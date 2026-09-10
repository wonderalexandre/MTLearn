"""SQLite schema for immutable entries and ordered logical sample bindings."""
import json
import sqlite3

from ..preparation._identity import canonical_json, implementation_identity

SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE entries (
    key TEXT PRIMARY KEY, identity TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('preparing','valid','failed')),
    size_bytes INTEGER, sha256 TEXT, summary TEXT, error TEXT
);
CREATE TABLE manifests (
    name TEXT PRIMARY KEY, config TEXT NOT NULL, source_version TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL, split TEXT NOT NULL,
    sample_count INTEGER NOT NULL, state TEXT NOT NULL, error TEXT
);
CREATE TABLE samples (
    manifest TEXT NOT NULL REFERENCES manifests(name), position INTEGER NOT NULL,
    sample_id TEXT NOT NULL, bindings TEXT NOT NULL, shape TEXT NOT NULL,
    PRIMARY KEY(manifest, position), UNIQUE(manifest, sample_id)
);
CREATE TABLE contributions (
    manifest TEXT NOT NULL, position INTEGER NOT NULL, channel INTEGER NOT NULL,
    tree_key TEXT NOT NULL, entry_key TEXT NOT NULL REFERENCES entries(key),
    PRIMARY KEY(manifest, position, channel, tree_key),
    FOREIGN KEY(manifest, position) REFERENCES samples(manifest, position)
);
"""


def connect(path, *, readonly):
    exists = path.exists()
    if readonly and not exists:
        raise FileNotFoundError(path)
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) if readonly else sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        if not readonly:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
        expected = {"format_version": 2, "implementation": implementation_identity()}
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if not tables and not readonly:
            # Schema and version marker commit together. An interruption during
            # initial creation leaves an empty database that a new writer repairs.
            db.executescript("BEGIN IMMEDIATE;" + SCHEMA)
            try:
                db.execute("INSERT INTO metadata VALUES (?,?)", ("contract", canonical_json(expected)))
                db.commit()
            except BaseException:
                db.rollback()
                raise
        row = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if row is None or json.loads(row[0]) != expected:
            raise ValueError("DiskStore format/backend is incompatible; use a separate store directory.")
    except BaseException:
        db.close()
        raise
    return db
