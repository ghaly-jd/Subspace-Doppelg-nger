from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from shared.archive_storage import delete_registration, load_archive_entries


class ArchiveStorageTests(unittest.TestCase):
    def test_load_archive_entries_returns_database_registrations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            database_path = root / "database.json"
            database_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "registrations": [
                            {
                                "id": "reg-1",
                                "nickname": "Student A",
                                "motion_type": "Motion Ritual",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            entries = load_archive_entries(config)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["id"], "reg-1")

    def test_delete_registration_removes_database_entry_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            registrations_dir = root / "registrations"
            registrations_dir.mkdir()
            sequence_path = registrations_dir / "reg-1_sequence.npy"
            nested_path = registrations_dir / "reg-1_nested.npy"
            metadata_path = registrations_dir / "reg-1.json"
            keep_path = registrations_dir / "keep_sequence.npy"
            sequence_path.write_bytes(b"sequence")
            nested_path.write_bytes(b"nested")
            keep_path.write_bytes(b"keep")
            metadata_path.write_text(
                json.dumps(
                    {
                        "id": "reg-1",
                        "sequence_path": str(sequence_path),
                        "ritual": {
                            "full": {
                                "normalized_sequence_path": str(nested_path),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            database_path = root / "database.json"
            database_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "registrations": [
                            {
                                "id": "reg-1",
                                "metadata_path": str(metadata_path),
                                "sequence_path": str(sequence_path),
                            },
                            {
                                "id": "keep",
                                "sequence_path": str(keep_path),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = delete_registration(config, "reg-1")

            self.assertTrue(result["deleted"])
            self.assertFalse(sequence_path.exists())
            self.assertFalse(nested_path.exists())
            self.assertFalse(metadata_path.exists())
            self.assertTrue(keep_path.exists())
            remaining = json.loads(database_path.read_text(encoding="utf-8"))["registrations"]
            self.assertEqual([entry["id"] for entry in remaining], ["keep"])

    def _config(self, root: Path) -> dict:
        return {
            "data": {
                "root": str(root),
                "registrations_dir": str(root / "registrations"),
                "sessions_dir": str(root / "sessions"),
                "database_path": str(root / "database.json"),
            }
        }


if __name__ == "__main__":
    unittest.main()
