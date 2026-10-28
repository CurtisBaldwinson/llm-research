"""Data-staging regressions that require neither PyTorch nor network access."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.trace import fetch_data


class DataStagingChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trace = self.root / "TRACE"
        for i, task in enumerate(fetch_data.TRACE_TASKS):
            folder = self.trace / task
            folder.mkdir(parents=True)
            (folder / "train.json").write_text(json.dumps([{}] * 5000))
            (folder / "test.json").write_text(json.dumps([{}] * (i + 1)))
        self.first = self.trace / fetch_data.TRACE_TASKS[0]

    def test_uneven_held_out_counts_are_preserved(self):
        manifest = fetch_data._assert_record_counts(self.trace)
        self.assertEqual(
            [manifest[t]["test"] for t in fetch_data.TRACE_TASKS],
            list(range(1, 9)),
        )

    def test_small_training_variant_is_rejected(self):
        (self.first / "train.json").write_text(json.dumps([{}] * 1000))
        with self.assertRaisesRegex(RuntimeError, "wrong-size variant"):
            fetch_data._assert_record_counts(self.trace)

    def test_empty_held_out_split_is_rejected(self):
        (self.first / "test.json").write_text("[]")
        with self.assertRaisesRegex(RuntimeError, "empty/missing held-out split"):
            fetch_data._assert_record_counts(self.trace)

    def test_object_instead_of_records_is_rejected(self):
        (self.first / "test.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "not a JSON list"):
            fetch_data._assert_record_counts(self.trace)

    def test_rerun_refreshes_stale_provenance_without_download(self):
        provenance = self.trace / "_provenance.json"
        provenance.write_text('{"expected_test_count": 2000}')
        with patch.object(
            fetch_data,
            "_download_archive",
            side_effect=AssertionError("unexpected download"),
        ):
            self.assertEqual(fetch_data.stage_trace(self.root), self.trace)
            first = provenance.read_text()
            self.assertEqual(fetch_data.stage_trace(self.root), self.trace)
            self.assertEqual(provenance.read_text(), first)
        doc = json.loads(first)
        self.assertNotIn("expected_test_count", doc)
        self.assertEqual(doc["min_test_count"], 1)
        self.assertEqual(doc["record_counts"][fetch_data.TRACE_TASKS[-1]]["test"], 8)
        self.assertTrue(doc["archive_sha256"].startswith("unknown"))


if __name__ == "__main__":
    unittest.main()
