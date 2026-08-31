"""Audit log tests.

The tamper tests are the point of the module, so they exercise all three ways a
log can be falsified: editing an entry, deleting one, and appending a forged one.
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from recoup.audit.log import (
    GENESIS_HASH,
    AuditLog,
    EntryKind,
    canonical_json,
    read_entries,
    verify_chain,
)
from recoup.domain.money import Money

NOW = datetime(2026, 8, 21, 9, 30, tzinfo=timezone.utc)


class TestCanonicalJson(unittest.TestCase):
    def test_key_order_does_not_affect_output(self):
        self.assertEqual(
            canonical_json({"b": 1, "a": 2}),
            canonical_json({"a": 2, "b": 1}),
        )

    def test_money_serialises_as_paise_not_text(self):
        encoded = json.loads(canonical_json({"amount": Money.from_rupees("499")}))
        self.assertEqual(encoded["amount"], {"paise": 49900, "currency": "INR"})

    def test_unserialisable_type_is_refused_loudly(self):
        with self.assertRaises(TypeError):
            canonical_json({"handle": object()})


class TestAppend(unittest.TestCase):
    def test_first_entry_links_to_genesis(self):
        log = AuditLog()
        entry = log.append(EntryKind.EVENT_INGESTED, case_id="case_1", at=NOW)
        self.assertEqual(entry.prev_hash, GENESIS_HASH)
        self.assertEqual(entry.seq, 0)

    def test_entries_chain_to_their_predecessor(self):
        log = AuditLog()
        first = log.append(EntryKind.EVENT_INGESTED, case_id="case_1", at=NOW)
        second = log.append(EntryKind.DIAGNOSED, case_id="case_1", at=NOW)
        self.assertEqual(second.prev_hash, first.entry_hash)

    def test_naive_timestamp_is_refused(self):
        log = AuditLog()
        with self.assertRaises(ValueError):
            log.append(EntryKind.EVENT_INGESTED, case_id="c", at=datetime(2026, 8, 21))

    def test_identical_payloads_still_hash_differently(self):
        # Because prev_hash and seq differ, so a replayed entry is distinguishable
        # from the original rather than colliding with it.
        log = AuditLog()
        a = log.append(EntryKind.EVENT_INGESTED, case_id="c", at=NOW, detail="x")
        b = log.append(EntryKind.EVENT_INGESTED, case_id="c", at=NOW, detail="x")
        self.assertNotEqual(a.entry_hash, b.entry_hash)

    def test_money_survives_a_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            with AuditLog(path) as log:
                log.append(
                    EntryKind.EV_COMPUTED,
                    case_id="c",
                    at=NOW,
                    expected_value=Money.from_rupees("174.65"),
                )
            reloaded = list(read_entries(path))
            self.assertEqual(reloaded[0].payload["expected_value"]["paise"], 17465)


class TestQueries(unittest.TestCase):
    def setUp(self):
        self.log = AuditLog()
        self.log.append(EntryKind.EVENT_INGESTED, case_id="case_1", at=NOW)
        self.log.append(EntryKind.RULE_EVALUATED, case_id="case_1", at=NOW, rule="quiet_hours")
        self.log.append(EntryKind.EVENT_INGESTED, case_id="case_2", at=NOW)

    def test_filter_by_case(self):
        self.assertEqual(len(self.log.for_case("case_1")), 2)

    def test_filter_by_kind(self):
        self.assertEqual(len(self.log.of_kind(EntryKind.EVENT_INGESTED)), 2)

    def test_length_and_iteration(self):
        self.assertEqual(len(self.log), 3)
        self.assertEqual([e.seq for e in self.log], [0, 1, 2])


class TestTamperDetection(unittest.TestCase):
    def _write_log(self, path: Path) -> AuditLog:
        log = AuditLog(path)
        for index in range(5):
            log.append(
                EntryKind.ACTION_EXECUTED,
                case_id=f"case_{index}",
                at=NOW + timedelta(minutes=index),
                amount=Money.from_rupees("100"),
            )
        log.close()
        return log

    def test_intact_chain_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            log = self._write_log(path)
            self.assertTrue(log.verify().ok)
            self.assertTrue(log.verify_on_disk().ok)

    def test_editing_an_entry_in_place_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            log = self._write_log(path)

            lines = path.read_text(encoding="utf-8").splitlines()
            forged = json.loads(lines[2])
            # Someone inflates a recovered amount after the fact.
            forged["payload"]["amount"]["paise"] = 999_999
            lines[2] = json.dumps(forged, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            status = log.verify_on_disk()
            self.assertFalse(status.ok)
            self.assertEqual(status.broken_at, 2)
            self.assertIn("edited in place", status.detail)

    def test_deleting_an_entry_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            log = self._write_log(path)

            lines = path.read_text(encoding="utf-8").splitlines()
            del lines[2]  # hide an action that was taken
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            status = log.verify_on_disk()
            self.assertFalse(status.ok)
            self.assertEqual(status.broken_at, 2)
            self.assertIn("sequence gap", status.detail)

    def test_appending_a_forged_entry_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            log = self._write_log(path)

            # A forger who does not know the tip hash cannot link correctly.
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "seq": 5,
                            "at": NOW.isoformat(),
                            "kind": "action_result",
                            "case_id": "case_fake",
                            "payload": {"succeeded": True},
                            "prev_hash": GENESIS_HASH,
                            "entry_hash": "deadbeef" * 8,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            status = log.verify_on_disk()
            self.assertFalse(status.ok)
            self.assertEqual(status.broken_at, 5)

    def test_reordering_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            log = self._write_log(path)
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[1], lines[3] = lines[3], lines[1]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertFalse(log.verify_on_disk().ok)

    def test_empty_chain_is_valid(self):
        self.assertTrue(verify_chain([]).ok)


class TestPersistence(unittest.TestCase):
    def test_reopening_resumes_the_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            with AuditLog(path) as first:
                first.append(EntryKind.EVENT_INGESTED, case_id="c", at=NOW)
                tip = first.tip_hash

            with AuditLog(path) as second:
                self.assertEqual(len(second), 1)
                self.assertEqual(second.tip_hash, tip)

                second.append(EntryKind.DIAGNOSED, case_id="c", at=NOW)
                self.assertTrue(second.verify_on_disk().ok)
            self.assertEqual(len(list(read_entries(path))), 2)

    def test_truncate_starts_a_fresh_chain(self):
        """The mode every simulation run uses.

        Without it, re-running an arm appends onto the previous run's log. The
        chain still verifies — each entry correctly references its predecessor —
        so nothing complains, and every count read off the file is doubled. A log
        that is both valid and wrong is the worst of the three options.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            with AuditLog(path) as first:
                first.append(EntryKind.EVENT_INGESTED, case_id="c", at=NOW)

            with AuditLog(path, truncate=True) as second:
                self.assertEqual(len(second), 0)
                second.append(EntryKind.EVENT_INGESTED, case_id="d", at=NOW)
                self.assertTrue(second.verify_on_disk().ok)

            entries = list(read_entries(path))
            self.assertEqual([e.case_id for e in entries], ["d"])

    def test_unflushed_entries_are_complete_once_closed(self):
        """What the simulation harness relies on when it turns flushing off."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            log = AuditLog(path, truncate=True, flush_each=False)
            for index in range(200):
                log.append(EntryKind.EVENT_INGESTED, case_id=f"c{index}", at=NOW)
            log.close()
            self.assertEqual(len(list(read_entries(path))), 200)
            self.assertTrue(verify_chain(list(read_entries(path))).ok)

    def test_closing_twice_is_harmless(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AuditLog(Path(tmp) / "audit.jsonl")
            log.close()
            log.close()


if __name__ == "__main__":
    unittest.main()
