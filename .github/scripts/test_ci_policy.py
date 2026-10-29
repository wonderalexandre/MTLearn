"""Check public validation, promotion triggers and the aggregate CI result."""

import copy
import json
import os
from pathlib import Path
import subprocess
import unittest

import yaml


WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


def lookup(context, path):
    value = context
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def should_validate(expression, context):
    """Evaluate the policy's limited OR/inequality syntax, including toJSON."""
    def operand(text):
        text = text.strip()
        if text.startswith("'") and text.endswith("'"):
            return text[1:-1]
        if text.startswith("toJSON(") and text.endswith(")"):
            return json.dumps(lookup(context, text[7:-1]))
        if not text.startswith("github."):
            raise AssertionError(f"Unsupported policy operand: {text}")
        return lookup(context, text)

    for clause in expression.split("||"):
        left, right = map(operand, clause.strip().split(" != "))
        # Actions compares strings without regard to case.
        if isinstance(left, str) and isinstance(right, str):
            left, right = left.casefold(), right.casefold()
        if left != right:
            return True
    return False


class CiPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ci = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
        cls.condition = cls.ci["repository-promotion"]["if"]

    def test_private_validation_never_allocates_runners(self):
        for event in ("pull_request", "push", "schedule", "workflow_dispatch"):
            for changed_files in (0, 1, None):
                with self.subTest(event=event, changed_files=changed_files):
                    context = {"github": {
                        "repository": "wonderalexandre/MTLearn-dev",
                        "event_name": event,
                        "event": {"pull_request": {"changed_files": changed_files}},
                    }}
                    self.assertFalse(should_validate(self.condition, context))

    def test_public_and_fork_validation_remain_enabled(self):
        for repository in ("wonderalexandre/MTLearn", "someone/MTLearn", None):
            for event in ("pull_request", "push", "schedule", "workflow_dispatch"):
                with self.subTest(repository=repository, event=event):
                    self.assertTrue(should_validate(self.condition, {"github": {
                        "repository": repository, "event_name": event,
                    }}))

    def test_all_validation_jobs_share_policy_and_feed_result(self):
        gate = self.ci["ci-result"]
        jobs = set(self.ci) - {"ci-result"}
        self.assertEqual(set(gate["needs"]), jobs)
        self.assertEqual(
            " ".join(gate["if"].split()), f"always() && ( {self.condition} )"
        )
        for job in jobs:
            with self.subTest(job=job):
                self.assertEqual(self.ci[job]["if"], self.condition)
                self.assertFalse(self.ci[job].get("continue-on-error", False))
        for filename, job in (("docs.yml", "build-docs"),
                              ("package.yml", "build-distributions"),
                              ("notebooks.yml", "validate-notebooks")):
            workflow = yaml.safe_load((WORKFLOWS / filename).read_text())
            self.assertEqual(workflow["jobs"][job]["if"], self.condition)

    def test_release_builds_depend_on_the_repository_guard(self):
        release = yaml.safe_load((WORKFLOWS / "release.yml").read_text())["jobs"]
        self.assertEqual(release["validate-version"]["if"], self.condition)
        def ancestors(name):
            needs = release[name].get("needs", [])
            needs = [needs] if isinstance(needs, str) else needs
            return set(needs).union(*(ancestors(parent) for parent in needs))

        for name, job in release.items():
            if name == "validate-version":
                continue
            with self.subTest(job=name):
                self.assertIn("validate-version", ancestors(name))
                self.assertNotIn("always()", job.get("if", ""))

    def test_promotion_is_polled_and_dispatched_only_in_public(self):
        promote = yaml.safe_load((WORKFLOWS / "promote-public.yml").read_text())
        sync = yaml.safe_load((WORKFLOWS / "sync-private.yml").read_text())
        # PyYAML uses YAML 1.1, where an unquoted "on" key is a boolean.
        triggers = promote.get("on", promote.get(True))
        self.assertEqual(set(triggers), {"schedule", "repository_dispatch", "workflow_dispatch",
                                         "pull_request_target"})
        self.assertEqual(triggers["schedule"], [{"cron": "7,22,37,52 * * * *"}])
        self.assertEqual(triggers["repository_dispatch"]["types"], ["resume-promotion"])
        self.assertEqual(triggers["pull_request_target"], {
            "types": ["opened", "reopened", "synchronize"], "branches": ["main"],
        })
        expected = ("github.repository == 'wonderalexandre/MTLearn' && "
                    "vars.PROMOTION_ENABLED == 'true'")
        for workflow, job in ((promote, "promote"), (sync, "synchronize")):
            condition = " ".join(workflow["jobs"][job]["if"].split())
            if job == "promote":
                self.assertEqual(condition, expected + " && "
                    "(github.event_name != 'pull_request_target' || "
                    "(github.event.pull_request.head.repo.full_name == github.repository && "
                    "startsWith(github.event.pull_request.head.ref, 'promotion/private-pr-')))")
            else:
                self.assertEqual(condition, expected)
            checkout = next(step for step in workflow["jobs"][job]["steps"]
                            if step.get("uses", "").startswith("actions/checkout@"))
            self.assertEqual(checkout["with"]["ref"], "main")
            self.assertFalse(checkout["with"]["persist-credentials"])
        self.assertEqual(promote["concurrency"], sync["concurrency"])
        self.assertFalse(promote["concurrency"]["cancel-in-progress"])

    def test_aggregate_result_requires_every_job_to_succeed(self):
        gate = self.ci["ci-result"]
        script = gate["steps"][0]["run"]
        success = {job: {"result": "success"} for job in gate["needs"]}
        cases = [(success, True), ({}, False)]
        for result in ("failure", "cancelled", "skipped", "unknown", None):
            for job in gate["needs"]:
                jobs = copy.deepcopy(success)
                jobs[job]["result"] = result
                cases.append((jobs, False))
        malformed = copy.deepcopy(success)
        malformed[gate["needs"][0]] = {}
        cases.append((malformed, False))
        for jobs, expected_success in cases:
            with self.subTest(jobs=jobs):
                result = subprocess.run(
                    ["bash", "-e", "-o", "pipefail", "-c", script],
                    env={**os.environ, "CI_NEEDS": json.dumps(jobs)},
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode == 0, expected_success,
                                 result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
