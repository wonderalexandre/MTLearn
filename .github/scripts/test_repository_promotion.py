"""Exercise publication with real local Git repositories and an in-memory GitHub API."""

import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from repository_promotion import (
    Git, GitHub, PRIVATE, PUBLIC, PROMOTION_PREFIX, SYNC_BRANCH, PromotionError, MergeConflict,
    PUBLISHED_MARKER, command, main, metadata, promote, promotion_body, synchronize,
)


class LocalGit(Git):
    def __init__(self, root, remotes):
        super().__init__(root)
        self.remotes = remotes
        self.pushes = []

    def url(self, repo):
        return str(self.remotes[repo])

    def push(self, repo, branch, commit, previous=None):
        super().push(repo, branch, commit, previous)
        self.pushes.append((repo, branch, commit))


class FakeGitHub(GitHub):
    def __init__(self, git):
        self.git = git
        self.pulls = {PRIVATE: [], PUBLIC: []}
        self.sync_requests = 0
        self.queue_resumes = 0
        self.auto_merges = []

    def branch_tip(self, repo, branch):
        return self.git.text("--git-dir=" + self.git.url(repo), "rev-parse", branch)

    def api(self, path, method="GET", data=None):
        url = urlsplit(path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        parts = url.path.split("/")
        repo = "/".join(parts[1:3])
        resource = parts[3:]
        if resource[0] == "commits":
            rows = [p for p in self.pulls[repo] if p.get("merge_commit_sha") == resource[1]]
        elif len(resource) == 2:
            row = next(p for p in self.pulls[repo] if p["number"] == int(resource[1]))
            if method == "PATCH":
                row.update(data)
            if row["state"] == "open":
                row["head"]["sha"] = self.branch_tip(repo, row["head"]["ref"])
            return copy.deepcopy(row)
        elif method == "POST":
            number = len(self.pulls[repo]) + 1
            row = dict(
                number=number, title=data["title"], body=data["body"], state="open",
                merged_at=None, merge_commit_sha=None, auto_merge=None,
                head=dict(ref=data["head"], sha=self.branch_tip(repo, data["head"])),
                base=dict(ref=data["base"]), html_url=f"https://github.com/{repo}/pull/{number}",
            )
            self.pulls[repo].append(row)
            return copy.deepcopy(row)
        else:
            rows = self.pulls[repo]
            if query.get("state") not in (None, "all"):
                rows = [p for p in rows if p["state"] == query["state"]]
            if query.get("head"):
                rows = [p for p in rows if p["head"]["ref"] == query["head"].split(":", 1)[1]]
            if query.get("base"):
                rows = [p for p in rows if p["base"]["ref"] == query["base"]]
        start = (int(query.get("page", 1)) - 1) * int(query.get("per_page", 100))
        return copy.deepcopy(rows[start:start + int(query.get("per_page", 100))])

    def auto_merge(self, repo, pull, method):
        if not pull.get("auto_merge"):
            self.auto_merges.append((repo, pull["number"], method))
            next(p for p in self.pulls[repo] if p["number"] == pull["number"])["auto_merge"] = {
                "merge_method": method.upper(),
            }

    def request_sync(self):
        self.sync_requests += 1

    def resume_queue(self):
        self.queue_resumes += 1


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        remotes = {PUBLIC: self.root / "public.git", PRIVATE: self.root / "private.git"}
        for remote in remotes.values():
            command(["git", "init", "--bare", "--initial-branch=main", str(remote)])
        work = self.root / "work"
        command(["git", "init", "--initial-branch=main", str(work)])
        self.git = LocalGit(work, remotes)
        self.git.run("config", "user.name", "Promotion test")
        self.git.run("config", "user.email", "test@example.com")
        # Ignore machine-specific signing settings in real-commit tests.
        self.git.run("config", "commit.gpgsign", "false")
        self.initial = self.commit("value.txt", "zero\n", "initial")
        for repo in (PUBLIC, PRIVATE):
            self.git.push(repo, "main", self.initial)
        self.git.pushes.clear()
        self.github = FakeGitHub(self.git)

    def commit(self, filename, content, message):
        path = Path(self.git.root) / filename
        path.write_text(content)
        self.git.run("add", filename)
        self.git.run("commit", "-m", message)
        return self.git.text("rev-parse", "HEAD")

    def tip(self, repo):
        return self.github.branch_tip(repo, "main")

    def private_change(self, content, filename="value.txt"):
        base = self.tip(PRIVATE)
        self.git.checkout("private-work", base)
        source = self.commit(filename, content, "fix: private change")
        self.git.push(PRIVATE, "main", source, base)
        number = len(self.github.pulls[PRIVATE]) + 1
        row = dict(
            number=number, title=f"fix: private change {number}", body="Private review notes",
            state="closed", merged_at="2026-10-27T00:00:00Z", merge_commit_sha=source,
            head=dict(ref=f"feature/{number}", sha=source), base=dict(ref="main"),
        )
        self.github.pulls[PRIVATE].append(row)
        return row

    def external_change(self, filename="external.txt", content="public\n"):
        base = self.tip(PUBLIC)
        self.git.checkout("public-work", base)
        head = self.commit(filename, content, "fix: independent public change")
        self.git.push(PUBLIC, "main", head, base)
        return head

    def publish(self):
        return promote(self.git, self.github, self.tip(PRIVATE), self.tip(PUBLIC))

    def sync(self):
        return synchronize(self.git, self.github, self.tip(PRIVATE), self.tip(PUBLIC))

    def merge_pull(self, repo, number):
        pull = next(p for p in self.github.pulls[repo] if p["number"] == number)
        base = self.tip(repo)
        self.git.checkout("merge-work", base)
        head = self.github.branch_tip(repo, pull["head"]["ref"])
        if repo == PUBLIC:
            pull["head"]["sha"] = head
            self.git.run("--git-dir=" + self.git.url(repo), "update-ref",
                         f"refs/pull/{number}/head", head)
            self.git.run("merge", "--squash", head)
            self.git.run("commit", "-m", f"Public squash #{number}")
        else:
            self.git.run("merge", "--no-ff", head, "-m", f"Private return #{number}")
        commit = self.git.text("rev-parse", "HEAD")
        self.git.push(repo, "main", commit, base)
        pull.update(state="closed", merged_at="2026-10-27T01:00:00Z", merge_commit_sha=commit)
        if repo == PUBLIC:
            self.git.run("--git-dir=" + self.git.url(repo), "update-ref", "-d",
                         "refs/heads/" + pull["head"]["ref"])
        return commit

    def contents(self, commit, filename="value.txt"):
        return self.git.text("show", commit + ":" + filename)

    def correct_public_pr(self, content="corrected\n", filename="value.txt"):
        pull = self.github.pulls[PUBLIC][-1]
        old = self.github.branch_tip(PUBLIC, pull["head"]["ref"])
        self.git.checkout("public-correction", old)
        head = self.commit(filename, content, "Fix public CI")
        self.git.push(PUBLIC, pull["head"]["ref"], head, old)
        pull["head"]["sha"] = head
        return head

    def resolve_public_conflict(self, content):
        pull = self.github.pulls[PUBLIC][-1]
        old = self.github.branch_tip(PUBLIC, pull["head"]["ref"])
        self.git.checkout("resolve-public-conflict", old)
        self.git.run("merge", "--no-commit", self.tip(PUBLIC), check=False)
        head = self.commit("value.txt", content, "Resolve public main conflict")
        self.git.push(PUBLIC, pull["head"]["ref"], head, old)
        pull["head"]["sha"] = head
        return head

    def expect_public_conflict(self):
        with self.assertRaisesRegex(MergeConflict, "needs conflict resolution") as error:
            self.publish()
        return str(error.exception)

    def resolve_private_sync(self, content):
        old = self.tip(PRIVATE)
        self.git.checkout("manual-private-sync", old)
        result = self.git.run("merge", "--no-commit", self.tip(PUBLIC), check=False)
        self.assertEqual(result.returncode, 1)
        resolved = self.commit("value.txt", content, "Resolve private synchronization")
        self.git.push(PRIVATE, "main", resolved, old)
        return resolved

    def test_round_trip_and_reruns_do_not_publish_sync_pr(self):
        source = self.private_change("one\n")["merge_commit_sha"]
        self.publish()
        public_pr = self.github.pulls[PUBLIC][0]
        self.assertNotIn("Private review notes", public_pr["body"])
        self.assertEqual(metadata(public_pr)["source"], source)
        count = len(self.git.pushes)
        self.publish()
        self.assertEqual(len(self.git.pushes), count)
        self.assertEqual(len(self.github.auto_merges), 1)
        public_merge = self.merge_pull(PUBLIC, 1)
        self.sync()
        sync_pr = self.github.pulls[PRIVATE][-1]
        self.assertEqual(sync_pr["head"]["ref"], SYNC_BRANCH)
        self.assertEqual(self.github.auto_merges[-1][2], "merge")
        private_merge = self.merge_pull(PRIVATE, sync_pr["number"])
        self.assertTrue(self.git.ancestor(public_merge, private_merge))
        self.assertEqual(self.contents(private_merge), "one")
        self.assertIn("All merged", self.publish())
        self.sync()
        self.assertEqual(self.github.queue_resumes, 2)
        self.assertEqual(len(self.github.pulls[PUBLIC]), 1)

    def test_private_sync_resumes_the_next_publication_without_a_scheduled_run(self):
        self.private_change("one\n")
        second = self.private_change("two\n")
        self.publish()
        self.merge_pull(PUBLIC, 1)
        pending_resumes = []
        auto_merge = self.github.auto_merge

        def direct_private_merge(repo, pull, method):
            auto_merge(repo, pull, method)
            if repo == PRIVATE:
                self.assertEqual(pull["head"]["ref"], SYNC_BRANCH)
                self.assertEqual(method, "merge")
                self.merge_pull(repo, pull["number"])

        def resume_after_merge():
            # Model delivery of repository_dispatch after the current run exits.
            self.assertTrue(self.git.ancestor(self.tip(PUBLIC), self.tip(PRIVATE)))
            self.assertEqual(self.github.pulls[PRIVATE][-1]["state"], "closed")
            self.github.queue_resumes += 1
            pending_resumes.append("resume-promotion")

        with patch.object(self.github, "auto_merge", side_effect=direct_private_merge), \
                patch.object(self.github, "resume_queue", side_effect=resume_after_merge):
            self.assertIn("Synchronization completed", self.sync())
            self.assertEqual(pending_resumes, ["resume-promotion"])
            pending_resumes.pop()
            self.publish()
            self.assertEqual(metadata(self.github.pulls[PUBLIC][1])["source"],
                             second["merge_commit_sha"])
            self.merge_pull(PUBLIC, 2)
            self.sync()
            self.assertEqual(pending_resumes, ["resume-promotion"])
            pending_resumes.pop()
            self.assertIn("All merged", self.publish())
        self.assertEqual(self.github.queue_resumes, 2)
        self.assertEqual(len(self.github.pulls[PUBLIC]), 2)
        self.assertEqual(self.git.tree(self.tip(PRIVATE)), self.git.tree(self.tip(PUBLIC)))

    def test_failed_private_sync_merge_does_not_resume_the_queue(self):
        self.external_change()
        private = self.tip(PRIVATE)
        with patch.object(self.github, "auto_merge", side_effect=PromotionError("Merge failed")), \
                patch.object(self.github, "resume_queue") as resume:
            with self.assertRaisesRegex(PromotionError, "Merge failed"):
                self.sync()
            resume.assert_not_called()
        self.assertEqual(self.tip(PRIVATE), private)
        self.assertEqual(self.github.pulls[PRIVATE][-1]["state"], "open")

    def test_failed_resume_dispatch_is_retried_without_merging_sync_again(self):
        self.external_change()

        def direct_private_merge(repo, pull, method):
            self.assertEqual((repo, method), (PRIVATE, "merge"))
            self.merge_pull(repo, pull["number"])

        with patch.object(self.github, "auto_merge", side_effect=direct_private_merge), \
                patch.object(self.github, "resume_queue", side_effect=PromotionError("Dispatch failed")):
            with self.assertRaisesRegex(PromotionError, "Dispatch failed"):
                self.sync()
        self.assertTrue(self.git.ancestor(self.tip(PUBLIC), self.tip(PRIVATE)))
        before = len(self.git.pushes)
        with patch.object(self.github, "auto_merge") as merge:
            self.assertIn("already includes public main", self.sync())
            merge.assert_not_called()
        self.assertEqual(self.github.queue_resumes, 1)
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(len(self.github.pulls[PRIVATE]), 1)

    def test_later_private_work_survives_return_and_is_published_next(self):
        self.private_change("one\n")
        self.publish()
        later = self.private_change("two\n")["merge_commit_sha"]
        self.publish()
        self.assertEqual(len(self.github.pulls[PUBLIC]), 1)
        self.merge_pull(PUBLIC, 1)
        self.sync()
        returned = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(returned), "two")
        self.assertTrue(self.git.ancestor(later, returned))
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.publish()
        second = self.github.pulls[PUBLIC][-1]
        self.assertEqual(metadata(second)["source"], later)
        self.assertEqual(self.contents(second["head"]["sha"]), "two")

    def test_external_public_change_is_merged_without_replacing_the_promotion_origin(self):
        self.private_change("one\n")
        self.publish()
        old = self.github.pulls[PUBLIC][0]["head"]["sha"]
        public = self.external_change()
        self.publish()
        self.assertEqual(self.github.sync_requests, 0)
        self.sync()
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.publish()
        pull = self.github.pulls[PUBLIC][0]
        self.assertNotEqual(pull["head"]["sha"], old)
        self.assertEqual(metadata(pull)["public_base"], self.initial)
        self.assertTrue(self.git.ancestor(old, pull["head"]["sha"]))
        self.assertTrue(self.git.ancestor(public, pull["head"]["sha"]))
        self.assertEqual(self.contents(pull["head"]["sha"], "external.txt"), "public")
        self.merge_pull(PUBLIC, 1)
        self.sync()
        self.assertEqual(self.contents(self.github.branch_tip(PRIVATE, SYNC_BRANCH)), "one")

    def test_successive_returns_reuse_branch_but_create_distinct_prs(self):
        for index in range(3):
            self.external_change(filename=f"external-{index}.txt")
            self.sync()
            pull = self.github.pulls[PRIVATE][-1]
            self.merge_pull(PRIVATE, pull["number"])
        self.assertEqual(len(self.github.pulls[PRIVATE]), 3)
        self.assertTrue(self.git.ancestor(self.tip(PUBLIC), self.tip(PRIVATE)))

    def test_repeated_sync_keeps_the_open_pr_and_its_commit(self):
        self.external_change()
        self.sync()
        first = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        pushes = len(self.git.pushes)
        self.sync()
        self.assertEqual(self.github.branch_tip(PRIVATE, SYNC_BRANCH), first)
        self.assertEqual(len(self.github.pulls[PRIVATE]), 1)
        self.assertEqual(len(self.git.pushes), pushes)

    def test_delayed_return_integrates_public_followup_without_losing_private_work(self):
        self.private_change("one\n")
        self.publish()
        self.private_change("two\n")
        self.merge_pull(PUBLIC, 1)
        public = self.external_change()
        self.sync()
        head = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(head), "two")
        self.assertEqual(self.contents(head, "external.txt"), "public")
        self.assertTrue(self.git.ancestor(public, head))
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.publish()
        self.merge_pull(PUBLIC, 2)
        self.sync()
        head = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(head), "two")
        self.assertEqual(self.contents(head, "external.txt"), "public")

    def test_conflicting_external_change_does_not_push_partial_sync(self):
        self.private_change("private\n")
        self.external_change(filename="value.txt", content="external\n")
        before = len(self.git.pushes)
        with self.assertRaisesRegex(PromotionError, "conflict"):
            self.sync()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(self.contents(self.tip(PRIVATE)), "private")

    def test_initial_promotion_opens_before_conflicting_public_change_is_synced(self):
        self.private_change("private\n")
        external = self.external_change(filename="value.txt", content="external\n")
        message = self.expect_public_conflict()
        self.assertIn("needs conflict resolution", message)
        self.assertEqual(self.github.sync_requests, 0)
        self.assertEqual(self.github.auto_merges, [])
        pull = self.github.pulls[PUBLIC][0]
        origin = pull["head"]["sha"]
        self.assertEqual(metadata(pull)["public_base"], self.initial)
        self.assertEqual(self.contents(origin), "private")
        before = len(self.git.pushes)
        self.expect_public_conflict()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(len(self.github.pulls[PUBLIC]), 1)
        resolved = self.resolve_public_conflict("resolved\n")
        self.assertTrue(self.git.ancestor(origin, resolved))
        self.assertTrue(self.git.ancestor(external, resolved))
        self.publish()
        public = self.merge_pull(PUBLIC, 1)
        self.sync()
        returned = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(returned), "resolved")
        self.assertTrue(self.git.ancestor(public, returned))
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.assertIn("All merged", self.publish())

    def test_initial_promotion_merges_independent_public_advance(self):
        self.private_change("private\n")
        external = self.external_change()
        self.publish()
        pull = self.github.pulls[PUBLIC][0]
        record = metadata(pull)
        self.assertEqual(record["public_base"], self.initial)
        self.assertTrue(self.git.ancestor(record["promotion_commit"], pull["head"]["sha"]))
        self.assertTrue(self.git.ancestor(external, pull["head"]["sha"]))
        self.merge_pull(PUBLIC, 1)
        self.sync()
        returned = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(returned), "private")
        self.assertEqual(self.contents(returned, "external.txt"), "public")

    def test_next_promotion_waits_for_previous_publication_to_enter_private(self):
        self.private_change("one\n")
        self.publish()
        self.private_change("two\n")
        self.merge_pull(PUBLIC, 1)
        self.external_change()
        before = len(self.git.pushes)
        self.assertIn("Waiting for earlier public promotions", self.publish())
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(len(self.github.pulls[PUBLIC]), 1)
        self.assertEqual(self.github.sync_requests, 1)
        self.sync()
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.publish()
        self.assertEqual(len(self.github.pulls[PUBLIC]), 2)

    def test_recovery_of_initial_conflicted_promotion_after_pr_creation_failure(self):
        self.private_change("private\n")
        self.external_change(filename="value.txt", content="external\n")
        api = self.github.api

        def fail_creation(path, method="GET", data=None):
            if method == "POST" and path == f"repos/{PUBLIC}/pulls":
                raise PromotionError("PR creation failed")
            return api(path, method, data)

        with patch.object(self.github, "api", side_effect=fail_creation):
            with self.assertRaisesRegex(PromotionError, "PR creation failed"):
                self.publish()
        origin = self.github.branch_tip(PUBLIC, PROMOTION_PREFIX + "1")
        self.assertEqual(self.github.pulls[PUBLIC], [])
        before = len(self.git.pushes)
        self.expect_public_conflict()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(metadata(self.github.pulls[PUBLIC][0])["promotion_commit"], origin)

    def test_public_corrections_are_preserved_on_repeated_promotion(self):
        self.private_change("one\n")
        self.publish()
        pull = self.github.pulls[PUBLIC][0]
        old = pull["head"]["sha"]
        self.git.checkout("manual-work", old)
        changed = self.commit("unexpected.txt", "manual\n", "Manual public edit")
        self.git.push(PUBLIC, pull["head"]["ref"], changed, old)
        pull["head"]["sha"] = changed
        before = len(self.git.pushes)
        self.publish()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(self.github.branch_tip(PUBLIC, pull["head"]["ref"]), changed)

    def test_public_corrections_are_integrated_after_branch_deletion(self):
        self.private_change("one\n")
        self.publish()
        pull = self.github.pulls[PUBLIC][0]
        old = pull["head"]["sha"]
        self.git.checkout("tamper-work", old)
        changed = self.commit("unexpected.txt", "extra\n", "Extra change")
        self.git.push(PUBLIC, pull["head"]["ref"], changed, old)
        pull["head"]["sha"] = changed
        public = self.merge_pull(PUBLIC, 1)
        self.assertIsNone(self.git.remote_branch(PUBLIC, pull["head"]["ref"]))
        self.sync()
        returned = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(returned, "unexpected.txt"), "extra")
        self.assertTrue(self.git.ancestor(public, returned))

    def test_corrected_feature_keeps_independent_later_private_work(self):
        self.private_change("one\n")
        self.publish()
        later = self.private_change("private only\n", filename="private.txt")["merge_commit_sha"]
        self.correct_public_pr()
        public = self.merge_pull(PUBLIC, 1)
        self.sync()
        head = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(head), "corrected")
        self.assertEqual(self.contents(head, "private.txt"), "private only")
        self.assertTrue(self.git.ancestor(later, head))
        self.assertTrue(self.git.ancestor(public, head))
        before = len(self.git.pushes)
        self.sync()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(self.github.branch_tip(PRIVATE, SYNC_BRANCH), head)
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.publish()
        self.assertEqual(metadata(self.github.pulls[PUBLIC][-1])["source"], later)

    def test_correction_to_new_file_has_no_add_add_conflict(self):
        self.private_change("feature\n", filename="feature.txt")
        self.publish()
        self.correct_public_pr("fixed feature\n", filename="feature.txt")
        self.merge_pull(PUBLIC, 1)
        self.sync()
        self.assertEqual(self.contents(self.github.branch_tip(PRIVATE, SYNC_BRANCH),
                                       "feature.txt"), "fixed feature")

    def test_public_partial_revert_is_not_silently_discarded(self):
        source_pr = self.private_change("one\n")
        old = source_pr["merge_commit_sha"]
        self.commit("feature.txt", "keep feature\n", "More feature content")
        self.git.run("reset", "--soft", self.initial)
        self.git.run("commit", "-m", "Private feature squash")
        source = self.git.text("rev-parse", "HEAD")
        self.git.push(PRIVATE, "main", source, old)
        source_pr["merge_commit_sha"] = source
        source_pr["head"]["sha"] = source
        self.publish()
        self.correct_public_pr("zero\n")
        self.merge_pull(PUBLIC, 1)
        self.sync()
        head = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(head), "zero")
        self.assertEqual(self.contents(head, "feature.txt"), "keep feature")

    def test_public_correction_conflict_never_pushes_partial_sync(self):
        self.private_change("one\n")
        self.publish()
        private = self.private_change("later private\n")["merge_commit_sha"]
        self.correct_public_pr()
        self.merge_pull(PUBLIC, 1)
        before = len(self.git.pushes)
        with self.assertRaisesRegex(MergeConflict, "value.txt"):
            self.sync()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(self.tip(PRIVATE), private)
        self.assertIsNone(self.git.remote_branch(PRIVATE, SYNC_BRANCH))

    def test_corrected_promotion_uses_last_integrated_public_base(self):
        self.private_change("one\n")
        self.publish()
        fixed = self.correct_public_pr()
        public = self.external_change()
        self.sync()
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.private_change("private followup\n", filename="external.txt")
        self.publish()
        pull = self.github.pulls[PUBLIC][0]
        self.assertTrue(self.git.ancestor(fixed, pull["head"]["sha"]))
        self.assertTrue(self.git.ancestor(public, pull["head"]["sha"]))
        before = len(self.git.pushes)
        self.publish()
        self.assertEqual(len(self.git.pushes), before)
        self.merge_pull(PUBLIC, 1)
        self.sync()
        head = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(head), "corrected")
        self.assertEqual(self.contents(head, "external.txt"), "private followup")

    def publish_over_conflicting_public_change(self, *, private_overlap=False):
        self.private_change("one\n")
        self.publish()
        self.private_change("keep private work\n", filename="private.txt")
        if private_overlap:
            self.private_change("private overlap\n")
        self.correct_public_pr("two\n")
        self.correct_public_pr("CI correction\n", filename="ci.txt")
        external = self.external_change(filename="value.txt", content="two\n")
        self.publish()  # Must not wait for the conflicting public return.
        public = self.merge_pull(PUBLIC, 1)
        return external, public

    def test_promotion_integrates_an_earlier_conflicting_public_commit_together(self):
        external, public = self.publish_over_conflicting_public_change()
        self.sync()
        head = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(head), "two")
        self.assertEqual(self.contents(head, "private.txt"), "keep private work")
        self.assertEqual(self.contents(head, "ci.txt"), "CI correction")
        self.assertTrue(self.git.ancestor(external, head))
        self.assertTrue(self.git.ancestor(public, head))

    def test_grouped_return_handles_the_same_fix_already_present_in_private(self):
        external, public = self.publish_over_conflicting_public_change()
        fix = self.private_change("two\n")
        self.sync()
        head = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(head), "two")
        self.assertTrue(self.git.ancestor(public, head))
        self.assertTrue(self.git.ancestor(external, head))
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        # Publish the independent earlier private PR first, then reach the fix.
        self.publish()
        self.merge_pull(PUBLIC, 2)
        self.sync()
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.assertIn("All merged", self.publish())
        self.assertIsNotNone(PUBLISHED_MARKER.search(fix["body"]))

    def test_grouped_return_still_rejects_a_real_private_conflict(self):
        self.publish_over_conflicting_public_change(private_overlap=True)
        before = len(self.git.pushes)
        with self.assertRaisesRegex(MergeConflict, "value.txt"):
            self.sync()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(self.contents(self.tip(PRIVATE)), "private overlap")

    def test_invalid_origin_cannot_resolve_a_blocked_public_return(self):
        self.publish_over_conflicting_public_change()
        pull = self.github.pulls[PUBLIC][0]
        record = metadata(pull)
        record["tree"] = self.git.tree(self.initial)
        pull["body"] = promotion_body(record)
        before = len(self.git.pushes)
        with self.assertRaisesRegex(PromotionError, "valid promotion origin"):
            self.sync()
        self.assertEqual(len(self.git.pushes), before)

    def test_conflicting_branch_update_preserves_the_correction(self):
        self.private_change("one\n")
        self.publish()
        fixed = self.correct_public_pr()
        self.external_change(filename="value.txt", content="other public change\n")
        before = len(self.git.pushes)
        self.expect_public_conflict()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(self.github.branch_tip(PUBLIC, PROMOTION_PREFIX + "1"), fixed)

    def test_public_update_lease_preserves_a_concurrent_correction(self):
        self.private_change("one\n")
        self.publish()
        self.correct_public_pr()
        self.external_change()
        push = self.git.push
        concurrent = None

        def concurrent_push(repo, branch, candidate, previous=None):
            nonlocal concurrent
            self.git.checkout("concurrent-correction", previous)
            concurrent = self.commit("concurrent.txt", "keep me\n", "Concurrent correction")
            push(repo, branch, concurrent, previous)
            push(repo, branch, candidate, previous)

        with patch.object(self.git, "push", side_effect=concurrent_push):
            with self.assertRaises(PromotionError):
                self.publish()
        self.assertEqual(self.github.branch_tip(PUBLIC, PROMOTION_PREFIX + "1"), concurrent)
        self.assertEqual(self.contents(concurrent, "concurrent.txt"), "keep me")

    def test_rewritten_promotion_origin_is_not_accepted(self):
        self.private_change("one\n")
        self.publish()
        pull = self.github.pulls[PUBLIC][0]
        old = pull["head"]["sha"]
        self.git.checkout("replacement", self.initial)
        replacement = self.commit("value.txt", "unrelated\n", "Replace promotion")
        self.git.push(PUBLIC, pull["head"]["ref"], replacement, old)
        pull["head"]["sha"] = replacement
        before = len(self.git.pushes)
        with self.assertRaisesRegex(PromotionError, "valid promotion origin"):
            self.publish()
        self.assertEqual(len(self.git.pushes), before)

    def test_squash_must_match_retained_pr_head(self):
        self.private_change("one\n")
        self.publish()
        public = self.merge_pull(PUBLIC, 1)
        self.git.checkout("forged-squash", self.initial)
        forged = self.commit("value.txt", "unreviewed\n", "Forged squash")
        self.git.push(PUBLIC, "main", forged, public)
        self.github.pulls[PUBLIC][0]["merge_commit_sha"] = forged
        before = len(self.git.pushes)
        with self.assertRaisesRegex(PromotionError, "does not match the merged PR head"):
            self.sync()
        self.assertEqual(len(self.git.pushes), before)

    def test_old_metadata_without_original_sha_still_accepts_corrections(self):
        self.private_change("one\n")
        self.publish()
        pull = self.github.pulls[PUBLIC][0]
        record = metadata(pull)
        del record["promotion_commit"]
        pull["body"] = promotion_body(record)
        self.correct_public_pr()
        self.merge_pull(PUBLIC, 1)
        self.sync()
        self.assertEqual(self.contents(self.github.branch_tip(PRIVATE, SYNC_BRANCH)), "corrected")

    def test_already_published_fix_is_recorded_and_queue_continues(self):
        self.private_change("one\n")
        self.publish()
        fix = self.private_change("corrected\n")
        self.correct_public_pr()
        public = self.merge_pull(PUBLIC, 1)
        self.sync()
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.assertIn("All merged", self.publish())
        saved = fix["body"]
        receipt = json.loads(PUBLISHED_MARKER.search(saved)[1])
        self.assertEqual(receipt["source"], fix["merge_commit_sha"])
        self.assertEqual(receipt["public_commit"], public)
        self.assertIn("Private review notes", saved)
        before = len(self.git.pushes)
        self.publish()
        self.assertEqual(fix["body"], saved)
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(len(self.github.pulls[PUBLIC]), 1)
        later = self.private_change("next feature\n", filename="next.txt")
        self.publish()
        self.assertEqual(metadata(self.github.pulls[PUBLIC][-1])["source"], later["merge_commit_sha"])

    def test_false_already_published_receipt_cannot_skip_a_change(self):
        source_pr = self.private_change("one\n")
        record = dict(private_pr=1, source=source_pr["merge_commit_sha"],
                      public_commit=self.initial, tree=self.git.tree(self.initial))
        source_pr["body"] = "<!-- mtlearn-already-published: " + json.dumps(record) + " -->"
        before = len(self.git.pushes)
        with self.assertRaisesRegex(PromotionError, "Invalid already-published record"):
            self.publish()
        self.assertEqual(len(self.git.pushes), before)

    def test_already_published_record_survives_a_lost_api_response(self):
        self.private_change("one\n")
        self.publish()
        fix = self.private_change("corrected\n")
        self.correct_public_pr()
        self.merge_pull(PUBLIC, 1)
        self.sync()
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        api = self.github.api

        def lost_response(path, method="GET", data=None):
            result = api(path, method, data)
            if method == "PATCH" and path == f"repos/{PRIVATE}/pulls/{fix['number']}":
                raise PromotionError("Lost API response")
            return result

        with patch.object(self.github, "api", side_effect=lost_response):
            with self.assertRaisesRegex(PromotionError, "Lost API response"):
                self.publish()
        self.assertIn("All merged", self.publish())
        self.assertEqual(len(PUBLISHED_MARKER.findall(fix["body"])), 1)
        self.assertEqual(len(self.github.pulls[PUBLIC]), 1)

    def test_conflicting_publication_never_creates_an_already_published_receipt(self):
        source_pr = self.private_change("one\n")
        source = source_pr["merge_commit_sha"]
        public = self.external_change(filename="value.txt", content="public\n")
        resolved = self.git.commit_merge(source, public, self.git.tree(source), "Manual return")
        self.git.push(PRIVATE, "main", resolved, source)
        self.expect_public_conflict()
        pull = self.github.pulls[PUBLIC][0]
        self.assertEqual(metadata(pull)["public_base"], self.initial)
        self.assertEqual(self.contents(pull["head"]["sha"]), "one")
        self.assertNotIn("mtlearn-already-published", source_pr["body"])

    def test_equal_private_public_trees_after_manual_sync_do_not_skip_source_review(self):
        source_pr = self.private_change("private\n")
        source = source_pr["merge_commit_sha"]
        public = self.external_change(filename="value.txt", content="external\n")
        resolved = self.git.commit_merge(source, public, self.git.tree(public), "Accept public version")
        self.git.push(PRIVATE, "main", resolved, source)
        self.assertEqual(self.git.tree(resolved), self.git.tree(public))
        self.expect_public_conflict()
        self.assertEqual(len(self.github.pulls[PUBLIC]), 1)
        self.assertNotIn("mtlearn-already-published", source_pr["body"])
        self.assertEqual(self.contents(self.github.pulls[PUBLIC][0]["head"]["sha"]), "private")

    def test_matching_private_and_public_resolutions_sync_without_another_manual_merge(self):
        self.private_change("private\n")
        self.external_change(filename="value.txt", content="external\n")
        self.expect_public_conflict()
        self.resolve_private_sync("resolved\n")
        later = self.private_change("keep private work\n", filename="private.txt")
        self.resolve_public_conflict("resolved\n")
        self.publish()
        public = self.merge_pull(PUBLIC, 1)
        self.sync()
        returned = self.github.branch_tip(PRIVATE, SYNC_BRANCH)
        self.assertEqual(self.contents(returned), "resolved")
        self.assertEqual(self.contents(returned, "private.txt"), "keep private work")
        self.assertTrue(self.git.ancestor(public, returned))
        self.assertTrue(self.git.ancestor(later["merge_commit_sha"], returned))
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.publish()
        self.merge_pull(PUBLIC, 2)
        self.sync()
        self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
        self.assertEqual(self.git.tree(self.tip(PRIVATE)), self.git.tree(self.tip(PUBLIC)))
        self.assertIn("All merged", self.publish())

    def test_different_private_and_public_resolutions_remain_a_conflict(self):
        self.private_change("private\n")
        self.external_change(filename="value.txt", content="external\n")
        self.expect_public_conflict()
        private = self.resolve_private_sync("private resolution\n")
        self.resolve_public_conflict("public resolution\n")
        self.publish()
        self.merge_pull(PUBLIC, 1)
        before = len(self.git.pushes)
        with self.assertRaisesRegex(MergeConflict, "prior public changes still differ.*value.txt"):
            self.sync()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(self.tip(PRIVATE), private)

    def test_historical_fallback_cannot_silently_discard_a_public_revert(self):
        old = self.initial
        self.initial = self.commit(" setting.txt", "zero\n", "Shared setting")
        for repo in (PUBLIC, PRIVATE):
            self.git.push(repo, "main", self.initial, old)
        self.private_change("private\n")
        self.external_change(filename="value.txt", content="external\n")
        self.external_change(filename=" setting.txt", content="one\n")
        self.expect_public_conflict()
        self.resolve_private_sync("resolved\n")
        self.resolve_public_conflict("resolved\n")
        self.correct_public_pr("zero\n", filename=" setting.txt")
        self.publish()
        public = self.merge_pull(PUBLIC, 1)
        record = metadata(self.github.pulls[PUBLIC][0])
        unsafe_tree = self.git.merge_tree(self.tip(PRIVATE), public, base=record["tree"])
        self.assertEqual(self.contents(unsafe_tree, " setting.txt"), "one")
        self.assertEqual(self.contents(public, " setting.txt"), "zero")
        before = len(self.git.pushes)
        with self.assertRaisesRegex(MergeConflict, "prior public changes still differ.*setting.txt"):
            self.sync()
        self.assertEqual(len(self.git.pushes), before)

    def test_reconstruction_git_error_does_not_use_the_historical_fallback(self):
        self.private_change("private\n")
        self.external_change(filename="value.txt", content="external\n")
        self.expect_public_conflict()
        self.resolve_private_sync("resolved\n")
        self.resolve_public_conflict("resolved\n")
        self.publish()
        self.merge_pull(PUBLIC, 1)
        merge_tree = self.git.merge_tree

        def fail_reconstruction(*args, **kwargs):
            if kwargs.get("context") == "Expected promotion reconstruction":
                raise PromotionError("Git failed (exit 128)")
            return merge_tree(*args, **kwargs)

        before = len(self.git.pushes)
        with patch.object(self.git, "merge_tree", side_effect=fail_reconstruction):
            with self.assertRaisesRegex(PromotionError, "exit 128"):
                self.sync()
        self.assertEqual(len(self.git.pushes), before)

    def test_empty_public_resolution_requires_review_without_enabling_auto_merge(self):
        source_pr = self.private_change("private\n")
        self.external_change(filename="value.txt", content="external\n")
        self.expect_public_conflict()
        self.resolve_private_sync("external\n")
        self.resolve_public_conflict("external\n")
        with self.assertRaisesRegex(PromotionError, "no remaining content changes"):
            self.publish()
        self.assertEqual(self.github.auto_merges, [])
        self.assertNotIn("mtlearn-already-published", source_pr["body"])
        self.assertEqual(self.github.pulls[PUBLIC][0]["state"], "open")

    def test_historical_base_never_omits_an_earlier_publication(self):
        self.private_change("first feature\n", filename="first.txt")
        self.publish()
        source_pr = self.private_change("private\n")
        source = source_pr["merge_commit_sha"]
        self.correct_public_pr("external\n")
        public = self.merge_pull(PUBLIC, 1)
        resolved = self.git.commit_merge(source, public, self.git.tree(source), "Manual resolution")
        self.git.push(PRIVATE, "main", resolved, source)
        before = len(self.git.pushes)
        with self.assertRaisesRegex(PromotionError, "containing all earlier publications"):
            self.publish()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(len(self.github.pulls[PUBLIC]), 1)

    def test_git_error_in_publication_comparison_does_not_create_a_branch(self):
        self.private_change("private\n")
        before = len(self.git.pushes)
        with patch.object(self.git, "merge_tree", side_effect=PromotionError("Git failed (exit 128)")):
            with self.assertRaisesRegex(PromotionError, "exit 128"):
                self.publish()
        self.assertEqual(len(self.git.pushes), before)
        self.assertEqual(self.github.pulls[PUBLIC], [])

    def test_closed_public_pr_blocks_later_publications(self):
        self.private_change("one\n")
        self.publish()
        self.github.pulls[PUBLIC][0]["state"] = "closed"
        self.private_change("two\n")
        with self.assertRaisesRegex(PromotionError, "closed without merging"):
            self.publish()
        self.assertEqual(len(self.github.pulls[PUBLIC]), 1)

    def test_run_interrupted_after_push_reuses_the_existing_branch(self):
        self.private_change("one\n")
        self.publish()
        head = self.github.pulls[PUBLIC][0]["head"]["sha"]
        self.github.pulls[PUBLIC].clear()
        count = len(self.git.pushes)
        self.publish()
        self.assertEqual(len(self.git.pushes), count)
        self.assertEqual(self.github.pulls[PUBLIC][0]["head"]["sha"], head)

    def test_interrupted_creation_recovers_corrections_after_public_main_advances(self):
        self.private_change("one\n")
        self.publish()
        original = self.github.pulls[PUBLIC][0]["head"]["sha"]
        fixed = self.correct_public_pr()
        self.github.pulls[PUBLIC].clear()
        public = self.external_change()
        self.publish()
        pull = self.github.pulls[PUBLIC][0]
        record = metadata(pull)
        self.assertEqual(record["promotion_commit"], original)
        self.assertEqual(record["public_base"], self.initial)
        self.assertTrue(self.git.ancestor(fixed, pull["head"]["sha"]))
        self.assertTrue(self.git.ancestor(public, pull["head"]["sha"]))
        self.assertEqual(self.contents(pull["head"]["sha"]), "corrected")
        self.merge_pull(PUBLIC, 1)
        self.sync()
        self.assertEqual(self.contents(self.github.branch_tip(PRIVATE, SYNC_BRANCH)), "corrected")

    def test_unmerged_private_branch_is_not_exposed(self):
        base = self.tip(PRIVATE)
        self.git.checkout("unmerged", base)
        private = self.commit("draft.txt", "internal\n", "Unmerged draft")
        self.git.push(PRIVATE, "feature/draft", private)
        self.assertIn("All merged", self.publish())
        self.assertEqual(self.github.pulls[PUBLIC], [])
        self.assertEqual(self.tip(PUBLIC), self.initial)

    def private_merge(self):
        base = self.tip(PRIVATE)
        self.git.checkout("private-feature", base)
        self.commit("value.txt", "feature\n", "First reviewed commit")
        head = self.commit("feature.txt", "complete\n", "Second reviewed commit")
        self.git.checkout("private-merge", base)
        self.git.run("merge", "--no-ff", head, "-m", "Approved private merge")
        source = self.git.text("rev-parse", "HEAD")
        self.git.push(PRIVATE, "main", source, base)
        row = dict(number=1, title="Reviewed feature", body="Private review notes",
                   state="closed", merged_at="now", merge_commit_sha=source,
                   head=dict(ref="feature", sha=head), base=dict(ref="main"))
        self.github.pulls[PRIVATE].append(row)
        return row, {source: (1, base, head)}

    def test_allowlisted_merge_publishes_as_one_commit(self):
        source_pr, recovery = self.private_merge()
        source = source_pr["merge_commit_sha"]
        with patch.dict("repository_promotion.APPROVED_MERGE_RECOVERIES", recovery, clear=True):
            self.assertIn("Publication pending", self.publish())
        pull = self.github.pulls[PUBLIC][0]
        record = metadata(pull)
        self.assertEqual(record["source"], source)
        self.assertEqual(len(self.git.parents(record["promotion_commit"])), 1)
        self.assertEqual(self.git.tree(record["promotion_commit"]), self.git.tree(source))
        self.assertIn("(cherry picked from commit " + source + ")",
                      self.git.text("show", "-s", "--format=%B", record["promotion_commit"]))
        self.assertEqual(self.tip(PRIVATE), source)
        self.assertEqual(len(self.git.parents(source)), 2)

    def test_allowlisted_merge_roundtrip_preserves_history_and_resumes_queue(self):
        source_pr, recovery = self.private_merge()
        source = source_pr["merge_commit_sha"]
        with patch.dict("repository_promotion.APPROVED_MERGE_RECOVERIES", recovery, clear=True):
            self.publish()
            public = self.merge_pull(PUBLIC, 1)
            self.sync()
            self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
            self.assertTrue(self.git.ancestor(source, self.tip(PRIVATE)))
            self.assertTrue(self.git.ancestor(public, self.tip(PRIVATE)))
            self.assertGreater(self.github.queue_resumes, 0)
            self.assertIn("All merged", self.publish())
            later = self.private_change("next\n", filename="next.txt")
            self.assertIn("Publication pending", self.publish())
            self.assertEqual(metadata(self.github.pulls[PUBLIC][-1])["source"],
                             later["merge_commit_sha"])

    def test_allowlisted_merge_keeps_public_review_corrections(self):
        source_pr, recovery = self.private_merge()
        with patch.dict("repository_promotion.APPROVED_MERGE_RECOVERIES", recovery, clear=True):
            self.publish()
            self.correct_public_pr("corrected by public review\n")
            self.merge_pull(PUBLIC, 1)
            self.sync()
            self.merge_pull(PRIVATE, self.github.pulls[PRIVATE][-1]["number"])
            self.assertEqual(self.contents(self.tip(PRIVATE)), "corrected by public review")
            self.assertEqual(self.contents(self.tip(PRIVATE), "feature.txt"), "complete")
            self.assertTrue(self.git.ancestor(source_pr["merge_commit_sha"], self.tip(PRIVATE)))

    def test_allowlisted_merge_can_record_and_revalidate_existing_publication(self):
        source_pr, recovery = self.private_merge()
        source = source_pr["merge_commit_sha"]
        public = self.git.text("commit-tree", self.git.tree(source), "-p", self.tip(PUBLIC),
                               "-m", "Previously reviewed public change")
        self.git.push(PUBLIC, "main", public, self.tip(PUBLIC))
        with patch.dict("repository_promotion.APPROVED_MERGE_RECOVERIES", recovery, clear=True):
            self.assertIn("All merged", self.publish())
            self.assertIn("mtlearn-already-published", self.github.pulls[PRIVATE][0]["body"])
            self.assertIn("All merged", self.publish())
        self.assertFalse(self.github.pulls[PUBLIC])

    def test_allowlisted_merge_recovery_is_idempotent(self):
        source_pr, recovery = self.private_merge()
        with patch.dict("repository_promotion.APPROVED_MERGE_RECOVERIES", recovery, clear=True):
            self.publish()
            head = self.github.branch_tip(PUBLIC, PROMOTION_PREFIX + "1")
            pushes = len(self.git.pushes)
            self.publish()
            self.assertEqual(self.github.branch_tip(PUBLIC, PROMOTION_PREFIX + "1"), head)
            self.assertEqual(len(self.git.pushes), pushes)
            self.assertEqual(len(self.github.pulls[PUBLIC]), 1)

    def test_merge_recovery_rejects_wrong_private_pr_number(self):
        source_pr, recovery = self.private_merge()
        source_pr["number"] = 2
        with patch.dict("repository_promotion.APPROVED_MERGE_RECOVERIES", recovery, clear=True):
            with self.assertRaisesRegex(PromotionError, "must use squash"):
                self.publish()
        self.assertFalse(self.github.pulls[PUBLIC])

    def test_merge_recovery_rejects_wrong_parent_identity(self):
        source_pr, recovery = self.private_merge()
        source = source_pr["merge_commit_sha"]
        recovery[source] = (1, self.initial, self.initial)
        with patch.dict("repository_promotion.APPROVED_MERGE_RECOVERIES", recovery, clear=True):
            with self.assertRaisesRegex(PromotionError, "must use squash"):
                self.publish()
        self.assertFalse(self.github.pulls[PUBLIC])

    def test_merge_recovery_rejects_unreviewed_merge_tree(self):
        source_pr, recovery = self.private_merge()
        source = source_pr["merge_commit_sha"]
        extra = self.commit("unreviewed.txt", "not approved\n", "Unreviewed change")
        _, base, head = recovery[source]
        forged = self.git.commit_merge(base, head, self.git.tree(extra), "Forged merge")
        self.git.push(PRIVATE, "main", forged, source)
        source_pr["merge_commit_sha"] = forged
        with patch.dict("repository_promotion.APPROVED_MERGE_RECOVERIES",
                        {forged: (1, base, head)}, clear=True):
            with self.assertRaisesRegex(PromotionError, "must use squash"):
                self.publish()
        self.assertFalse(self.github.pulls[PUBLIC])

    def test_merge_roundtrip_requires_the_explicit_recovery(self):
        source_pr, recovery = self.private_merge()
        with patch.dict("repository_promotion.APPROVED_MERGE_RECOVERIES", recovery, clear=True):
            self.publish()
            self.merge_pull(PUBLIC, 1)
        with patch.dict("repository_promotion.APPROVED_MERGE_RECOVERIES", {}, clear=True):
            with self.assertRaisesRegex(PromotionError, "does not match approved private history"):
                self.sync()

    def test_non_squash_private_merge_requires_review(self):
        base = self.tip(PRIVATE)
        self.git.checkout("feature", base)
        feature = self.commit("value.txt", "one\n", "Feature")
        self.git.checkout("merge-private", base)
        self.git.run("merge", "--no-ff", feature, "-m", "Private merge")
        merge = self.git.text("rev-parse", "HEAD")
        self.git.push(PRIVATE, "main", merge, base)
        self.github.pulls[PRIVATE].append(dict(
            number=1, title="feature", state="closed", merged_at="now",
            merge_commit_sha=merge, head=dict(ref="feature"), base=dict(ref="main"),
        ))
        with self.assertRaisesRegex(PromotionError, "must use squash"):
            self.publish()

    def test_lease_preserves_a_concurrent_remote_update(self):
        base = self.tip(PRIVATE)
        later = self.private_change("one\n")["merge_commit_sha"]
        with self.assertRaises(PromotionError):
            self.git.push(PRIVATE, "main", base, base)
        self.assertEqual(self.tip(PRIVATE), later)

    def test_queue_order_follows_history_instead_of_pr_creation_number(self):
        first = self.private_change("one\n")
        first["number"] = 9
        second = self.private_change("two\n")
        second["number"] = 1
        self.github.pulls[PRIVATE].reverse()
        self.publish()
        self.assertEqual(self.github.pulls[PUBLIC][0]["head"]["ref"], PROMOTION_PREFIX + "9")


class MetadataTests(unittest.TestCase):
    def test_rejects_missing_invalid_or_option_like_object_ids(self):
        for body in ("", "<!-- mtlearn-promotion: {invalid} -->",
                     promotion_body(dict(private_pr=1, source="--help",
                                         public_base="a" * 40, tree="b" * 40))):
            with self.subTest(body=body), self.assertRaises(PromotionError):
                metadata(dict(body=body))

    def test_lists_more_than_one_page(self):
        class PagedGitHub(GitHub):
            def api(self, path, method="GET", data=None):
                page = int(parse_qs(urlsplit(path).query)["page"][0])
                return list(range((page - 1) * 100, min(page * 100, 207)))
        self.assertEqual(PagedGitHub().pages("repos/test/example/pulls"), list(range(207)))


class MergeTreeTests(unittest.TestCase):
    def test_conflict_tree_is_never_used_as_a_successful_result(self):
        git = Git()
        result = subprocess.CompletedProcess([], 1, "a" * 40 + "\nCONFLICT: file.txt\n", "")
        with patch.object(git, "run", return_value=result):
            with self.assertRaisesRegex(MergeConflict, "file.txt"):
                git.merge_tree("left", "right", context="Expected promotion reconstruction")

    def test_git_errors_are_not_classified_as_content_conflicts(self):
        git = Git()
        result = subprocess.CompletedProcess([], 128, "", "fatal: missing object")
        with patch.object(git, "run", return_value=result):
            with self.assertRaisesRegex(PromotionError, "exit 128") as error:
                git.merge_tree("left", "right")
        self.assertNotIsInstance(error.exception, MergeConflict)


class PublicControllerTests(unittest.TestCase):
    def test_conflict_fails_the_controller_and_records_the_pr_link_in_the_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.md"
            message = "Public PR #1 needs conflict resolution: https://github.com/example/repo/pull/1"
            with patch.dict(os.environ, {"GITHUB_REPOSITORY": PUBLIC,
                                         "GITHUB_STEP_SUMMARY": str(summary)}), \
                    patch("sys.argv", ["repository_promotion.py", "promote"]), \
                    patch("repository_promotion.Git") as git_type, \
                    patch("repository_promotion.GitHub"), \
                    patch("repository_promotion.promote", side_effect=MergeConflict(message)):
                git = git_type.return_value
                git.text.return_value = ""
                git.fetch_main.side_effect = ["private-tip", "public-tip"]
                with self.assertRaisesRegex(MergeConflict, "needs conflict resolution"):
                    main()
            self.assertEqual(summary.read_text(), message + "\n")

    def test_public_merge_waits_for_checks(self):
        pull = dict(number=1, head=dict(ref=PROMOTION_PREFIX + "4", sha="a" * 40))
        with patch("repository_promotion.command") as run:
            GitHub().auto_merge(PUBLIC, pull, "squash")
            self.assertEqual(run.call_args.args[0], [
                "gh", "pr", "merge", "1", "--repo", PUBLIC, "--squash",
                "--match-head-commit", "a" * 40, "--auto",
            ])
            run.reset_mock()
            pull["auto_merge"] = {"merge_method": "SQUASH"}
            GitHub().auto_merge(PUBLIC, pull, "squash")
            run.assert_not_called()

    def test_private_sync_merges_without_enabling_auto_merge(self):
        pull = dict(number=5, head=dict(ref=SYNC_BRANCH, sha="b" * 40))
        with patch("repository_promotion.command") as run:
            GitHub().auto_merge(PRIVATE, pull, "merge")
            self.assertEqual(run.call_args.args[0], [
                "gh", "pr", "merge", "5", "--repo", PRIVATE, "--merge",
                "--match-head-commit", "b" * 40,
            ])

    def test_direct_merge_is_restricted_to_private_sync(self):
        for repo, branch, method in ((PRIVATE, "feature", "merge"),
                                     (PRIVATE, SYNC_BRANCH, "squash"),
                                     ("other/repository", SYNC_BRANCH, "merge")):
            with self.subTest(repo=repo, branch=branch, method=method), \
                    patch("repository_promotion.command") as run:
                pull = dict(number=1, head=dict(ref=branch, sha="c" * 40))
                with self.assertRaisesRegex(PromotionError, "Only the reserved"):
                    GitHub().auto_merge(repo, pull, method)
                run.assert_not_called()

    def test_all_dispatches_target_public_actions(self):
        class RecordingGitHub(GitHub):
            def __init__(self):
                self.calls = []

            def api(self, path, method="GET", data=None):
                self.calls.append((path, method, data))

        github = RecordingGitHub()
        github.resume_queue()
        github.request_sync()
        self.assertEqual(github.calls, [
            (f"repos/{PUBLIC}/dispatches", "POST", {"event_type": "resume-promotion"}),
            (f"repos/{PUBLIC}/dispatches", "POST", {"event_type": "sync-private"}),
        ])

    def test_private_execution_is_rejected_before_accessing_git(self):
        for mode in ("promote", "sync"):
            with self.subTest(mode=mode), \
                    patch.dict(os.environ, {"GITHUB_REPOSITORY": PRIVATE}), \
                    patch("sys.argv", ["repository_promotion.py", mode]), \
                    patch("repository_promotion.Git") as git:
                with self.assertRaisesRegex(PromotionError, "must run in the public"):
                    main()
                git.assert_not_called()

    def test_both_operations_execute_from_public(self):
        for mode, operation in (("promote", "promote"), ("sync", "synchronize")):
            with self.subTest(mode=mode), \
                    patch.dict(os.environ, {"GITHUB_REPOSITORY": PUBLIC,
                                            "GITHUB_STEP_SUMMARY": ""}), \
                    patch("sys.argv", ["repository_promotion.py", mode]), \
                    patch("repository_promotion.Git") as git_type, \
                    patch("repository_promotion.GitHub") as github_type, \
                    patch("repository_promotion." + operation, return_value="finished") as run, \
                    redirect_stdout(io.StringIO()):
                git = git_type.return_value
                git.text.return_value = ""
                git.fetch_main.side_effect = ["private-tip", "public-tip"]
                main()
                run.assert_called_once_with(git, github_type.return_value,
                                            "private-tip", "public-tip")


if __name__ == "__main__":
    unittest.main()
