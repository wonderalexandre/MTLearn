#!/usr/bin/env python3
"""Publish approved private PRs and reconcile the two repository histories.

Operational limits:
- Until scheduled executions are verified, start an idle publication queue after
  merging private work with:
  gh workflow run promote-public.yml --repo wonderalexandre/MTLearn
  A successful private synchronization resumes the remaining queue automatically.
- Closing a publication PR without merging blocks later publications. Reopen it
  to continue; closing it does not cancel the change already in private main.
- Development PRs require squash merges; synchronization PRs require merge commits.
- If conflict resolution leaves a public PR with no content changes, completion
  requires manual review. Do not assume GitHub can squash an empty PR or silently
  treat the original private change as published. Cancellation is not automated.
- A conflicting expected tree can use the original promotion tree only when both
  sides agree on every path changed by already-integrated public commits. Other
  cases require manual synchronization; an unguarded historical base can lose fixes.
- Public main supplies the running controller. The PR introducing a new version
  still uses the previous controller until merged. A blocked rollout can require
  a separate public maintenance PR, with public CI and normal review.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlencode


PUBLIC = "wonderalexandre/MTLearn"
PRIVATE = "wonderalexandre/MTLearn-dev"
SYNC_BRANCH = "sync/public-main"
PROMOTION_PREFIX = "promotion/private-pr-"
MARKER = re.compile(r"<!-- mtlearn-promotion: (\{[^\n]+\}) -->")
PUBLISHED_MARKER = re.compile(r"<!-- mtlearn-already-published: (\{[^\n]+\}) -->")
APPROVED_MERGE_RECOVERIES = {
    "5fa238708ab0a903d1c4e8a9e80520ba1a91ade6": (
        8,
        "47e9d95b98e84034c3cda2dc34ea297144a01b7b",
        "fa09ef45af13afee38c07512441d72d372ba3c28",
    ),
}


class PromotionError(RuntimeError):
    pass


class MergeConflict(PromotionError):
    pass


def command(args, *, cwd=None, data=None, check=True):
    result = subprocess.run(args, cwd=cwd, input=data, text=True, capture_output=True)
    if check and result.returncode:
        raise PromotionError(result.stderr.strip() or result.stdout.strip())
    return result


class GitHub:
    def api(self, path, method="GET", data=None):
        args = ["gh", "api", "--method", method, path]
        if data is not None:
            args += ["--input", "-"]
        result = command(args, data=json.dumps(data) if data is not None else None)
        return json.loads(result.stdout) if result.stdout.strip() else None

    def pages(self, path, **query):
        rows = []
        for page in range(1, 10001):
            batch = self.api(path + "?" + urlencode(dict(query, per_page=100, page=page)))
            rows.extend(batch)
            if len(batch) < 100:
                return rows
        raise PromotionError("Pagination limit exceeded.")

    def pull(self, repo, number):
        return self.api(f"repos/{repo}/pulls/{number}")

    def find_pull(self, repo, branch, state="all"):
        rows = self.pages(
            f"repos/{repo}/pulls", state=state, base="main",
            head=f"{repo.split('/')[0]}:{branch}",
        )
        if len(rows) > 1:
            raise PromotionError(f"Multiple PRs use {repo}:{branch}; resolve them first.")
        return rows[0] if rows else None

    def auto_merge(self, repo, pull, method):
        args = ["gh", "pr", "merge", str(pull["number"]), "--repo", repo,
                "--" + method, "--match-head-commit", pull["head"]["sha"]]
        if repo == PUBLIC:
            if pull.get("auto_merge"):
                return
            args.append("--auto")
        elif repo != PRIVATE or pull["head"]["ref"] != SYNC_BRANCH or method != "merge":
            raise PromotionError("Only the reserved private synchronization PR can merge directly.")
        # Private returns have no required CI; GitHub rejects --auto when already clean.
        command(args)

    def request_sync(self):
        self.api(f"repos/{PUBLIC}/dispatches", "POST", {"event_type": "sync-private"})

    def resume_queue(self):
        self.api(f"repos/{PUBLIC}/dispatches", "POST", {"event_type": "resume-promotion"})


class Git:
    def __init__(self, root="."):
        self.root = root

    def run(self, *args, check=True):
        return command(["git", *args], cwd=self.root, check=check)

    def text(self, *args):
        return self.run(*args).stdout.strip()

    def url(self, repo):
        return f"https://github.com/{repo}.git"

    def fetch_main(self, repo, name):
        self.run("fetch", "--no-tags", self.url(repo),
                 f"+refs/heads/main:refs/remotes/{name}/main")
        return self.text("rev-parse", f"refs/remotes/{name}/main")

    def ancestor(self, older, newer):
        result = self.run("merge-base", "--is-ancestor", older, newer, check=False)
        if result.returncode not in (0, 1):
            raise PromotionError(result.stderr.strip())
        return result.returncode == 0

    def tree(self, commit):
        return self.text("rev-parse", commit + "^{tree}")

    def changed_paths(self, left, right):
        result = self.run("diff", "--name-only", "-z", "--no-renames", left, right, "--")
        return {path for path in result.stdout.split("\0") if path}

    def parents(self, commit):
        return self.text("show", "-s", "--format=%P", commit).split()

    def remote_branch(self, repo, branch):
        result = self.run("ls-remote", "--exit-code", "--heads",
                          self.url(repo), "refs/heads/" + branch,
                          check=False)
        if result.returncode == 2:
            return None
        if result.returncode:
            raise PromotionError(result.stderr.strip())
        self.run("fetch", "--no-tags", self.url(repo),
                 "refs/heads/" + branch)
        return self.text("rev-parse", "FETCH_HEAD")

    def pull_head(self, repo, number):
        # GitHub retains this ref after deleting the merged publication branch.
        self.run("fetch", "--no-tags", self.url(repo), f"refs/pull/{number}/head")
        return self.text("rev-parse", "FETCH_HEAD")

    def merge_tree(self, left, right, *, base=None, context="Merge"):
        args = ["merge-tree", "--write-tree"]
        if base is not None:
            args.append("--merge-base=" + base)
        result = self.run(*args, left, right, check=False)
        details = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        if result.returncode == 1:
            raise MergeConflict(f"{context} has conflicts; resolve them manually.\n{details}")
        if result.returncode:
            raise PromotionError(f"Git merge-tree failed during {context.lower()} "
                                 f"(exit {result.returncode}).\n{details}")
        return result.stdout.splitlines()[0]

    def commit_merge(self, left, right, tree, message):
        return self.text("commit-tree", tree, "-p", left, "-p", right, "-m", message)

    def checkout(self, branch, base):
        self.run("checkout", "-B", branch, base)

    def replay(self, source, base, *, mainline=None):
        self.checkout("automation-replay", base)
        args = ["cherry-pick", "-x"]
        if mainline is not None:
            args += ["--mainline", str(mainline)]
        result = self.run(*args, source, check=False)
        if result.returncode:
            self.run("cherry-pick", "--abort", check=False)
            raise PromotionError(
                "The approved change cannot be applied to public main. "
                "Resolve the conflict or an already-applied change before retrying.\n"
                + result.stderr.strip()
            )
        return self.text("rev-parse", "HEAD")

    def push(self, repo, branch, commit, previous=None):
        args = ["-c", "push.followTags=false", "push"]
        # A lease also prevents overwriting a branch created after the initial read.
        args += [f"--force-with-lease=refs/heads/{branch}:{previous or ''}"]
        args += [self.url(repo), f"{commit}:refs/heads/{branch}"]
        self.run(*args)


def read_record(pull, marker, required, optional=()):
    matches = marker.findall(pull.get("body") or "")
    if len(matches) != 1:
        raise PromotionError("Missing or ambiguous promotion metadata.")
    try:
        result = json.loads(matches[0])
    except json.JSONDecodeError as error:
        raise PromotionError("Invalid promotion metadata.") from error
    if (not isinstance(result, dict) or not set(required) <= set(result)
            or not set(result) <= set(required) | set(optional)):
        raise PromotionError("Invalid promotion metadata.")
    if type(result["private_pr"]) is not int or result["private_pr"] <= 0:
        raise PromotionError("Invalid private PR number.")
    for key in set(result) - {"private_pr"}:
        if not isinstance(result[key], str) or not re.fullmatch(r"[0-9a-f]{40}", result[key]):
            raise PromotionError("Invalid promotion object ID.")
    return result


def metadata(pull):
    return read_record(pull, MARKER, {"private_pr", "source", "public_base", "tree"},
                       {"promotion_commit"})


def promotion_body(record):
    return (
        "Publish changes approved in the private development repository.\n\n"
        "The public checks must pass before this PR can merge.\n\n"
        "Resolve conflicts with public main in this branch, preserving the original "
        "promotion commit. Closing this PR without merging blocks later publications.\n\n"
        "<!-- mtlearn-promotion: " + json.dumps(record, sort_keys=True) + " -->\n"
    )


def merged(pull):
    return bool(pull.get("merged_at"))


def approved_source_base(git, source, number):
    parents = git.parents(source)
    if len(parents) == 1:
        return parents[0]
    recovery = APPROVED_MERGE_RECOVERIES.get(source)
    if len(parents) != 2 or recovery != (number, *parents):
        return None
    expected = git.merge_tree(parents[0], parents[1], context="Approved merge recovery")
    return parents[0] if expected == git.tree(source) else None


def promotion_origin(git, head, source, number, public_tip, record=None):
    """Validate or recover the original cherry-pick, including interrupted runs."""
    if record and (record["source"] != source or record["private_pr"] != number):
        raise PromotionError("The public PR refers to another private change.")
    source_base = approved_source_base(git, source, number)
    if source_base is None:
        raise PromotionError("The publication source is not an approved private change.")
    public_history = set(git.text("rev-list", "--first-parent", public_tip).splitlines())
    candidates = ([record["promotion_commit"]] if record and "promotion_commit" in record
                  else git.text("rev-list", head, "--not", public_tip).splitlines())
    origins = []
    for candidate in candidates:
        parents = git.parents(candidate)
        trailer = f"(cherry picked from commit {source})"
        if (len(parents) != 1 or parents[0] not in public_history
                or not git.ancestor(candidate, head)
                or trailer not in git.text("show", "-s", "--format=%B", candidate).splitlines()):
            continue
        if record and (parents[0] != record["public_base"]
                       or git.tree(candidate) != record["tree"]):
            continue
        expected = git.merge_tree(parents[0], source, base=source_base,
                                  context="Promotion origin reconstruction")
        if git.tree(candidate) == expected:
            origins.append(dict(private_pr=number, source=source, public_base=parents[0],
                                tree=expected, promotion_commit=candidate))
    if len(origins) != 1:
        raise PromotionError("The publication history has no unique valid promotion origin; "
                             "preserve and review the branch.")
    return origins[0]


def already_published(git, source_pr, public_tip):
    """A durable receipt is accepted only with its original content proof."""
    if "<!-- mtlearn-already-published:" not in (source_pr.get("body") or ""):
        return False
    record = read_record(source_pr, PUBLISHED_MARKER,
                         {"private_pr", "source", "public_commit", "tree"})
    source = source_pr["merge_commit_sha"]
    public = record["public_commit"]
    source_base = approved_source_base(git, source, source_pr["number"])
    if (record["private_pr"] != source_pr["number"] or record["source"] != source
            or public not in git.text("rev-list", "--first-parent", public_tip).splitlines()
            or source_base is None
            or record["tree"] != git.tree(public)
            or git.merge_tree(public, source, base=source_base,
                              context="Already-published verification") != record["tree"]):
        raise PromotionError("Invalid already-published record on the private PR.")
    return True


def record_already_published(git, github, source_pr, public_tip):
    number, source = source_pr["number"], source_pr["merge_commit_sha"]
    current = github.pull(PRIVATE, number)
    if not merged(current) or current["merge_commit_sha"] != source:
        raise PromotionError("The private PR changed while recording publication; retry.")
    if already_published(git, current, public_tip):
        return
    record = dict(private_pr=number, source=source, public_commit=public_tip,
                  tree=git.tree(public_tip))
    body = (current.get("body") or "").rstrip()
    body += "\n\n<!-- mtlearn-already-published: " + json.dumps(record, sort_keys=True) + " -->\n"
    github.api(f"repos/{PRIVATE}/pulls/{number}", "PATCH", {"body": body})


def publication_base(git, source, private_tip, public_history):
    """Find an integrated base on which the original change can still be reviewed.

    The caller limits history to bases containing every earlier publication.
    A later manual synchronization may have resolved away the original change;
    that is not proof that the change was published and must not skip its review.
    """
    for base in public_history:
        if not git.ancestor(base, private_tip):
            continue
        try:
            tree = git.merge_tree(base, source, base=git.parents(source)[0],
                                  context="Approved change publication")
        except MergeConflict:
            continue
        if tree != git.tree(base):
            return base
    raise PromotionError("The approved change has no usable integrated public base "
                         "containing all earlier publications; review its history manually.")


def promote(git, github, private_tip, public_tip):
    pulls = github.pages(f"repos/{PRIVATE}/pulls", state="closed", base="main")
    pulls = [p for p in pulls if merged(p) and p["head"]["ref"] != SYNC_BRANCH]
    order = git.text("rev-list", "--first-parent", "--reverse", private_tip).splitlines()
    positions = {sha: i for i, sha in enumerate(order)}
    if any(p["merge_commit_sha"] not in positions for p in pulls):
        raise PromotionError("A merged private PR is outside the main history.")
    pulls.sort(key=lambda p: positions[p["merge_commit_sha"]])
    public_history = git.text("rev-list", "--first-parent", public_tip).splitlines()
    public_positions = {sha: i for i, sha in enumerate(public_history)}
    earlier_publications = set()

    for source_pr in pulls:
        number = source_pr["number"]
        source = source_pr["merge_commit_sha"]
        if already_published(git, source_pr, public_tip):
            earlier_publications.add(read_record(
                source_pr, PUBLISHED_MARKER,
                {"private_pr", "source", "public_commit", "tree"})["public_commit"])
            continue
        branch = PROMOTION_PREFIX + str(number)
        public_pr = github.find_pull(PUBLIC, branch)
        old_record = None
        if public_pr:
            old_record = metadata(public_pr)
            if old_record["source"] != source or old_record["private_pr"] != number:
                raise PromotionError("The public PR refers to another private change.")
            if merged(public_pr):
                published = public_pr["merge_commit_sha"]
                if published not in public_positions:
                    raise PromotionError("An earlier publication is outside public main history.")
                earlier_publications.add(published)
                continue
            if public_pr["state"] != "open":
                raise PromotionError(f"Public PR #{public_pr['number']} was closed without merging. "
                                     "Reopen it to continue; publication cancellation is not supported.")
        if approved_source_base(git, source, number) is None:
            raise PromotionError(f"Private PR #{number} must use squash merge.")
        if any(not git.ancestor(commit, private_tip) for commit in earlier_publications):
            github.request_sync()
            return "Waiting for earlier public promotions to be reconciled into private main."
        previous = git.remote_branch(PUBLIC, branch)
        if public_pr and previous != public_pr["head"]["sha"]:
            raise PromotionError("The publication branch changed or disappeared; retry with a fresh PR head.")
        if previous:
            record = promotion_origin(git, previous, source, number, public_tip, old_record)
            candidate = previous
        else:
            try:
                expected = git.merge_tree(public_tip, source, base=git.parents(source)[0],
                                          context="Already-published comparison")
            except MergeConflict:
                # A conflict is not an already-published proof. Open a reviewable
                # promotion on a compatible historical base instead.
                expected = None
            if expected == git.tree(public_tip):
                record_already_published(git, github, source_pr, public_tip)
                earlier_publications.add(public_tip)
                continue
            latest_publication_index = min((public_positions[c] for c in earlier_publications),
                                           default=len(public_history) - 1)
            base = publication_base(git, source, private_tip,
                                    public_history[:latest_publication_index + 1])
            if len(git.parents(source)) == 1:
                candidate = git.replay(source, base)
            else:
                candidate = git.replay(source, base, mainline=1)
            record = dict(private_pr=number, source=source, public_base=base,
                          tree=git.tree(candidate), promotion_commit=candidate)
        if candidate != previous:
            git.push(PUBLIC, branch, candidate, previous)
        if public_pr:
            if old_record != record:
                body = MARKER.sub(lambda _: MARKER.search(promotion_body(record))[0],
                                  public_pr["body"])
                github.api(f"repos/{PUBLIC}/pulls/{public_pr['number']}", "PATCH", {"body": body})
            public_pr = github.pull(PUBLIC, public_pr["number"])
        else:
            public_pr = github.api(f"repos/{PUBLIC}/pulls", "POST", {
                "head": branch, "base": "main", "title": source_pr["title"],
                "body": promotion_body(record),
            })
        # Keep the original branch and PR available even when updating conflicts.
        # Waiting for all public changes to enter private first would deadlock.
        if not git.ancestor(public_tip, candidate):
            try:
                tree = git.merge_tree(candidate, public_tip, context="Publication branch update")
            except MergeConflict as error:
                raise MergeConflict(
                    f"Public PR #{public_pr['number']} needs conflict resolution: "
                    f"{public_pr['html_url']}\n{error}"
                ) from error
            updated = git.commit_merge(candidate, public_tip, tree, "Merge public main into promotion")
            if not git.ancestor(candidate, updated):
                raise PromotionError("A publication update must preserve the existing branch history.")
            git.push(PUBLIC, branch, updated, candidate)
            candidate = updated
            public_pr = github.pull(PUBLIC, public_pr["number"])
        if git.tree(candidate) == git.tree(public_tip):
            raise PromotionError(
                f"Public PR #{public_pr['number']} has no remaining content changes: "
                f"{public_pr['html_url']}\nEmpty promotion completion requires manual review; "
                "it is not automatically marked as published or cancelled."
            )
        github.auto_merge(PUBLIC, public_pr, "squash")
        return f"Publication pending: {public_pr['html_url']}"
    return "All merged private PRs have been published."


def verify_published_change(git, github, commit, private_tip, pull, integrated_public):
    """Return the expected tree at the last integrated public main commit.

    The original marker remains immutable. The integration base can be newer,
    or precede the squash parent when the squash resolves a blocked public change.
    """
    record = metadata(pull)
    source_pr = github.pull(PRIVATE, record["private_pr"])
    source = record["source"]
    source_base = approved_source_base(git, source, record["private_pr"])
    if (not merged(source_pr) or source_pr["base"]["ref"] != "main"
            or source_pr["head"]["ref"] == SYNC_BRANCH
            or source_pr["merge_commit_sha"] != source
            or pull["head"]["ref"] != PROMOTION_PREFIX + str(record["private_pr"])
            or len(git.parents(commit)) != 1
            or source_base is None
            or not git.ancestor(source, private_tip)
            or not git.ancestor(record["public_base"], integrated_public)
            or not git.ancestor(integrated_public, private_tip)
            or not git.ancestor(integrated_public, git.parents(commit)[0])):
        raise PromotionError("The public squash does not match approved private history.")
    head = git.pull_head(PUBLIC, pull["number"])
    if head != pull["head"]["sha"]:
        raise PromotionError("The retained public PR head does not match the merged PR.")
    promotion_origin(git, head, source, record["private_pr"], git.parents(commit)[0], record)
    published = git.merge_tree(git.parents(commit)[0], head, context="Public squash verification")
    if published != git.tree(commit):
        raise PromotionError("The public squash content does not match the merged PR head.")
    try:
        return git.merge_tree(integrated_public, source, base=source_base,
                              context="Expected promotion reconstruction")
    except MergeConflict as error:
        # Manual private synchronization can resolve the same conflict later
        # resolved in the public PR. The old base is safe only on paths where
        # these already-integrated public changes now agree on both sides.
        integrated_paths = git.changed_paths(record["public_base"], integrated_public)
        differing_paths = integrated_paths & git.changed_paths(private_tip, commit)
        if differing_paths:
            paths = ", ".join(repr(path) for path in sorted(differing_paths))
            raise MergeConflict(
                "Expected promotion reconstruction has conflicts and the prior public "
                f"changes still differ on: {paths}. Resolve synchronization manually."
            ) from error
        return record["tree"]


def synchronize(git, github, private_tip, public_tip):
    if git.ancestor(public_tip, private_tip):
        github.resume_queue()
        return "Private main already includes public main."
    git.checkout("automation-sync", private_tip)
    commits = git.text("rev-list", "--first-parent", "--reverse",
                       private_tip + ".." + public_tip).splitlines()
    history = git.text("rev-list", "--first-parent", public_tip).splitlines()
    integrated_public = next((c for c in history if git.ancestor(c, private_tip)), None)
    if integrated_public is None:
        raise PromotionError("Private main has no integrated public main history.")
    publications = {}
    for commit in commits:
        associated = github.pages(f"repos/{PUBLIC}/commits/{commit}/pulls")
        promotions = [p for p in associated if merged(p)
                      and p["merge_commit_sha"] == commit and p["base"]["ref"] == "main"
                      and p["head"]["ref"].startswith(PROMOTION_PREFIX)]
        if len(promotions) > 1:
            raise PromotionError("Ambiguous public promotion history.")
        if promotions:
            publications[commit] = promotions[0]
    index = 0
    while index < len(commits):
        # Integrate through the next verified promotion as one unit. Its squash
        # includes preceding public commits and any reviewed conflict resolution.
        # Splitting that unit can conflict before reaching the resolution, or make
        # the original feature impossible to reconstruct on an intermediate base.
        following = next((i for i in range(index, len(commits))
                          if commits[i] in publications), None)
        if following is not None:
            index = following
        commit = commits[index]
        current = git.text("rev-parse", "HEAD")
        equivalent = False
        if commit in publications:
            expected = verify_published_change(git, github, commit, current,
                                                publications[commit], integrated_public)
            equivalent = expected == git.tree(commit)
            if not equivalent:
                tree = git.merge_tree(current, commit, base=expected,
                                      context="Public corrections and private work")
        else:
            tree = git.merge_tree(current, commit, context="Public synchronization")
        message = "Sync public main: " + commit
        if equivalent:
            git.run("merge", "--strategy=ours", "--no-ff", "-m", message, commit)
        else:
            candidate = git.commit_merge(current, commit, tree, message)
            git.checkout("automation-sync", candidate)
        integrated_public = commit
        index += 1
    candidate = git.text("rev-parse", "HEAD")
    previous = git.remote_branch(PRIVATE, SYNC_BRANCH)
    if (previous and git.ancestor(private_tip, previous) and git.ancestor(public_tip, previous)
            and git.tree(previous) == git.tree(candidate)):
        candidate = previous
    if candidate != previous:
        git.push(PRIVATE, SYNC_BRANCH, candidate, previous)
    sync_pr = github.find_pull(PRIVATE, SYNC_BRANCH, state="open")
    if not sync_pr:
        sync_pr = github.api(f"repos/{PRIVATE}/pulls", "POST", {
            "head": SYNC_BRANCH, "base": "main", "title": "chore: sync public main",
            "body": "Reconcile public main into private main.\n\n"
                    "Use a merge commit to retain the public ancestry. "
                    "This reserved branch is excluded from publication.\n",
        })
    else:
        sync_pr = github.pull(PRIVATE, sync_pr["number"])
    github.auto_merge(PRIVATE, sync_pr, "merge")
    # The reserved private PR merges directly, without --auto. Dispatch only
    # after that operation succeeds; private Actions cannot wake the public queue.
    github.resume_queue()
    return f"Synchronization completed; publication queue resumed: {sync_pr['html_url']}"


def write_summary(message):
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("promote", "sync"))
    args = parser.parse_args()
    if os.environ.get("GITHUB_REPOSITORY") != PUBLIC:
        raise PromotionError("Repository automation must run in the public repository.")
    git, github = Git(), GitHub()
    if git.text("status", "--porcelain", "--untracked-files=no"):
        raise PromotionError("The automation checkout must be clean.")
    private_tip = git.fetch_main(PRIVATE, "private")
    public_tip = git.fetch_main(PUBLIC, "public")
    try:
        message = (promote if args.mode == "promote" else synchronize)(
            git, github, private_tip, public_tip)
    except PromotionError as error:
        write_summary(str(error))
        raise
    print(message)
    write_summary(message)


if __name__ == "__main__":
    try:
        main()
    except PromotionError as error:
        raise SystemExit(str(error))
