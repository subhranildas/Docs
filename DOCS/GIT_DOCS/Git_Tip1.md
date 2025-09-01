## Removing Deleted Remote Branches from `git branch -a`

### The Problem

You’ve deleted some branches on the remote repository, but when you run:

```bash
git fetch
git branch -a
```

The deleted branches still show up in the list.
Why does this happen, and how can you fix it?

### Explanation

When a branch is deleted on the remote, Git does not automatically clean up your remote-tracking branches.
These stale references stay in your local clone until you explicitly prune them.

### Possible Solutions

- 1. Prune for a specific remote (usually origin)

```bash
git fetch --prune origin
```

- 2. Prune all remotes at once

```bash
git fetch --all --prune
```

- 3. Automatically prune on every fetch

Set this configuration to save effort:

```bash
git config --global fetch.prune true
```

After pruning, check again:

```bash
git branch -a
```

The deleted remote branches should no longer appear.
If we also have local branches (not just remote-tracking ones) with the same name,
they won’t be deleted automatically. We need to remove them manually:

```bash
git branch -d branch_name # safe delete (won’t delete if unmerged)
git branch -D branch_name # force delete
```

## Differentiating Branch Types

When you run git branch -a:

Local branches appear like:

```bash
- main
  feature-x
```

Remote-tracking branches appear like:

```bash
remotes/origin/main
remotes/origin/feature-x
```

Only the remote-tracking branches (remotes/...) get cleaned up with pruning.
