# Purging the old repositories before this one is published

The thesis repositories under the author's GitHub account carry a personal access token in their history,
and the working-tree copy of the repository README still carries it in the current revision. This repo was
built from the working tree only, never from that history, and the token is not in this repo (checked by
`tests/test_no_secrets.py`). The steps below are for the author to run on his own account. The build agent
does not push, does not run `git filter-repo` and does not touch the old repositories.

## 1. Revoke the token first

Revocation is the only step that makes the token harmless. Forks, clones, caches and GitHub's own views may
keep history that a rewrite does not reach.

1. GitHub, Settings, Developer settings, Personal access tokens. Delete the token (both the classic and the
   fine-grained lists).
2. Check Security, Secret scanning alerts on each affected repository and close them once the token is gone.

## 2. Rewrite the history of each affected repository

Work on a fresh mirror clone, not on the working copy.

```
pip install git-filter-repo
git clone --mirror git@github.com:Nasheor/<repo>.git
cd <repo>.git
```

Put the token, and only the token, in a file outside any repository, for example `C:\Users\<you>\expressions.txt`,
one line in the form `<token>==>REMOVED`. Do not commit that file anywhere. Then:

```
git filter-repo --replace-text C:\Users\<you>\expressions.txt
git push --force --mirror
```

`--replace-text` rewrites every blob in every commit, so the README and any other file that ever held the
string are cleaned together. Repeat for each affected repository (`Intelligent_Fleet_Allocation` and any
sibling thesis repository whose history carries it).

## 3. Clean the local working copies

The local working tree under `F:\Stuff\PhD\Codebase\Intelligent_Fleet_Allocation` still holds the string in
its README. Edit the README, commit, and either re-clone from the rewritten remote or run the same
`git filter-repo` command inside the working copy (it refuses to run on a non-fresh clone unless given
`--force`, which is acceptable here once the remote is clean). Afterwards:

```
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

## 4. Ask GitHub to drop cached views

Rewritten commits remain reachable by hash through cached pull requests and the activity feed until GitHub
runs garbage collection. Contact GitHub Support with the repository names and ask for the dereferenced
objects to be removed.

## 5. Then publish this repository

Once the token is revoked and the old histories are rewritten, the orchestrating session may push this
repo's `build/G11` branch to a private repository under `Nasheor`, and it becomes public only on the author's
word. Until then the README's "Publication status" paragraph stays as it is.
