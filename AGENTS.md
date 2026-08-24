# Agent Instructions

## Stacked PRs

This repo uses the `gh stack` extension (`github/gh-stack`) to manage
stacked branches/PRs — don't hand-roll stacking with raw `git rebase`
+ manual `gh pr create --base <parent>` calls when `gh stack` covers it.

- `gh stack init <branch1> <branch2> ...` — adopt existing branches into a
  stack (bottom to top), or create new ones.
- `gh stack link <pr-or-branch> ...` — register existing PRs as a stack on
  GitHub without needing local branch tracking.
- `gh stack view` — show the current stack and each PR's status.
- `gh stack sync` — fetch `main`, cascade-rebase the stack onto it, and
  push (`--force-with-lease --atomic`). Run this whenever the trunk branch
  moves out from under a stack.
- `gh stack submit` — open/update PRs for the stack on GitHub.

Known gotcha: if commit signing is configured through a agent (e.g. 1Password
SSH signing) that isn't available non-interactively, `git rebase --continue`
can fail with `1Password: failed to fill whole buffer` on an otherwise
conflict-free replay. Retry the failing step with
`git -c commit.gpgsign=false commit --no-edit` — content-wise it's the same
commit, just created without a GPG signature.
