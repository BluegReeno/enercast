---
description: Create an atomic commit for current changes
---

# Commit Changes

## Process

### 1. Review Changes

```bash
git status
git diff HEAD
git diff --stat HEAD
```

Check for new untracked files:
```bash
git ls-files --others --exclude-standard
```

### 2. Stage Files

Add the untracked and changed files relevant to the current work.

**Do NOT stage:**
- `.env` or credential files
- Large binary files
- Files unrelated to the current task

### 3. Create Commit

Write an atomic commit message with a conventional commit tag:

- `feat:` — New capability or feature
- `fix:` — Bug fix
- `refactor:` — Code restructure without behavior change
- `docs:` — Documentation only
- `test:` — Test additions or fixes
- `chore:` — Build, CI, tooling changes
- `perf:` — Performance improvement

**Use scope for clarity:**
```
feat(auth): add OAuth2 login flow
fix(api): resolve timeout on large payloads
refactor(db): simplify query builder
```

**Commit message format:**
```
tag(scope): concise description of what changed

[Optional body explaining WHY this change was made,
not just what changed. Include context that isn't
obvious from the diff.]

[Optional: Fixes #123, Closes #456]
```

### 4. Update STATUS.md

Before committing, update `.claude/STATUS.md` to reflect the work done:

- If a feature or improvement was completed, add/update the corresponding `[x]` entry
- If a task file in `.claude/tasks/` was completed, ensure STATUS.md reflects it
- Update the `Last Updated` date at the top if substantive changes were made
- Keep entries concise (one line per item, same style as existing entries)

**Always stage STATUS.md with the commit** so project state stays in sync with the code.

### 5. Capture AI Context Changes

If any AI context assets were modified in this commit, add a `Context:` section to the commit body:

```
feat(api): add rate limiting middleware

Added per-user rate limiting with sliding window.
Previously unlimited requests could overload the DB.

Context:
- Updated .claude/rules/api.md with rate limiting conventions
- Added .claude/docs/rate-limiting-guide.md
```

**What counts as AI context changes:**
- `.claude/rules/` — path-scoped conventions
- `.claude/commands/` — slash commands
- `.claude/docs/` — reference documentation
- `.claude/reference/` — pattern libraries
- `CLAUDE.md` — global rules

**Why this matters:** Your git log is long-term memory. Future agents use `git log` to understand project history. If context changes aren't captured in commits, the AI layer's evolution becomes invisible.
