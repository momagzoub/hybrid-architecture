# GITHUB_GUIDE.md — First-time GitHub for this project

> Written for someone who has never used Git or GitHub. The goal is to get the repo online, looking polished, and updateable — without learning more Git than you need.

There are two layers to this: **Git** (the tool that tracks changes to your files) and **GitHub** (the website that hosts a copy of your repo on the internet, lets others see it, and runs your CI). You'll use both, but Git is the one you'll actually live in.

---

## Part 1 — One-time setup (do this once, ever)

### 1.1 Install Git

On macOS: `brew install git` if you have Homebrew, otherwise download from [git-scm.com](https://git-scm.com/). On Linux: `sudo apt install git`. On Windows: download Git for Windows.

Verify: `git --version` should print something like `git version 2.45.0`.

### 1.2 Tell Git who you are

```bash
git config --global user.name "Mohamed Magzoub"
git config --global user.email "m0hamed@mit.edu"
git config --global init.defaultBranch main
```

The email here will appear publicly on every commit. If you want it private, use GitHub's `noreply` email — settings → emails → "Keep my email address private" → use the address shown there. (For an MIT email on a public repo this is often the right call.)

### 1.3 GitHub account

Already set up: [github.com/momagzoub](https://github.com/momagzoub). If you haven't already, apply for the free [GitHub Education / Pro plan](https://education.github.com/) with your MIT email — it unlocks unlimited private repos and Copilot at no cost.

### 1.4 Set up SSH (so you don't type passwords)

```bash
ssh-keygen -t ed25519 -C "m0hamed@mit.edu"
# press enter at every prompt to accept defaults
cat ~/.ssh/id_ed25519.pub
```

Copy the output. On GitHub → settings → SSH and GPG keys → New SSH key → paste. This lets your laptop push to GitHub without typing your password every time.

Test it: `ssh -T git@github.com` should say `Hi momagzoub!`.

---

## Part 2 — Putting this project on GitHub

### 2.1 Initialize the repo

From inside the `Hybrid Architecture` folder:

```bash
git init
git add CLAUDE.md PROJECT_PLAN.md README.md GITHUB_GUIDE.md docs/ notebooks/ src/ tests/ pyproject.toml .gitignore
git commit -m "initial project scaffold"
```

> **Important:** `.gitignore` should exclude `data/`, `results/`, model caches, `__pycache__/`, and `.ipynb_checkpoints/`. A scaffold is included with this project — don't remove it.

### 2.2 Create the GitHub repo

On github.com → "+" in the top-right → "New repository":
- **Name:** `hybrid-architecture` (lowercase, hyphens)
- **Description:** *An attention-pattern atlas of how language models learn what to compute in parallel.*
- **Visibility:** Public (this is a portfolio piece — make it visible)
- **DO NOT** check "Initialize with a README" — you already have one.

After creating, GitHub will show a few lines starting with `git remote add origin …`. Run them:

```bash
git remote add origin git@github.com:momagzoub/hybrid-architecture.git
git branch -M main
git push -u origin main
```

Reload the GitHub page. The README should now be visible.

---

## Part 3 — The daily Git rhythm

You only need four commands, 95% of the time:

```bash
git status                              # what's changed?
git add <file>                          # stage specific files
git commit -m "add KV cache walkthrough"  # save a checkpoint
git push                                # upload to GitHub
```

**Commit-message conventions** (matches `CLAUDE.md §5`):
- Present-tense imperative: `add KV cache walkthrough`, not `added` or `adds`.
- One concept per commit. If you have two unrelated changes, make two commits.
- 50-character limit on the subject line. If it doesn't fit, the commit does too much.

**Commit often.** Once or twice per work session is normal. Each commit is a save-point you can come back to.

### When you want to undo something

- *I want to undo a commit I haven't pushed yet:* `git reset --soft HEAD~1`. The changes stay in your working directory, just unstaged.
- *I want to throw away uncommitted changes to a file:* `git restore <file>`. **This is destructive — only do it if you're sure.**
- *I broke everything and want to go back to last commit:* `git reset --hard HEAD`. **Even more destructive.** Ask Claude before doing this in anger.

---

## Part 4 — Making the repo look like a portfolio piece

This is what makes GitHub work for you when an inference engineer at Anthropic or DeepMind glances at it.

### 4.1 The README is the only page most visitors read

Optimize it. The current `README.md` already follows the structure: punchy headline, one-paragraph TL;DR, why it matters, what's in the repo, how to reproduce, prior-work positioning.

When Phase 2 lands, **add a hero image at the top** — the atlas plot. A reader should be able to look at one image and *get* the project.

### 4.2 Pin the repo on your profile

GitHub profile → "Customize your pins" → drag this repo to the top. You get six pinned slots; this project lives in one of them.

### 4.3 The profile README

Create a repo named exactly after your GitHub username — `momagzoub/momagzoub`. The README inside it renders on your profile page at [github.com/momagzoub](https://github.com/momagzoub). Use it for: one paragraph about what you work on, a list of currently-active projects, MIT affiliation, contact info. Keep it short.

### 4.4 Use Releases for milestones

When Phase 2 / 3 / 4 lands, cut a Release on GitHub (Releases → "Draft a new release" → pick a tag like `v0.2-atlas`). Write a short summary of what's new. Releases are how you turn "things I did" into a timeline of accomplishments visible to recruiters.

### 4.5 GitHub Pages for the blog post

When Phase 5 lands and you write the long-form blog post, host it on GitHub Pages:
- Create a `docs/` folder with the post as `index.md` (or build a small Jekyll/MkDocs site).
- Settings → Pages → "Deploy from branch" → `main` → `/docs`.
- Your post is live at `https://momagzoub.github.io/hybrid-architecture/`.

Link this from the README, your LinkedIn, and the project pin.

### 4.6 CI badge

A green "tests passing" badge at the top of the README signals "this person tests their code." The `.github/workflows/ci.yml` scaffold included with this project does that automatically. Once tests are added and passing, the badge updates itself.

---

## Part 5 — What NOT to commit

- **Big files.** Models, dataset dumps, result tensors. They bloat the repo and slow down clones. Use `.gitignore` aggressively. If you need to share results, use HuggingFace Hub or a release artifact.
- **Secrets.** API keys, tokens, anything in a `.env` file. Once it's in `git`, it's effectively public forever (even if you delete it later — git history remembers). If you do leak a secret, *rotate it immediately*, then [follow GitHub's removal guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
- **Personal information.** Especially in code comments or notebook outputs. Notebooks save output cells; check before committing.

---

## Part 6 — When you get stuck

- `git status` is your best friend. It tells you exactly what state you're in.
- The Git book ([git-scm.com/book](https://git-scm.com/book/en/v2)) is free and excellent. First three chapters cover 99% of what you'll do.
- Ask Claude. "I ran X and got Y, what does it mean?" works better than "git is broken."
- If you genuinely can't recover, the worst case is `git clone <repo>` into a fresh directory, copy your uncommitted files into it, and start over. You almost never need this — but knowing it's an escape hatch helps.

---

## Quick reference card

```bash
# every session
git status              # what's the state?
git pull                # get any updates from GitHub
# … do work …
git add <files>         # stage changes
git commit -m "msg"     # save checkpoint
git push                # publish to GitHub

# occasionally
git log --oneline       # see recent commits
git diff                # see unstaged changes
git diff --cached       # see staged changes
git checkout -b <name>  # start a branch for an experiment
git merge <branch>      # merge a branch back into main
```

That's most of what you need for this project. Branches and merges become relevant if you start running risky experiments you might want to throw away — but you can ignore them until you do.
