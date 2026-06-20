//! Git-backed corroboration of code/commit references found in search results.

use crate::{CorroboratedRef, Corroboration};
use regex::Regex;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration as StdDuration, Instant};

pub(crate) fn corroborate_text(
    cwd: Option<&str>,
    project_root: Option<&str>,
    text: &str,
) -> Corroboration {
    let candidates = extract_corroboration_candidates(text);
    if candidates.is_empty() {
        return Corroboration {
            repo: None,
            refs: Vec::new(),
        };
    }

    let has_local_refs = candidates
        .iter()
        .any(|candidate| candidate.kind != CorroborationRefKind::Pr);
    let repo_lookup = if has_local_refs {
        locate_git_repo(cwd, project_root)
    } else {
        RepoLookup::NoRepo
    };
    let repo_path = match &repo_lookup {
        RepoLookup::Found(repo) => Some(repo.display().to_string()),
        RepoLookup::NoRepo | RepoLookup::Unknown => None,
    };

    let refs = candidates
        .into_iter()
        .map(|candidate| match candidate.kind {
            CorroborationRefKind::Pr => CorroboratedRef {
                kind: candidate.kind.as_str().to_string(),
                reference: candidate.reference,
                status: "unresolved".to_string(),
                detail: None,
                reason: Some("needs_network".to_string()),
            },
            _ => match &repo_lookup {
                RepoLookup::Found(repo) => resolve_local_ref(repo, cwd, project_root, candidate),
                RepoLookup::NoRepo => CorroboratedRef {
                    kind: candidate.kind.as_str().to_string(),
                    reference: candidate.reference,
                    status: "unresolved".to_string(),
                    detail: None,
                    reason: Some("no_repo".to_string()),
                },
                RepoLookup::Unknown => CorroboratedRef {
                    kind: candidate.kind.as_str().to_string(),
                    reference: candidate.reference,
                    status: "unknown".to_string(),
                    detail: None,
                    reason: Some("git_error".to_string()),
                },
            },
        })
        .collect();

    Corroboration {
        repo: repo_path,
        refs,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CorroborationRefKind {
    Commit,
    Branch,
    File,
    Pr,
}

impl CorroborationRefKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Commit => "commit",
            Self::Branch => "branch",
            Self::File => "file",
            Self::Pr => "pr",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CorroborationCandidate {
    pub(crate) kind: CorroborationRefKind,
    pub(crate) reference: String,
}

pub(crate) fn extract_corroboration_candidates(text: &str) -> Vec<CorroborationCandidate> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    let pr_hash = Regex::new(r"(?i)\b(?:PR\s*)?#([0-9]{1,8})\b").expect("valid PR regex");
    for captures in pr_hash.captures_iter(text) {
        push_corroboration_candidate(
            &mut candidates,
            &mut seen,
            CorroborationRefKind::Pr,
            format!("#{}", &captures[1]),
        );
    }
    let pr_pull = Regex::new(r"(?i)\bpull/([0-9]{1,8})\b").expect("valid pull regex");
    for captures in pr_pull.captures_iter(text) {
        push_corroboration_candidate(
            &mut candidates,
            &mut seen,
            CorroborationRefKind::Pr,
            format!("#{}", &captures[1]),
        );
    }

    let commit =
        Regex::new(r"(?i)(^|[^0-9a-f])([0-9a-f]{7,40})([^0-9a-f]|$)").expect("valid commit regex");
    for captures in commit.captures_iter(text) {
        push_corroboration_candidate(
            &mut candidates,
            &mut seen,
            CorroborationRefKind::Commit,
            captures[2].to_string(),
        );
    }

    for pattern in [
        r"(?i)\bbranch\s+([A-Za-z0-9][A-Za-z0-9._/\-]{0,200})",
        r"(?i)\bgit\s+(?:checkout|switch)\s+(?:--track\s+)?(?:-c\s+)?([A-Za-z0-9][A-Za-z0-9._/\-]{0,200})",
    ] {
        let regex = Regex::new(pattern).expect("valid branch regex");
        for captures in regex.captures_iter(text) {
            if let Some(reference) = clean_reference_token(&captures[1]) {
                push_corroboration_candidate(
                    &mut candidates,
                    &mut seen,
                    CorroborationRefKind::Branch,
                    reference,
                );
            }
        }
    }
    let origin_branch =
        Regex::new(r"\borigin/([A-Za-z0-9][A-Za-z0-9._/\-]{0,200})").expect("valid origin regex");
    for captures in origin_branch.captures_iter(text) {
        if let Some(reference) = clean_reference_token(&format!("origin/{}", &captures[1])) {
            push_corroboration_candidate(
                &mut candidates,
                &mut seen,
                CorroborationRefKind::Branch,
                reference,
            );
        }
    }

    for pattern in [
        r#"(?m)(?:^|[\s("'`])(/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+\.[A-Za-z][A-Za-z0-9._-]{0,20})"#,
        r"\b((?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+\.[A-Za-z][A-Za-z0-9._-]{0,20})\b",
    ] {
        let regex = Regex::new(pattern).expect("valid file regex");
        for captures in regex.captures_iter(text) {
            if let Some(reference) = clean_file_reference(&captures[1]) {
                push_corroboration_candidate(
                    &mut candidates,
                    &mut seen,
                    CorroborationRefKind::File,
                    reference,
                );
            }
        }
    }

    candidates
}

fn push_corroboration_candidate(
    candidates: &mut Vec<CorroborationCandidate>,
    seen: &mut HashSet<(CorroborationRefKind, String)>,
    kind: CorroborationRefKind,
    reference: String,
) {
    if reference.trim().is_empty() {
        return;
    }
    let key = (kind, reference.clone());
    if seen.insert(key) {
        candidates.push(CorroborationCandidate { kind, reference });
    }
}

fn clean_reference_token(value: &str) -> Option<String> {
    let trimmed = value.trim_matches(reference_boundary_character).trim();
    if trimmed.is_empty()
        || trimmed.starts_with('-')
        || trimmed.ends_with('/')
        || trimmed.contains("..")
        || trimmed.contains("://")
    {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn clean_file_reference(value: &str) -> Option<String> {
    let reference = clean_reference_token(value)?;
    if reference.starts_with("origin/")
        || reference.starts_with("http")
        || reference.contains('#')
        || reference.len() > 512
    {
        return None;
    }
    Some(reference)
}

fn reference_boundary_character(character: char) -> bool {
    matches!(
        character,
        '"' | '\'' | '`' | ')' | ']' | '}' | '>' | ',' | ';' | ':' | '!' | '.'
    )
}

enum RepoLookup {
    Found(PathBuf),
    NoRepo,
    Unknown,
}

fn locate_git_repo(cwd: Option<&str>, project_root: Option<&str>) -> RepoLookup {
    let Some(start) = repo_start_path(cwd, project_root) else {
        return RepoLookup::NoRepo;
    };
    if !start.exists() {
        return RepoLookup::NoRepo;
    }
    match run_git_read(&start, &["rev-parse", "--show-toplevel"]) {
        GitOutcome::Success(stdout) => {
            let repo = stdout.lines().next().unwrap_or("").trim();
            if repo.is_empty() {
                RepoLookup::Unknown
            } else {
                let repo = PathBuf::from(repo);
                RepoLookup::Found(fs::canonicalize(&repo).unwrap_or(repo))
            }
        }
        GitOutcome::NonZero => RepoLookup::NoRepo,
        GitOutcome::Failed => RepoLookup::Unknown,
    }
}

fn repo_start_path(cwd: Option<&str>, project_root: Option<&str>) -> Option<PathBuf> {
    cwd.filter(|value| !value.trim().is_empty())
        .or_else(|| project_root.filter(|value| !value.trim().is_empty()))
        .map(PathBuf::from)
}

fn resolve_local_ref(
    repo: &Path,
    cwd: Option<&str>,
    project_root: Option<&str>,
    candidate: CorroborationCandidate,
) -> CorroboratedRef {
    match candidate.kind {
        CorroborationRefKind::Commit => resolve_commit_ref(repo, candidate),
        CorroborationRefKind::Branch => resolve_branch_ref(repo, candidate),
        CorroborationRefKind::File => resolve_file_ref(repo, cwd, project_root, candidate),
        CorroborationRefKind::Pr => unreachable!("PR refs do not resolve locally"),
    }
}

fn resolve_commit_ref(repo: &Path, candidate: CorroborationCandidate) -> CorroboratedRef {
    let commitish = format!("{}^{{commit}}", candidate.reference);
    match run_git_read(repo, &["cat-file", "-e", &commitish]) {
        GitOutcome::Success(_) => {
            let detail = match run_git_read(
                repo,
                &[
                    "log",
                    "-1",
                    "--format=%h %s",
                    "--no-show-signature",
                    &candidate.reference,
                ],
            ) {
                GitOutcome::Success(stdout) => stdout
                    .lines()
                    .next()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(ToOwned::to_owned),
                GitOutcome::NonZero | GitOutcome::Failed => None,
            };
            CorroboratedRef {
                kind: candidate.kind.as_str().to_string(),
                reference: candidate.reference,
                status: "present".to_string(),
                detail,
                reason: None,
            }
        }
        GitOutcome::NonZero => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "missing".to_string(),
            detail: None,
            reason: None,
        },
        GitOutcome::Failed => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "unknown".to_string(),
            detail: None,
            reason: Some("git_error".to_string()),
        },
    }
}

fn resolve_branch_ref(repo: &Path, candidate: CorroborationCandidate) -> CorroboratedRef {
    let full_ref = if let Some(remote_branch) = candidate.reference.strip_prefix("origin/") {
        format!("refs/remotes/origin/{remote_branch}")
    } else {
        format!("refs/heads/{}", candidate.reference)
    };
    match run_git_read(repo, &["rev-parse", "--verify", "--quiet", &full_ref]) {
        GitOutcome::Success(_) => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "present".to_string(),
            detail: None,
            reason: None,
        },
        GitOutcome::NonZero => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "missing".to_string(),
            detail: None,
            reason: None,
        },
        GitOutcome::Failed => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "unknown".to_string(),
            detail: None,
            reason: Some("git_error".to_string()),
        },
    }
}

fn resolve_file_ref(
    repo: &Path,
    cwd: Option<&str>,
    project_root: Option<&str>,
    candidate: CorroborationCandidate,
) -> CorroboratedRef {
    let Some(relative_path) =
        candidate_file_repo_path(repo, cwd, project_root, &candidate.reference)
    else {
        return CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "missing".to_string(),
            detail: None,
            reason: None,
        };
    };
    let relative_text = relative_path.to_string_lossy().to_string();
    let on_disk = repo.join(&relative_path).exists();
    match run_git_read(repo, &["ls-files", "--error-unmatch", "--", &relative_text]) {
        GitOutcome::Success(_) => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "present".to_string(),
            detail: Some("tracked".to_string()),
            reason: None,
        },
        GitOutcome::NonZero if on_disk => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "untracked".to_string(),
            detail: None,
            reason: None,
        },
        GitOutcome::NonZero => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "missing".to_string(),
            detail: None,
            reason: None,
        },
        GitOutcome::Failed => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "unknown".to_string(),
            detail: None,
            reason: Some("git_error".to_string()),
        },
    }
}

fn candidate_file_repo_path(
    repo: &Path,
    cwd: Option<&str>,
    project_root: Option<&str>,
    reference: &str,
) -> Option<PathBuf> {
    let reference_path = PathBuf::from(reference);
    if reference_path.is_absolute() {
        return path_under_repo(repo, &reference_path);
    }

    for base in [
        cwd.and_then(|value| (!value.trim().is_empty()).then_some(PathBuf::from(value))),
        project_root.and_then(|value| (!value.trim().is_empty()).then_some(PathBuf::from(value))),
        Some(repo.to_path_buf()),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(path) = path_under_repo(repo, &base.join(reference)) {
            return Some(path);
        }
    }
    None
}

fn path_under_repo(repo: &Path, path: &Path) -> Option<PathBuf> {
    let normalized = normalize_path(path);
    let normalized_repo = normalize_path(repo);
    normalized
        .strip_prefix(&normalized_repo)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())
        .map(Path::to_path_buf)
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

enum GitOutcome {
    Success(String),
    NonZero,
    Failed,
}

fn run_git_read(repo: &Path, args: &[&str]) -> GitOutcome {
    record_git_invocation(args);

    let mut command = ProcessCommand::new(git_binary());
    command
        .arg("-c")
        .arg("core.hooksPath=/dev/null")
        .arg("-c")
        .arg("pager.branch=false")
        .arg("-C")
        .arg(repo)
        .arg("--no-pager")
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .env("PAGER", "cat")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let Ok(mut child) = command.spawn() else {
        return GitOutcome::Failed;
    };
    let started = Instant::now();
    let timeout = StdDuration::from_secs(2);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(StdDuration::from_millis(10));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait_with_output();
                return GitOutcome::Failed;
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait_with_output();
                return GitOutcome::Failed;
            }
        }
    }

    match child.wait_with_output() {
        Ok(output) if output.status.success() => {
            GitOutcome::Success(String::from_utf8_lossy(&output.stdout).to_string())
        }
        Ok(_) => GitOutcome::NonZero,
        Err(_) => GitOutcome::Failed,
    }
}

fn git_binary() -> String {
    std::env::var("NABU_GIT").unwrap_or_else(|_| "git".to_string())
}

#[cfg(test)]
fn record_git_invocation(args: &[&str]) {
    git_invocations()
        .lock()
        .unwrap()
        .push(args.iter().map(|arg| (*arg).to_string()).collect());
}

#[cfg(not(test))]
fn record_git_invocation(_args: &[&str]) {}

#[cfg(test)]
pub(crate) fn git_invocations() -> &'static std::sync::Mutex<Vec<Vec<String>>> {
    static INVOCATIONS: std::sync::OnceLock<std::sync::Mutex<Vec<Vec<String>>>> =
        std::sync::OnceLock::new();
    INVOCATIONS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}
