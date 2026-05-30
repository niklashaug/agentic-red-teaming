from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen


DEFAULT_GITEA_URL = "http://localhost:3000"
DEFAULT_LOGSERVER_URL = "http://localhost:8000"
DEFAULT_OWNER = "research-admin"
DEFAULT_BOT_USER = "triage-bot"
DEFAULT_REPORTER_USER = "issue-reporter"
DEFAULT_REPO = "issue-triage-lab"
DEFAULT_TOKEN_FILE = Path(".runtime/gitea/token")
DEFAULT_ADMIN_TOKEN_FILE = Path(".runtime/gitea/admin-token")
DEFAULT_BOT_TOKEN_FILE = Path(".runtime/gitea/bot-token")
DEFAULT_REPORTER_TOKEN_FILE = Path(".runtime/gitea/reporter-token")
DEFAULT_TIMEOUT_SECONDS = 15
UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


class GiteaApiError(RuntimeError):
    def __init__(self, status: int, message: str, payload: Any | None = None) -> None:
        super().__init__(f"Gitea API returned HTTP {status}: {message}")
        self.status = status
        self.payload = payload


@dataclass(frozen=True)
class GiteaSettings:
    base_url: str
    token: str
    owner: str
    repo: str


TOKEN_FILES = {
    "admin": DEFAULT_ADMIN_TOKEN_FILE,
    "bot": DEFAULT_BOT_TOKEN_FILE,
    "reporter": DEFAULT_REPORTER_TOKEN_FILE,
}


def load_gitea_token(
    token_file: Path | None = None, *, role: str = "bot"
) -> str:
    env_key = f"GITEA_{role.upper()}_TOKEN"
    token = os.environ.get(env_key, "").strip()
    if token:
        return token

    token = os.environ.get("GITEA_TOKEN", "").strip()
    if token:
        return token

    if token_file is None:
        token_file = TOKEN_FILES.get(role, DEFAULT_TOKEN_FILE)

    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()

    if DEFAULT_TOKEN_FILE.exists():
        return DEFAULT_TOKEN_FILE.read_text(encoding="utf-8").strip()

    raise RuntimeError(
        f"No Gitea token configured for role '{role}'. "
        "Set GITEA_TOKEN or run scripts/bootstrap_gitea.py."
    )


def load_gitea_settings(
    *,
    base_url: str | None = None,
    token: str | None = None,
    owner: str | None = None,
    repo: str | None = None,
    role: str = "bot",
) -> GiteaSettings:
    return GiteaSettings(
        base_url=(base_url or os.environ.get("GITEA_URL") or DEFAULT_GITEA_URL).rstrip(
            "/"
        ),
        token=token or load_gitea_token(role=role),
        owner=owner or os.environ.get("GITEA_REPO_OWNER", DEFAULT_OWNER),
        repo=repo or os.environ.get("GITEA_REPO_NAME", DEFAULT_REPO),
    )


def decode_gitea_content_payload(payload: dict[str, Any]) -> str:
    encoded_content = payload.get("content")
    if not isinstance(encoded_content, str):
        raise ValueError("Gitea content payload does not contain a base64 string.")
    compact_content = "".join(encoded_content.split())
    return base64.b64decode(compact_content).decode("utf-8")


def _read_response(response: Any) -> Any:
    raw = response.read().decode("utf-8")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _request_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    expected_statuses: tuple[int, ...] = (200,),
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"token {token}"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            body = _read_response(response)
    except HTTPError as error:
        body = _read_response(error)
        if error.code in expected_statuses:
            return body
        raise GiteaApiError(error.code, str(error.reason), body) from error

    if status not in expected_statuses:
        raise GiteaApiError(status, "unexpected response status", body)
    return body


def _repo_path(path: str) -> str:
    return quote(path.strip("/"), safe="/")


class GiteaClient:
    def __init__(self, settings: GiteaSettings) -> None:
        self.settings = settings

    @property
    def api_url(self) -> str:
        return urljoin(f"{self.settings.base_url}/", "api/v1/")

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        expected_statuses: tuple[int, ...] = (200,),
    ) -> Any:
        return _request_json(
            method,
            urljoin(self.api_url, path.lstrip("/")),
            token=self.settings.token,
            payload=payload,
            expected_statuses=expected_statuses,
        )

    def delete_repo(self) -> None:
        self.request(
            "DELETE",
            f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}",
            expected_statuses=(204, 404),
        )

    def create_repo(self, *, private: bool = True) -> dict[str, Any]:
        return self.request(
            "POST",
            "user/repos",
            payload={
                "name": self.settings.repo,
                "description": "Reproducible issue-triage prompt-injection testbed.",
                "private": private,
                "auto_init": True,
                "default_branch": "main",
                "readme": "Default",
            },
            expected_statuses=(201,),
        )

    def add_collaborator(self, username: str, permission: str = "write") -> Any:
        return self.request(
            "PUT",
            (
                f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                f"/collaborators/{quote(username)}"
            ),
            payload={"permission": permission},
            expected_statuses=(204,),
        )

    def get_file_metadata(self, filepath: str) -> dict[str, Any] | None:
        try:
            return self.request(
                "GET",
                (
                    f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                    f"/contents/{_repo_path(filepath)}"
                ),
                expected_statuses=(200,),
            )
        except GiteaApiError as error:
            if error.status == 404:
                return None
            raise

    def write_file(
        self,
        filepath: str,
        content: str,
        *,
        message: str,
        branch: str = "main",
    ) -> dict[str, Any]:
        encoded_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
        metadata = self.get_file_metadata(filepath)
        payload: dict[str, Any] = {
            "branch": branch,
            "content": encoded_content,
            "message": message,
        }
        method = "POST"
        if metadata is not None:
            method = "PUT"
            payload["sha"] = metadata["sha"]

        return self.request(
            method,
            (
                f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                f"/contents/{_repo_path(filepath)}"
            ),
            payload=payload,
            expected_statuses=(200, 201),
        )

    def read_file(self, filepath: str) -> str:
        payload = self.request(
            "GET",
            (
                f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                f"/contents/{_repo_path(filepath)}"
            ),
            expected_statuses=(200,),
        )
        if isinstance(payload, list):
            return json.dumps(payload, ensure_ascii=True)
        return decode_gitea_content_payload(payload)

    def create_label(self, name: str, color: str) -> dict[str, Any] | None:
        try:
            return self.request(
                "POST",
                (
                    f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                    "/labels"
                ),
                payload={"name": name, "color": color.lstrip("#")},
                expected_statuses=(201,),
            )
        except GiteaApiError as error:
            if error.status in {409, 422}:
                return None
            raise

    def ensure_labels(self, labels: list[str]) -> None:
        for label in labels:
            color_seed = sum(ord(character) for character in label)
            color = f"{(color_seed * 2654435761) & 0xFFFFFF:06x}"
            self.create_label(label, color)

    def create_issue(self, title: str, body: str) -> dict[str, Any]:
        return self.request(
            "POST",
            f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}/issues",
            payload={"title": title, "body": body},
            expected_statuses=(201,),
        )

    def get_issue(self, issue_id: int) -> dict[str, Any]:
        return self.request(
            "GET",
            (
                f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                f"/issues/{issue_id}"
            ),
            expected_statuses=(200,),
        )

    def list_issue_comments(self, issue_id: int) -> list[dict[str, Any]]:
        comments = self.request(
            "GET",
            (
                f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                f"/issues/{issue_id}/comments"
            ),
            expected_statuses=(200,),
        )
        return comments if isinstance(comments, list) else []

    def list_issues(self) -> list[dict[str, Any]]:
        issues = self.request(
            "GET",
            f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}/issues",
            expected_statuses=(200,),
        )
        return issues if isinstance(issues, list) else []

    def search_similar_issues(self, query: str) -> list[dict[str, Any]]:
        terms = [term.casefold() for term in re.findall(r"\w+", query) if len(term) > 2]
        matches = []
        for issue in self.list_issues():
            haystack = f"{issue.get('title', '')} {issue.get('body', '')}".casefold()
            if not terms or any(term in haystack for term in terms):
                matches.append(
                    {
                        "number": issue.get("number"),
                        "title": issue.get("title"),
                        "state": issue.get("state"),
                        "body": issue.get("body", "")[:240],
                    }
                )
        return matches[:5]

    def post_issue_comment(self, issue_id: int, text: str) -> dict[str, Any]:
        return self.request(
            "POST",
            (
                f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                f"/issues/{issue_id}/comments"
            ),
            payload={"body": text},
            expected_statuses=(201,),
        )

    def manage_issue_labels(
        self, issue_id: int, labels: list[str]
    ) -> list[dict[str, Any]]:
        self.ensure_labels(labels)
        response = self.request(
            "PUT",
            (
                f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                f"/issues/{issue_id}/labels"
            ),
            payload={"labels": labels},
            expected_statuses=(200,),
        )
        return response if isinstance(response, list) else []


def gitea_client(
    role: str = "bot", *, owner: str | None = None, repo: str | None = None
) -> GiteaClient:
    return GiteaClient(load_gitea_settings(owner=owner, repo=repo, role=role))


def default_gitea_client(
    *, owner: str | None = None, repo: str | None = None
) -> GiteaClient:
    return gitea_client("bot", owner=owner, repo=repo)


def normalize_trace_id(trace_id: Any) -> str:
    if isinstance(trace_id, dict):
        for key in ("trace_id", "trace", "id"):
            if key in trace_id:
                return normalize_trace_id(trace_id[key])

    raw_trace_id = str(trace_id).strip()
    uuid_match = UUID_PATTERN.search(raw_trace_id)
    if uuid_match:
        return uuid_match.group(0).lower()

    match = re.search(
        r"\b(?:trace[_ -]?id|trace|id)\s*[=:]?\s*([A-Za-z0-9-]+)\b",
        raw_trace_id,
    )
    if match:
        value = match.group(1)
        if value.isdigit():
            return value

    fallback_match = re.search(r"\b(\d+)\b", raw_trace_id)
    if fallback_match:
        return fallback_match.group(1)

    raise ValueError(f"Could not extract a trace_id from: {trace_id!r}")


def normalize_log_trace_id(trace_id: Any) -> str:
    return normalize_trace_id(trace_id)


def query_logs(trace_id: Any) -> dict[str, Any]:
    base_url = os.environ.get("LOGSERVER_URL", DEFAULT_LOGSERVER_URL).rstrip("/")
    normalized_trace_id = normalize_trace_id(trace_id)
    return _request_json(
        "GET",
        f"{base_url}/api/logs/{quote(normalized_trace_id)}",
        expected_statuses=(200,),
    )


def query_monitoring(query: str) -> dict[str, Any]:
    base_url = os.environ.get("LOGSERVER_URL", DEFAULT_LOGSERVER_URL).rstrip("/")
    encoded_query = urlencode({"query": query})
    return _request_json(
        "GET",
        f"{base_url}/api/monitoring?{encoded_query}",
        expected_statuses=(200,),
    )


def read_repo_file(
    filepath: str, *, owner: str | None = None, repo: str | None = None
) -> str:
    return gitea_client("bot", owner=owner, repo=repo).read_file(filepath)


def search_repo_files(
    filepath: str, *, owner: str | None = None, repo: str | None = None
) -> str:
    return read_repo_file(filepath, owner=owner, repo=repo)


def post_issue_comment(
    issue_id: int, text: str, *, owner: str | None = None, repo: str | None = None
) -> dict[str, Any]:
    return gitea_client("bot", owner=owner, repo=repo).post_issue_comment(
        issue_id, text
    )


def set_issue_labels(
    issue_id: int,
    labels: list[str],
    *,
    owner: str | None = None,
    repo: str | None = None,
) -> list[dict[str, Any]]:
    return gitea_client("bot", owner=owner, repo=repo).manage_issue_labels(
        issue_id, labels
    )


def manage_issue_labels(
    issue_id: int,
    labels: list[str],
    *,
    owner: str | None = None,
    repo: str | None = None,
) -> list[dict[str, Any]]:
    return set_issue_labels(issue_id, labels, owner=owner, repo=repo)
