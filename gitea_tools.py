from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import httpx


DEFAULT_GITEA_URL = "http://localhost:3000"
DEFAULT_LOGSERVER_URL = "http://localhost:8000"
DEFAULT_OWNER = "research-admin"
DEFAULT_BOT_USER = "triage-bot"
DEFAULT_REPORTER_USER = "issue-reporter"
DEFAULT_REPO = "issue-triage-lab"
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
        try:
            token_file = TOKEN_FILES[role]
        except KeyError as error:
            raise RuntimeError(
                f"Unknown Gitea token role '{role}'. "
                f"Known roles: {', '.join(sorted(TOKEN_FILES))}."
            ) from error

    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()

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


def _read_response(response: httpx.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text


def _request_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    expected_statuses: tuple[int, ...] = (200,),
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"token {token}"

    request_kwargs: dict[str, Any] = {"params": params}
    if payload is not None:
        request_kwargs["json"] = payload

    with httpx.Client(headers=headers, timeout=timeout) as client:
        response = client.request(method, url, **request_kwargs)

    body = _read_response(response)
    if response.status_code not in expected_statuses:
        message = response.reason_phrase or "unexpected response status"
        raise GiteaApiError(response.status_code, message, body)
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

    def get_current_user(self) -> dict[str, Any]:
        return self.request("GET", "user")

    def migrate_repo(
        self,
        clone_addr: str,
        *,
        private: bool = True,
        service: str = "git",
    ) -> dict[str, Any]:
        user = self.get_current_user()
        return self.request(
            "POST",
            "repos/migrate",
            payload={
                "clone_addr": clone_addr,
                "repo_name": self.settings.repo,
                "uid": user["id"],
                "private": private,
                "service": service,
                "mirror": False,
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
            hex_color = color if color.startswith("#") else f"#{color}"
            return self.request(
                "POST",
                (
                    f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                    "/labels"
                ),
                payload={"name": name, "color": hex_color},
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

    def list_labels(self) -> list[dict[str, Any]]:
        result = self.request(
            "GET",
            f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}/labels",
            expected_statuses=(200,),
        )
        return result if isinstance(result, list) else []

    def manage_issue_labels(
        self, issue_id: int, labels: list[str]
    ) -> list[dict[str, Any]]:
        name_to_id = {
            lbl["name"]: lbl["id"]
            for lbl in self.list_labels()
            if isinstance(lbl.get("name"), str) and isinstance(lbl.get("id"), int)
        }
        label_ids = [name_to_id[name] for name in labels if name in name_to_id]
        response = self.request(
            "PUT",
            (
                f"repos/{quote(self.settings.owner)}/{quote(self.settings.repo)}"
                f"/issues/{issue_id}/labels"
            ),
            payload={"labels": label_ids},
            expected_statuses=(200,),
        )
        return response if isinstance(response, list) else []


def gitea_client(
    role: str = "bot", *, owner: str | None = None, repo: str | None = None
) -> GiteaClient:
    return GiteaClient(load_gitea_settings(owner=owner, repo=repo, role=role))


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
    return _request_json(
        "GET",
        f"{base_url}/api/monitoring",
        params={"query": query},
        expected_statuses=(200,),
    )
