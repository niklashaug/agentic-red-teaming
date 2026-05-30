from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_GITEA_URL = "http://localhost:3000"
DEFAULT_ADMIN_USER = "research-admin"
DEFAULT_ADMIN_PASSWORD = "research-password"
DEFAULT_ADMIN_EMAIL = "research-admin@example.invalid"
DEFAULT_TOKEN_FILE = Path(".runtime/gitea/token")
GITEA_CONFIG = "/etc/gitea/app.ini"


def run_command(
    command: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def wait_for_gitea(base_url: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    version_url = f"{base_url.rstrip('/')}/api/v1/version"
    while time.monotonic() < deadline:
        try:
            with urlopen(version_url, timeout=2) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(1)
    raise TimeoutError(f"Gitea did not become ready at {version_url}.")


def gitea_cli(
    args: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "gitea",
            "gitea",
            "--config",
            GITEA_CONFIG,
            *args,
        ],
        check=check,
    )


def ensure_admin_user(username: str, password: str, email: str) -> None:
    create = gitea_cli(
        [
            "admin",
            "user",
            "create",
            "--username",
            username,
            "--password",
            password,
            "--email",
            email,
            "--admin",
            "--must-change-password=false",
        ],
        check=False,
    )
    if create.returncode == 0:
        return

    combined_output = f"{create.stdout}\n{create.stderr}".lower()
    if "already exists" not in combined_output and "duplicate" not in combined_output:
        raise RuntimeError(
            "Failed to create Gitea admin user.\n"
            f"stdout:\n{create.stdout}\nstderr:\n{create.stderr}"
        )

    gitea_cli(
        [
            "admin",
            "user",
            "change-password",
            "--username",
            username,
            "--password",
            password,
            "--must-change-password=false",
        ]
    )


def parse_token(output: str) -> str:
    stripped = output.strip()
    if re.fullmatch(r"[A-Za-z0-9_\-]{20,}", stripped):
        return stripped

    token_matches = re.findall(r"[A-Za-z0-9_\-]{40,}", output)
    if token_matches:
        return token_matches[-1]
    raise RuntimeError(f"Could not parse generated token from output:\n{output}")


def generate_access_token(username: str, token_name: str) -> str:
    token = gitea_cli(
        [
            "admin",
            "user",
            "generate-access-token",
            "--username",
            username,
            "--token-name",
            token_name,
            "--scopes",
            "all",
            "--raw",
        ],
        check=False,
    )
    if token.returncode == 0:
        return parse_token(token.stdout)

    combined_output = f"{token.stdout}\n{token.stderr}"
    if "flag provided but not defined" not in combined_output:
        raise RuntimeError(
            "Failed to generate Gitea access token.\n"
            f"stdout:\n{token.stdout}\nstderr:\n{token.stderr}"
        )

    fallback = gitea_cli(
        [
            "admin",
            "user",
            "generate-access-token",
            "--username",
            username,
            "--token-name",
            token_name,
            "--scopes",
            "all",
        ]
    )
    return parse_token(fallback.stdout)


def write_token(token: str, token_file: Path) -> None:
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(token + "\n", encoding="utf-8")
    token_file.chmod(0o600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Gitea admin user and PAT.")
    parser.add_argument(
        "--gitea-url", default=os.environ.get("GITEA_URL", DEFAULT_GITEA_URL)
    )
    parser.add_argument(
        "--username", default=os.environ.get("GITEA_ADMIN_USER", DEFAULT_ADMIN_USER)
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("GITEA_ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD),
    )
    parser.add_argument(
        "--email", default=os.environ.get("GITEA_ADMIN_EMAIL", DEFAULT_ADMIN_EMAIL)
    )
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wait_for_gitea(args.gitea_url, args.timeout)
    ensure_admin_user(args.username, args.password, args.email)
    token_name = f"agent-token-{int(time.time())}"
    token = generate_access_token(args.username, token_name)
    write_token(token, args.token_file)
    print(f"Wrote Gitea token for {args.username} to {args.token_file}")


if __name__ == "__main__":
    main()
