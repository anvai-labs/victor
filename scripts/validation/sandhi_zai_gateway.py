#!/usr/bin/env python3
"""Manage a loopback Sandhi gateway backed only by a configured Victor ZAI account.

Run using the worktree .venv-codesign/bin/python. Provider credentials are read
from Victor's account/keyring and provisioned into Sandhi's OS-keyring vault.
State belongs in an ignored var/ directory, never in committed evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def private_write(path, text):
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(text)
    path.chmod(0o600)


def request(url, token=None, payload=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urlopen(Request(url, data=data, headers=headers), timeout=180) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except HTTPError as exc:
        # Do not print server response bodies: credential provisioning carries secrets.
        raise RuntimeError(f"Gateway returned HTTP {exc.code} for {url}") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "start", "status", "smoke", "stop"])
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--port", type=int, default=18788)
    parser.add_argument("--account", default="zai-glm53-openai")
    parser.add_argument("--model", default="glm-5.3")
    args = parser.parse_args()
    root = args.state_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    config_path = root / "gateway.json"
    config = (
        json.loads(config_path.read_text())
        if config_path.exists()
        else {
            "url": f"http://127.0.0.1:{args.port}",
            "model": args.model,
            "account": args.account,
            "binary": str(args.binary.resolve()) if args.binary else None,
        }
    )
    url = config["url"]
    admin_path = root / "admin-token"
    if not admin_path.exists():
        private_write(admin_path, secrets.token_urlsafe(32))
    token = admin_path.read_text().strip()
    if args.action in {"start", "setup"}:
        try:
            request(url + "/admin/version", token)
        except (URLError, ConnectionError):
            binary = config["binary"]
            if not binary or not Path(binary).is_file():
                raise ValueError("Supply --binary pointing to a built sandhi-proxy")
            env = {k: v for k, v in os.environ.items() if not k.startswith("SANDHI_")}
            env.update(
                SANDHI_BIND=url.removeprefix("http://"),
                SANDHI_PUBLIC_URL=url,
                SANDHI_STORE=str(root / "usage.db"),
                SANDHI_ADMIN_TOKEN=token,
                SANDHI_VAULT_BACKEND="keyring",
            )
            with (root / "proxy.log").open("a") as log:
                proc = subprocess.Popen(
                    [binary], env=env, stdout=log, stderr=log, start_new_session=True
                )
            (root / "proxy.pid").write_text(str(proc.pid))
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError("Gateway exited; inspect the local proxy.log")
                try:
                    request(url + "/admin/version", token)
                    break
                except (URLError, ConnectionError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("Gateway did not become ready")
        private_write(config_path, json.dumps(config, indent=2) + "\n")
    if args.action == "setup":
        from victor.config.accounts import AccountManager
        from victor.config.api_keys import get_api_key

        account = AccountManager().resolve_provider_config(account_name=config["account"])
        secret = account.get("api_key") or get_api_key("zai")
        if not secret:
            raise ValueError("Configured ZAI credential is unavailable")
        request(
            url + "/admin/keys",
            token,
            {
                "provider": "zai",
                "label": "victor-multiagent",
                "scheme": "bearer",
                "base_url": account["base_url"],
                "secret": secret,
            },
        )
        client_path = root / "client.json"
        if not client_path.exists():
            shared = request(
                url + "/admin/keys/share",
                token,
                {
                    "upstream": "zai:victor-multiagent",
                    "subject": "victor-local",
                    "group": "multiagent-validation",
                    "models": [config["model"]],
                },
            )
            private_write(
                client_path, json.dumps({"url": url, "virtual_key": shared["virtual_key"]})
            )
        client = json.loads(client_path.read_text())
        private_write(
            root / "client.env",
            f"export SANDHI_GATEWAY_URL='{url}'\n"
            f"export SANDHI_GATEWAY_VIRTUAL_KEY_ZAI='{client['virtual_key']}'\n",
        )
        print(
            json.dumps(
                {
                    "configured": True,
                    "url": url,
                    "provider": "zai",
                    "model": config["model"],
                    "state_dir": str(root),
                }
            )
        )
    elif args.action == "smoke":
        client = json.loads((root / "client.json").read_text())
        response = request(
            url + "/v1/chat/completions",
            client["virtual_key"],
            {
                "model": config["model"],
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 64,
                "stream": False,
            },
        )
        assert response.get("choices"), "Missing completion"
        print(
            json.dumps(
                {
                    "model": response.get("model"),
                    "usage": response.get("usage"),
                    "content": response["choices"][0]["message"].get("content"),
                }
            )
        )
    elif args.action == "status":
        version = request(url + "/admin/version", token)
        print(json.dumps({"url": url, "ready": True, "version": version}, indent=2))
    elif args.action == "stop":
        os.kill(int((root / "proxy.pid").read_text()), signal.SIGTERM)
        print("Stopped configured gateway")
    else:
        print(f"Gateway ready at {url}")


if __name__ == "__main__":
    main()
