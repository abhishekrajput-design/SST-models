#!/usr/bin/env python
"""Scrape Audiofy calls using a token recovered from the local browser profile.

The token is used in memory only and is never printed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = REPO_ROOT / "call_processor" / "tools" / "legacy"
sys.path.insert(0, str(LEGACY_DIR))

from scrape_audiofy import AudiofyClient, save_and_cut, scrape  # noqa: E402


JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


def chrome_user_data_dir() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise RuntimeError("LOCALAPPDATA is not set")
    return Path(local) / "Google" / "Chrome" / "User Data"


def tokens_from_text(text: str) -> list[str]:
    seen = set()
    out = []
    for match in JWT_RE.findall(text or ""):
        if match not in seen:
            seen.add(match)
            out.append(match)
    return out


def tokens_from_leveldb(user_data: Path, profiles: list[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen = set()
    for profile in profiles:
        leveldb = user_data / profile / "Local Storage" / "leveldb"
        if not leveldb.exists():
            continue
        for path in sorted(leveldb.glob("*")):
            if path.suffix.lower() not in {".ldb", ".log"}:
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            text = raw.decode("latin-1", errors="ignore")
            if "audiofy" not in text.lower() and "access" not in text.lower():
                continue
            for token in tokens_from_text(text):
                if token in seen:
                    continue
                seen.add(token)
                found.append((profile, token))
    return found


def tokens_from_playwright(user_data: Path, profile: str, headless: bool) -> list[tuple[str, str]]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(f"[profile] Playwright unavailable: {exc}")
        return []

    tokens: list[tuple[str, str]] = []
    try:
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                str(user_data),
                channel="chrome",
                headless=headless,
                args=[f"--profile-directory={profile}"],
            )
            page = context.new_page()
            page.goto("https://beta.audiofy.co.uk", wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2500)
            values = page.evaluate(
                """() => {
                    const out = [];
                    for (const store of [localStorage, sessionStorage]) {
                      for (let i = 0; i < store.length; i++) {
                        const key = store.key(i);
                        out.push(`${key}=${store.getItem(key)}`);
                      }
                    }
                    out.push(document.cookie || "");
                    return out.join("\\n");
                }"""
            )
            for token in tokens_from_text(values):
                tokens.append((profile, token))
            context.close()
    except Exception as exc:
        print(f"[profile] Playwright profile read failed for {profile}: {exc}")
    return tokens


def valid_client(candidates: list[tuple[str, str]]) -> AudiofyClient | None:
    for source, token in candidates:
        client = AudiofyClient()
        try:
            client.use_token(token)
            users = client.get_all_users()
            if users:
                print(f"[auth] valid Audiofy token from {source}; users={len(users)}")
                return client
        except Exception as exc:
            print(f"[auth] token from {source} rejected: {type(exc).__name__}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=25)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--skip-playwright", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--debug-response", action="store_true")
    args = parser.parse_args()
    if not args.profile:
        args.profile = ["Default"]

    user_data = chrome_user_data_dir()
    candidates: list[tuple[str, str]] = []
    if not args.skip_playwright:
        for profile in args.profile:
            candidates.extend(tokens_from_playwright(user_data, profile, args.headless))
    candidates.extend(tokens_from_leveldb(user_data, args.profile))
    if not candidates:
        print("[auth] no Audiofy JWT candidates found in requested profile(s)")
        return 1

    client = valid_client(candidates)
    if client is None:
        print("[auth] no valid Audiofy token found")
        return 1

    scrape_args = SimpleNamespace(
        agent=args.agent,
        pages=args.pages,
        max_calls=args.max_calls,
        days=args.days,
        debug_response=args.debug_response,
        no_audio=args.no_audio,
    )
    calls = scrape(client, scrape_args)
    print(f"[scrape] collected={len(calls)}")
    if calls:
        save_and_cut(client, calls, skip_audio=args.no_audio)
    return 0 if calls else 2


if __name__ == "__main__":
    raise SystemExit(main())
