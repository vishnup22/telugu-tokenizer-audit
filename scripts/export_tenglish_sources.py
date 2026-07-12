#!/usr/bin/env python3
"""Export public Telugu/Tenglish sources to JSONL for corpus filtering.

This script writes one JSON object per line with at least:
- text
- source
- source_id
- url
- platform

Supported exporters:
- YouTube video comments via the official YouTube Data API
- Reddit public post/comment JSON via public subreddit post JSON endpoints
- X recent search via the official X API v2 recent search endpoint

The output is meant to be fed into:

    python scripts/collect_data.py --include-tenglish --tenglish-sources-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def _http_get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace") if err.fp else ""
        raise RuntimeError(
            f"HTTP {err.code} from {url}. Response body: {body[:1000]}"
        ) from err


def _write_jsonl(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def export_youtube_comments(
    video_id: str,
    output_path: Path,
    api_key: str,
    verbose: bool = True,
) -> int:
    rows: list[dict] = []
    page_token: str | None = None
    page = 0

    while True:
        page += 1
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": "100",
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        url = "https://www.googleapis.com/youtube/v3/commentThreads?" + urllib.parse.urlencode(params)

        if verbose:
            print(f"[youtube] page {page}: requesting comments for {video_id}", flush=True)
        payload = _http_get_json(url)
        items = payload.get("items", [])
        if verbose:
            print(f"[youtube] page {page}: received {len(items)} threads", flush=True)

        for item in items:
            snippet = item.get("snippet", {})
            top = snippet.get("topLevelComment", {}).get("snippet", {})
            text = (top.get("textOriginal") or top.get("textDisplay") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "source": "youtube",
                    "source_id": item.get("id", ""),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "platform": "youtube",
                    "video_id": video_id,
                    "author": top.get("authorDisplayName", ""),
                    "published_at": top.get("publishedAt", ""),
                }
            )

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

        time.sleep(0.1)

    if verbose:
        print(f"[youtube] writing {len(rows)} comments to {output_path}", flush=True)
    return _write_jsonl(rows, output_path)


def export_reddit_thread(subreddit: str, post_id: str, output_path: Path) -> int:
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit=500&depth=10&sort=top"
    headers = {"User-Agent": "telugu-tokenizer-audit/1.0"}
    payload = _http_get_json(url, headers=headers)

    rows: list[dict] = []
    if not isinstance(payload, list) or len(payload) < 2:
        return _write_jsonl(rows, output_path)

    post = payload[0].get("data", {}).get("children", [{}])[0].get("data", {})
    thread_url = f"https://www.reddit.com{post.get('permalink', f'/r/{subreddit}/comments/{post_id}/')}"

    def walk(children: list[dict]) -> None:
        for child in children:
            kind = child.get("kind")
            data = child.get("data", {})
            if kind == "t1":
                body = (data.get("body") or "").strip()
                if body:
                    rows.append(
                        {
                            "text": body,
                            "source": "reddit",
                            "source_id": data.get("id", ""),
                            "url": thread_url,
                            "platform": "reddit",
                            "subreddit": subreddit,
                            "post_id": post_id,
                            "author": data.get("author", ""),
                            "score": data.get("score", 0),
                        }
                    )
                replies = data.get("replies", {})
                if isinstance(replies, dict):
                    walk(replies.get("data", {}).get("children", []))

    walk(payload[1].get("data", {}).get("children", []))
    return _write_jsonl(rows, output_path)


def export_x_recent_search(query: str, output_path: Path, bearer_token: str, max_results: int = 100) -> int:
    headers = {"Authorization": f"Bearer {bearer_token}"}
    rows: list[dict] = []
    next_token: str | None = None

    while True:
        params = {
            "query": query,
            "max_results": str(min(max_results, 100)),
            "tweet.fields": "created_at,lang,public_metrics,author_id,conversation_id",
        }
        if next_token:
            params["next_token"] = next_token
        url = "https://api.x.com/2/tweets/search/recent?" + urllib.parse.urlencode(params)
        payload = _http_get_json(url, headers=headers)

        for item in payload.get("data", []):
            text = (item.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "source": "x",
                    "source_id": item.get("id", ""),
                    "url": f"https://x.com/i/web/status/{item.get('id', '')}",
                    "platform": "x",
                    "author_id": item.get("author_id", ""),
                    "conversation_id": item.get("conversation_id", ""),
                    "created_at": item.get("created_at", ""),
                    "lang": item.get("lang", ""),
                }
            )

        meta = payload.get("meta", {})
        next_token = meta.get("next_token")
        if not next_token:
            break

        time.sleep(0.5)

    return _write_jsonl(rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)

    yt = sub.add_parser("youtube", help="Export YouTube comments for one video")
    yt.add_argument("--video-id", required=True)
    yt.add_argument("--output", required=True)
    yt.add_argument("--api-key", default=os.getenv("YOUTUBE_API_KEY"))

    rd = sub.add_parser("reddit", help="Export Reddit comments from a public thread")
    rd.add_argument("--subreddit", required=True)
    rd.add_argument("--post-id", required=True)
    rd.add_argument("--output", required=True)

    x = sub.add_parser("x", help="Export X posts via recent search")
    x.add_argument("--query", required=True)
    x.add_argument("--output", required=True)
    x.add_argument("--bearer-token", default=os.getenv("X_BEARER_TOKEN"))
    x.add_argument("--max-results", type=int, default=100)

    args = parser.parse_args()

    if args.kind == "youtube":
        if not args.api_key:
            raise SystemExit("Missing YouTube API key. Set YOUTUBE_API_KEY or pass --api-key.")
        n = export_youtube_comments(args.video_id, Path(args.output), args.api_key, verbose=True)
    elif args.kind == "reddit":
        n = export_reddit_thread(args.subreddit, args.post_id, Path(args.output))
    elif args.kind == "x":
        if not args.bearer_token:
            raise SystemExit("Missing X bearer token. Set X_BEARER_TOKEN or pass --bearer-token.")
        n = export_x_recent_search(args.query, Path(args.output), args.bearer_token, args.max_results)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown exporter kind: {args.kind}")

    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()


