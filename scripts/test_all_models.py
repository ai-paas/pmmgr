#!/usr/bin/env python3
"""Query every chat model registered in pmmgr over the HTTP API.

Judges response generation only: an HTTP 200 is not enough, because
/model/chat/{id_or_path}/{query} returns an empty string with 200 when the
model produces nothing. A model passes only when every pass returns a
non-empty, non-whitespace body.

Usage:
    python3 scripts/test_all_models.py \
        --base-url http://101.202.128.10:8001 \
        --query "Tell me about RAG briefly" --passes 2 \
        --json-out results/2026-10-26_rag_generation_test.json \
        --md-out results/2026-10-26_rag_generation_test.md
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_QUERY = "Hello"


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def list_models(base_url, timeout):
    _, data = get_json(f"{base_url}/model/info/list", timeout)
    return [m["id_or_path"] for m in data]


def chat_url(base_url, id_or_path, query):
    model = urllib.parse.quote(id_or_path, safe="")
    q = urllib.parse.quote(query, safe="")
    return f"{base_url}/model/chat/{model}/{q}"


def call_once(url, timeout):
    started = time.time()
    status = None
    raw = None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8")
        body = json.loads(raw)
        if not isinstance(body, str):
            return {
                "http": status,
                "secs": round(time.time() - started, 2),
                "chars": 0,
                "ok": False,
                "error": f"unexpected response type: {type(body).__name__}",
            }
        return {
            "http": status,
            "secs": round(time.time() - started, 2),
            "chars": len(body),
            "ok": bool(body.strip()),
            "answer": body,
        }
    except Exception as e:  # network errors, timeouts, non-JSON error pages
        err = raw if raw is not None else f"{type(e).__name__}: {e}"
        return {
            "http": status,
            "secs": round(time.time() - started, 2),
            "chars": 0,
            "ok": False,
            "error": str(err)[:2000],
        }


def write_markdown(path, base_url, query, passes, results):
    rows = []
    for i, (model, runs) in enumerate(results, 1):
        cells = []
        for r in runs:
            if r.get("answer") is not None:
                cells.append(f"{r['secs']}s / {r['chars']}자")
            else:
                cells.append(f"{r['secs']}s / 실패")
        verdict = "✅" if all(r["ok"] for r in runs) else "❌"
        rows.append(f"| {i} | `{model}` | " + " | ".join(cells) + f" | {verdict} |")

    all_runs = [r for _, runs in results for r in runs]
    secs = [r["secs"] for r in all_runs]
    chars = [r["chars"] for r in all_runs]
    ok_models = sum(1 for _, runs in results if all(r["ok"] for r in runs))

    header = (
        f"# Chat model generation test\n\n"
        f"- Base URL: `{base_url}`\n"
        f"- Query: `{query}`\n"
        f"- Passes per model: {passes}\n"
        f"- Models: {ok_models}/{len(results)} generated a non-empty answer\n"
        f"- Calls: {len(all_runs)} "
        f"(total {round(sum(secs), 1)}s, median {statistics.median(secs)}s, "
        f"min {min(secs)}s, max {max(secs)}s)\n"
        f"- Answer size: median {int(statistics.median(chars))} chars, "
        f"min {min(chars)}, max {max(chars)}\n"
        f"- Verdict: HTTP 200 with an empty body counts as a failure.\n"
        f"- Answer bodies are not stored in this report; pass --json-out for raw output.\n\n"
        f"| # | model | "
        + " | ".join(f"run{n + 1}" for n in range(passes))
        + " | 생성 |\n"
        f"|---|-------|" + "|".join("---" for _ in range(passes)) + "|---|\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-request timeout in seconds")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    models = list_models(base_url, args.timeout)
    print(f"models={len(models)} passes={args.passes}", flush=True)

    results = []
    failed = 0
    for i, model in enumerate(models, 1):
        runs = []
        for p in range(1, args.passes + 1):
            run = call_once(chat_url(base_url, model, args.query), args.timeout)
            runs.append(run)
            print(
                f"[{i}/{len(models)}] {model} pass{p}: ok={run['ok']} "
                f"http={run['http']} secs={run['secs']} chars={run['chars']} "
                f"err={str(run.get('error'))[:160]}",
                flush=True,
            )
            if not run["ok"]:
                failed += 1
        results.append((model, runs))

    if args.json_out:
        payload = {
            "base_url": base_url,
            "query": args.query,
            "passes": args.passes,
            "models": [
                {"id_or_path": model, "runs": runs} for model, runs in results
            ],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
    if args.md_out:
        write_markdown(args.md_out, base_url, args.query, args.passes, results)

    ok_models = sum(1 for _, runs in results if all(r["ok"] for r in runs))
    print(f"DONE {ok_models}/{len(results)} models generated answers, "
          f"{failed} failed calls", flush=True)
    return 0 if ok_models == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
