"""Run one live baseline or merged debate and save the evidence.

This does not change prompts or insert findings. It records the prompts,
tool activity, final state, elapsed time, and LangSmith trace when a real
API key is configured.

From the lab5 MAS directory:

    python eval_runner.py --check-ready
    python eval_runner.py --variant merged --run-id merged-01
    python eval_runner.py --variant baseline --run-id baseline-01

A run is refused when Ollama, qwen2.5:7b, or a non-placeholder Tavily key
is missing. LangSmith tracing is enabled only when a non-placeholder
LANGSMITH_API_KEY is present, and the runner then checks that a root trace
for this run id was stored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import signal
import sys
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parent
MODEL_NAME = "qwen2.5:7b"
LANGSMITH_PROJECT = "is590r-lab5-debate"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
PACKAGES = (
    "langchain",
    "langchain-core",
    "langchain-ollama",
    "langchain-tavily",
    "langgraph",
    "langsmith",
    "python-dotenv",
    "typing_extensions",
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _load_local_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(LAB_DIR / ".env")


def _key_status(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        return "missing"
    lowered = value.lower()
    if "yourkey" in lowered or value.endswith("placeholder") or "tvly-offline" in lowered:
        return "placeholder"
    return "present"


def _package_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode())
    except Exception as exc:
        return [f"unavailable: {exc.__class__.__name__}"]
    return [item.get("name", "") for item in payload.get("models", [])]


def readiness() -> dict:
    _load_local_env()
    models = _ollama_models()
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": _package_versions(),
        "ollama_models": models,
        "model_ready": MODEL_NAME in models,
        "tavily": _key_status("TAVILY_API_KEY"),
        "langsmith": _key_status("LANGSMITH_API_KEY"),
    }
    report["live_run_ready"] = (
        report["model_ready"]
        and report["tavily"] == "present"
        and all(version != "not installed" for version in report["packages"].values())
    )
    report["tracing_ready"] = report["langsmith"] == "present"
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _append_jsonl(path: Path, record: dict) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _trace_messages(messages) -> list[dict]:
    traced = []
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, str) and len(content) > 50000:
            content = content[:50000] + "\n...[truncated by tracer]"
        traced.append({
            "type": getattr(message, "type", None),
            "text": getattr(message, "text", None),
            "content": content,
            "tool_calls": getattr(message, "tool_calls", None) or [],
            "usage_metadata": getattr(message, "usage_metadata", None),
            "response_metadata": getattr(message, "response_metadata", None),
            "id": getattr(message, "id", None),
        })
    return traced


def _install_baseline_trace(module, trace_path: Path) -> None:
    """Record baseline prompts without changing the advocate or moderator text."""

    def wrap_agent(agent, participant):
        original = agent.invoke

        def invoke(payload, *args, **kwargs):
            content = payload["messages"][0]["content"] if isinstance(payload, dict) else str(payload)
            result = original(payload, *args, **kwargs)
            messages = result["messages"] if isinstance(result, dict) else [result]
            _append_jsonl(trace_path, {
                "event": "participant",
                "participant": participant,
                "input": content,
                "messages": _trace_messages(messages),
            })
            return result

        agent.invoke = invoke

    wrap_agent(module.build_advocate, "build_advocate")
    wrap_agent(module.buy_advocate, "buy_advocate")
    # ChatOllama is a pydantic model, so its invoke method cannot be replaced.
    # Wrap the moderator node and recompile the same edges instead.
    original_moderator = module.moderator

    def traced_moderator(state):
        recorded_input = (
            f"Case:\n{state['case']}\n\n"
            f"BUILD side's final argument:\n{state['position_a']}\n\n"
            f"BUY side's final argument:\n{state['position_b']}"
        )
        result = original_moderator(state)
        _append_jsonl(trace_path, {
            "event": "participant",
            "participant": "moderator",
            "input": recorded_input,
            "messages": [{
                "type": "ai",
                "text": result.get("resolution"),
                "content": result.get("resolution"),
                "tool_calls": [],
            }],
        })
        return result

    from langgraph.graph import END, START, StateGraph
    graph = StateGraph(module.DebateState)
    graph.add_node("agent_position_a", module.agent_position_a)
    graph.add_node("agent_position_b", module.agent_position_b)
    graph.add_node("moderator", traced_moderator)
    graph.add_edge(START, "agent_position_a")
    graph.add_edge("agent_position_a", "agent_position_b")
    graph.add_conditional_edges(
        "agent_position_b", module.continue_debate, ["agent_position_a", "moderator"]
    )
    graph.add_edge("moderator", END)
    module.debate_graph = graph.compile()


def _load_variant(variant: str):
    if variant == "merged":
        import debateMAS as module
        source = LAB_DIR / "debateMAS.py"
        return module, source
    source = LAB_DIR / "baseline" / "instructor_debateMAS_runnable.py"
    spec = importlib.util.spec_from_file_location("instructor_debate_runnable", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def _public_url_status(url: str) -> int | None:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return None


def _share_root_run(run, client) -> str | None:
    # Prefer the SDK method that returns the public URL directly.
    try:
        return client.share_run(run.id)
    except Exception:
        response = client.runs.share.create(
            str(run.id),
            session_id=str(run.session_id),
            trace_id=str(run.trace_id or run.id),
        )
        token = getattr(response, "share_token", None)
        if not token:
            return None
        return f"{client._host_url}/public/{token}/r"


def _verify_langsmith(run_id: str, started: datetime) -> dict:
    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langsmith import Client

    wait_for_all_tracers()
    client = Client()
    runs = list(client.list_runs(
        project_name=LANGSMITH_PROJECT,
        start_time=started - timedelta(seconds=30),
        is_root=True,
        limit=50,
    ))
    matched = []
    for run in runs:
        extra = getattr(run, "extra", None) or {}
        metadata_value = extra.get("metadata", {}) if isinstance(extra, dict) else {}
        if run.name == run_id or metadata_value.get("run_id") == run_id:
            matched.append(run)
    if not matched and runs:
        # A trace can be stored under the graph name. Keep only runs that
        # started after this process began and carry this run id in metadata
        # or tags. If nothing matches, report the candidates without sharing
        # an unrelated trace.
        matched = []
    result = {
        "project": LANGSMITH_PROJECT,
        "endpoint": LANGSMITH_ENDPOINT,
        "root_runs_seen": len(runs),
        "matched_run_ids": [str(run.id) for run in matched],
        "public_urls": [],
        "verified": False,
    }
    for run in matched:
        blob = json.dumps(
            {"inputs": getattr(run, "inputs", None), "outputs": getattr(run, "outputs", None)},
            default=str,
        )
        if "tvly-" in blob or "lsv2_" in blob or "TAVILY_API_KEY" in blob:
            result["share_blocked"] = "trace contained credential-like text"
            continue
        url = _share_root_run(run, client)
        if not url:
            continue
        status = _public_url_status(url)
        result["public_urls"].append({"run_id": str(run.id), "url": url, "unauthenticated_status": status})
    result["verified"] = any(item["unauthenticated_status"] == 200 for item in result["public_urls"])
    return result


def _summarize_trace(trace_path: Path) -> dict:
    participants = []
    tool_calls = []
    usage = []
    if not trace_path.exists():
        return {"participants": participants, "tool_calls": tool_calls, "usage": usage}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "participant":
            continue
        participants.append(record.get("participant"))
        for message in record.get("messages") or []:
            if message.get("usage_metadata"):
                usage.append(message["usage_metadata"])
            for call in message.get("tool_calls") or []:
                tool_calls.append({
                    "participant": record.get("participant"),
                    "name": call.get("name"),
                    "args": call.get("args"),
                })
    return {"participants": participants, "tool_calls": tool_calls, "usage": usage}


def run_variant(variant: str, run_id: str, evidence_root: Path, timeout_seconds: int) -> dict:
    report = readiness()
    if not report["live_run_ready"]:
        raise SystemExit(
            "Live run refused. "
            f"model_ready={report['model_ready']} tavily={report['tavily']} "
            f"packages={report['packages']}"
        )
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
    if report["tracing_ready"]:
        os.environ["LANGSMITH_TRACING"] = "true"
    else:
        os.environ.pop("LANGSMITH_TRACING", None)

    run_dir = evidence_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    os.environ["DEBATE_TRACE_PATH"] = str(trace_path)
    os.environ["DEBATE_RUN_ID"] = run_id

    module, source = _load_variant(variant)
    if variant == "baseline":
        _install_baseline_trace(module, trace_path)

    console_path = run_dir / "console.txt"
    started = datetime.now(timezone.utc)
    status = "completed"
    error_text = ""

    def _timeout(_signum, _frame):
        raise TimeoutError(f"run exceeded {timeout_seconds} seconds")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(timeout_seconds)
    stdout = sys.stdout
    with console_path.open("w", encoding="utf-8") as console:
        sys.stdout = Tee(stdout, console)
        try:
            print(f"[run] {run_id} variant={variant}")
            result = module.debate_graph.invoke(
                {
                    "case": module.CASE,
                    "position_a": "",
                    "position_b": "",
                    "round": 0,
                    "max_rounds": 2,
                    "resolution": "",
                    **({"history": []} if variant == "merged" else {}),
                },
                config={
                    "run_name": run_id,
                    "tags": ["lab5", variant, run_id],
                    "metadata": {"variant": variant, "run_id": run_id},
                },
            )
            print("\n--- Final BUILD argument ---")
            print(result.get("position_a", ""))
            print("\n--- Final BUY argument ---")
            print(result.get("position_b", ""))
            if variant == "merged":
                print("\n--- Complete debate history including fact checks ---")
                print(module.format_history(result))
            print("\n--- Moderator's resolution ---")
            print(result.get("resolution", ""))
        except Exception:
            status = "failed"
            error_text = traceback.format_exc()
            print(error_text)
            result = {}
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
            sys.stdout = stdout
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    state_path = run_dir / "final_state.json"
    state_path.write_text(json.dumps({
        "run_id": run_id,
        "variant": variant,
        "status": status,
        "elapsed_seconds": elapsed,
        "round": result.get("round"),
        "position_a": result.get("position_a"),
        "position_b": result.get("position_b"),
        "history": result.get("history"),
        "resolution": result.get("resolution"),
    }, indent=2), encoding="utf-8")

    tracing = {"verified": False, "reason": "LANGSMITH_API_KEY missing or placeholder"}
    if status == "completed" and report["tracing_ready"]:
        try:
            tracing = _verify_langsmith(run_id, started)
        except Exception as exc:
            tracing = {"verified": False, "error": f"{exc.__class__.__name__}: {exc}"}

    summary = _summarize_trace(trace_path)
    meta = {
        "run_id": run_id,
        "variant": variant,
        "status": status,
        "error": error_text,
        "started_at": started.isoformat(),
        "elapsed_seconds": elapsed,
        "timeout_seconds": timeout_seconds,
        "model": MODEL_NAME,
        "max_rounds": 2,
        "source": str(source),
        "source_sha256": _sha256(source),
        "instructor_baseline_sha256": _sha256(LAB_DIR / "baseline" / "instructor_debateMAS.py"),
        "python": report["python"],
        "executable": report["executable"],
        "packages": report["packages"],
        "search_calls": summary["tool_calls"],
        "participant_order": summary["participants"],
        "usage_metadata": summary["usage"],
        "tracing": tracing,
        "final_resolution": result.get("resolution"),
        "history_entry_count": len(result.get("history") or []),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    manifest_path = evidence_root / "manifest.json"
    manifest = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.append({
        "run_id": run_id,
        "variant": variant,
        "status": status,
        "started_at": started.isoformat(),
        "elapsed_seconds": elapsed,
        "source_sha256": meta["source_sha256"],
        "tracing_verified": tracing.get("verified", False),
        "public_urls": tracing.get("public_urls", []),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[saved] {run_dir} status={status} elapsed={elapsed:.1f}s")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Run or inspect a Lab 5 debate evaluation.")
    parser.add_argument("--check-ready", action="store_true")
    parser.add_argument("--variant", choices=("merged", "baseline"))
    parser.add_argument("--run-id")
    parser.add_argument("--evidence-root", default=str(LAB_DIR / "evidence"))
    parser.add_argument("--timeout-seconds", type=int, default=1500)
    args = parser.parse_args()
    if args.check_ready:
        print(json.dumps(readiness(), indent=2))
        return 0
    if not args.variant or not args.run_id:
        parser.error("--variant and --run-id are required unless --check-ready is set")
    run_variant(args.variant, args.run_id, Path(args.evidence_root), args.timeout_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
