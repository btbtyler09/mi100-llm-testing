#!/usr/bin/env python3
"""Measure real-text GLM decode and optionally check code, cache, C2 and tools.

Requires only Python's standard library. Evidence is written to --output.
This client never starts, stops or modifies a model server.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

PROMPT = (
    "Write a Python function that merges two sorted lists of integers. "
    "Use two indices and a loop. Explain why the function takes linear time.\n\n"
    "def merge_sorted(left, right):\n"
)
CODE_MESSAGES = [{
    "role": "user",
    "content": "Write only a complete Python function named merge_sorted(left, right) "
               "that merges two sorted lists using two indices and a loop. Handle "
               "empty inputs, negative integers and duplicates. Return the merged "
               "list. No explanation.",
}]


def decode_rate(events, token_count):
    """Exclude all tokens delivered in the first batch from steady decode."""
    timed = [event for event in events if event["token_ids"]]
    if len(timed) < 2 or timed[-1]["seconds"] <= timed[0]["seconds"]:
        raise ValueError("At least two timed token-bearing events are required")
    return ((token_count - len(timed[0]["token_ids"])) /
            (timed[-1]["seconds"] - timed[0]["seconds"]))


class Client:
    def __init__(self, base, model, timeout=300):
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def open(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=self.timeout)

    def get(self, path):
        with self.open(path) as response:
            return response.read().decode()

    def api(self, path, body):
        with self.open(path, body) as response:
            return json.load(response)

    def measure(self, tokens, output_tokens=128):
        body = dict(model=self.model, prompt=tokens, temperature=0, seed=17,
                    max_tokens=output_tokens, ignore_eos=True, stream=True,
                    return_token_ids=True, cache_salt=uuid.uuid4().hex,
                    stream_options={"include_usage": True})
        start = time.perf_counter()
        events, usage, done, finished = [], None, False, False
        with self.open("/v1/completions", body) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                elapsed = time.perf_counter() - start
                if raw == b"[DONE]":
                    done = True
                    break
                data = json.loads(raw)
                if "error" in data:
                    raise RuntimeError(data["error"])
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices", []):
                    finished |= choice.get("finish_reason") is not None
                    if choice.get("token_ids") or choice.get("text"):
                        events.append(dict(seconds=elapsed,
                                           text=choice.get("text", ""),
                                           token_ids=choice.get("token_ids") or []))
        ids = [token for event in events for token in event["token_ids"]]
        assert done and finished and usage, (done, finished, usage)
        assert usage["prompt_tokens"] == len(tokens), usage
        assert usage["completion_tokens"] == len(ids) == output_tokens, (usage, len(ids))
        timed = [event for event in events if event["token_ids"]]
        return dict(
            usage=usage, token_ids=ids, events=events,
            text="".join(event["text"] for event in events),
            absolute_start=start, ttft_seconds=timed[0]["seconds"],
            latency_seconds=time.perf_counter() - start,
            decode_tokens_per_second=decode_rate(events, len(ids)))

    def chat(self, messages, **extra):
        body = dict(model=self.model, messages=messages, temperature=0,
                    seed=17, max_tokens=32,
                    chat_template_kwargs={"reasoning_effort": "low"})
        body.update(extra)
        return self.api("/v1/chat/completions", body)


def check_merge(text):
    """Run the bounded generated function on six known input pairs."""
    code = text.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1].split("\n```", 1)[0]
    tree = ast.parse(code)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert tree.body[0].name == "merge_sorted"
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
               ast.Assign, ast.AugAssign, ast.While, ast.For, ast.If, ast.Expr,
               ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple,
               ast.Subscript, ast.Slice, ast.Call, ast.Attribute, ast.Compare,
               ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.And, ast.Or, ast.Not,
               ast.Add, ast.Sub, ast.Mult, ast.USub, ast.Lt, ast.LtE, ast.Gt,
               ast.GtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Break,
               ast.Continue, ast.Pass)
    assert any(isinstance(node, ast.Return) for node in ast.walk(tree))
    for node in ast.walk(tree):
        assert isinstance(node, allowed), type(node).__name__
        if isinstance(node, ast.Attribute):
            assert node.attr in ("append", "extend", "pop"), node.attr
        if isinstance(node, ast.Name):
            assert not node.id.startswith("__"), node.id
        if isinstance(node, ast.Call):
            assert isinstance(node.func, (ast.Name, ast.Attribute))
            if isinstance(node.func, ast.Name):
                assert node.func.id in ("len", "min", "max", "sorted", "range")
    runner = """import json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(3,3))
resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
scope={'__builtins__': {'len':len,'min':min,'max':max,'sorted':sorted,'range':range}}
exec(sys.stdin.read(),scope)
cases=[([],[]),([1],[]),([],[2]),([1,3,5],[2,4,6]),([-3,0,7],[-2,1,9]),([1,1,3],[1,2,2])]
for left,right in cases:
    assert scope['merge_sorted'](left.copy(),right.copy())==sorted(left+right)
print(json.dumps({'passed':True,'cases':len(cases)}))
"""
    result = subprocess.run([sys.executable, "-I", "-c", runner], input=code,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, (result.returncode, result.stderr)
    return json.loads(result.stdout)


def qualify(client, tokens, save):
    rows = {}

    def code_chat():
        result = client.chat(
            CODE_MESSAGES, max_tokens=512, cache_salt=uuid.uuid4().hex,
            chat_template_kwargs={"reasoning_effort": "low", "enable_thinking": False})
        assert result["choices"][0]["finish_reason"] == "stop", result
        return result

    # Evaluate final-answer code from parsed chat, never a reasoning draft
    # embedded in raw completions. Timing uses a separate fixed-length fixture.
    single = code_chat()
    rows["single_code_response"] = single
    save(rows)
    rows["single_code"] = check_merge(single["choices"][0]["message"]["content"])
    prefix = ("The archive secret is ORCHID-7091. The invoice amount is 42. "
              "Remember both facts.\n" +
              "This is ordinary background about computing systems. " * 190)
    salt = uuid.uuid4().hex
    rows["cache"] = []
    for question, expected in [
        ("What is the archive secret?", "ORCHID-7091"),
        ("What is the archive secret?", "ORCHID-7091"),
        ("What is the invoice amount?", "42"),
    ]:
        messages = [{"role": "user", "content": prefix + "\n" + question +
                     " Reply with only the value. No explanation."}]
        count = client.api("/tokenize", dict(
            model=client.model, messages=messages, add_generation_prompt=True,
            chat_template_kwargs={"reasoning_effort": "low"}))["count"]
        assert 1280 < count < 2048, count
        result = client.chat(messages, cache_salt=salt)
        rows["cache"].append(result)
        save(rows)
        assert result["choices"][0]["message"]["content"].strip() == expected, result
        cached = result["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert (cached == 0 if len(rows["cache"]) == 1 else cached > 0), result["usage"]
    barrier = threading.Barrier(2)
    samples, errors, stop = [], [], threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                for line in client.get("/metrics").splitlines():
                    if line.startswith("vllm:num_requests_running{"):
                        samples.append(float(line.rsplit(" ", 1)[1]))
            except Exception as error:
                errors.append(repr(error))
                return
            stop.wait(0.1)

    def stream():
        barrier.wait(timeout=10)
        return client.measure(tokens)

    observer = threading.Thread(target=monitor, daemon=True)
    observer.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(stream) for _ in range(2)]
            streams = [future.result() for future in futures]
    finally:
        stop.set()
        observer.join(timeout=10)
    first = [r["absolute_start"] + r["events"][0]["seconds"] for r in streams]
    last = [r["absolute_start"] + r["events"][-1]["seconds"] for r in streams]
    overlap = min(last) - max(first)
    rows["parallel"] = dict(streams=streams, overlap_seconds=overlap,
                            peak_running=max(samples, default=0), monitor_errors=errors)
    save(rows)
    assert not errors and max(samples, default=0) == 2 and overlap > 0, rows["parallel"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(code_chat) for _ in range(2)]
        code_responses = [future.result() for future in futures]
    rows["parallel_code_responses"] = code_responses
    save(rows)
    rows["parallel_code"] = [
        check_merge(r["choices"][0]["message"]["content"]) for r in code_responses]
    tools = [{"type": "function", "function": {
        "name": "fixture_lookup", "description": "Look up the value of a fixture key.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                       "required": ["key"], "additionalProperties": False}}}]
    messages = [{"role": "user", "content": "Use fixture_lookup to retrieve the value "
                 "of key release-check. Do not guess it. Then reply with its value only."}]
    payload = dict(model=client.model, messages=messages, tools=tools, tool_choice="auto",
                   temperature=0, seed=17, max_tokens=512,
                   chat_template_kwargs={"reasoning_effort": "max"})
    call = client.api("/v1/chat/completions", payload)
    rows["tool_call"] = call
    save(rows)
    choice = call["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    assert choice["finish_reason"] == "tool_calls" and len(calls) == 1, call
    tool = calls[0]
    assert tool["function"]["name"] == "fixture_lookup", tool
    assert json.loads(tool["function"]["arguments"]) == {"key": "release-check"}, tool
    marker = "ORCHID-" + uuid.uuid4().hex[:12]
    messages += [message, {"role": "tool", "tool_call_id": tool["id"],
                           "content": json.dumps({"value": marker})}]
    answer = client.api("/v1/chat/completions", payload)
    rows["tool_result"] = answer
    save(rows)
    final = answer["choices"][0]["message"]
    assert marker in (final.get("content") or ""), answer
    assert "<think>" not in (final.get("content") or "")
    assert "reasoning" in final or "reasoning_content" in final, final
    # Closing a live stream must release its request and permit reuse.
    cancel = dict(model=client.model, prompt=tokens, max_tokens=1024,
                  ignore_eos=True, stream=True, temperature=0)
    with client.open("/v1/completions", cancel) as response:
        for line in response:
            if line.startswith(b"data: ") and b'"text":' in line:
                break
    deadline = time.monotonic() + 15
    while True:
        active = [float(line.rsplit(" ", 1)[1]) for line in client.get("/metrics").splitlines()
                  if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))]
        if active and not any(active):
            break
        assert time.monotonic() < deadline, ("Request did not cancel", active)
        time.sleep(0.2)
    reuse = client.chat([{"role": "user", "content": "What is 2+2? Reply with only the digit."}])
    assert reuse["choices"][0]["message"]["content"].strip() == "4", reuse
    rows["cancellation_reuse"] = True
    rows["passed"] = True
    save(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="glm53-flash-uncensored")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--min-tps", type=float)
    parser.add_argument("--label", default="unspecified")
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0 or args.output_tokens < 2:
        parser.error("Require samples >= 1, warmups >= 0 and output-tokens >= 2.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Output already exists; choose a new file to preserve earlier results.")
    client = Client(args.base_url, args.model)
    rows = dict(label=args.label, passed=False, metric="returned-token decode rate; first batch excluded",
                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                arguments={**vars(args), "output": str(args.output)}, warmups=[], samples=[])

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(rows, indent=2) + "\n")
        temporary.replace(args.output)

    def save_checks(checks):
        rows["qualification"] = checks
        save()

    try:
        rows["models"] = json.loads(client.get("/v1/models"))
        rows["metrics_before"] = client.get("/metrics")
        tokens = client.api("/tokenize", dict(model=args.model, prompt=PROMPT))["tokens"]
        rows["prompt"] = dict(text=PROMPT, token_ids=tokens)
        for kind, count in (("warmups", args.warmups), ("samples", args.samples)):
            for index in range(count):
                result = client.measure(tokens, args.output_tokens)
                rows[kind].append(result)
                save()
                print(kind, index + 1, round(result["decode_tokens_per_second"], 4),
                      "tok/s", flush=True)
        rows["summary"] = {
            key: {"median": statistics.median(r[key] for r in rows["samples"]),
                  "min": min(r[key] for r in rows["samples"]),
                  "max": max(r[key] for r in rows["samples"])}
            for key in ("decode_tokens_per_second", "ttft_seconds", "latency_seconds")}
        rows["metrics_after"] = client.get("/metrics")
        save()
        if args.qualify:
            qualify(client, tokens, save_checks)
        median = rows["summary"]["decode_tokens_per_second"]["median"]
        if args.min_tps is not None:
            assert median >= args.min_tps, f"Median {median:.4f} below {args.min_tps} tokens/s"
        rows["passed"] = True
        print(json.dumps({"passed": True, "summary": rows["summary"],
                          "qualified": bool(args.qualify)}), flush=True)
    except BaseException as error:
        rows["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
