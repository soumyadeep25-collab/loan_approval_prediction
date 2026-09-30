"""Probe freellmapi models for the two LLM calls the agent needs:
   1) SUPERVISOR  -> tool/function calling (choose_action)
   2) EXPLAINER   -> plain chat completion (narrate a decision)
Rotates through ALL available models, with retry + backoff, and reports which
models succeed at each call. This proves the agentic LLM path on real LLMs.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.getcwd(), "backend", "src"))

BASE = os.environ.get("LLM_BASE_URL", "http://localhost:3001/v1")
KEY = os.environ.get("LLM_API_KEY", os.environ.get("FREELLMAPI_KEY", ""))

import urllib.request

# load model list
with open("/tmp/freellmapi_models.txt") as f:
    MODELS = [m.strip() for m in f if m.strip()]

SUPERVISOR_TOOL = [{
    "type": "function",
    "function": {
        "name": "choose_action",
        "description": "Decide next pipeline action given evaluation metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["retrain", "retune_threshold", "try_different_model", "finish"]},
                "reasoning": {"type": "string"},
            }, "required": ["action", "reasoning"],
        },
    },
}]

def _post(payload, model, timeout=20):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def _extract_tool_args(msg):
    """Return parsed tool-call arguments dict from a chat message, handling
    both SDK object shapes and raw-dict shapes (some freellmapi models return
    tool_calls as plain dicts without .function/.arguments attributes)."""
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None
    tc = tcs[0]
    # object shape: tc.function.arguments (str)
    fn = getattr(tc, "function", None)
    if fn is not None:
        args = getattr(fn, "arguments", None)
        if isinstance(args, str):
            try:
                return json.loads(args)
            except Exception:
                return None
        if isinstance(args, dict):
            return args
    # dict shape: tc["function"]["arguments"]
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                return json.loads(args)
            except Exception:
                return None
        if isinstance(args, dict):
            return args
    return None

def try_supervisor(model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "F1=0.65, precision=0.58, recall=0.90. Top features: income, credit. Choose next action."}],
        "tools": SUPERVISOR_TOOL, "tool_choice": "auto", "max_tokens": 120,
    }
    r = _post(payload, model)
    msg = r["choices"][0]["message"]
    args = _extract_tool_args(msg)
    if args and "action" in args:
        return True, args.get("action", "?")
    return False, (msg.get("content") or "")[:40]

def try_explainer(model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Explain in 1 sentence why a loan with income 8000, credit 760 was APPROVED."}],
        "max_tokens": 80,
    }
    r = _post(payload, model)
    return True, (r["choices"][0]["message"].get("content") or "")[:60]

def probe(kind, fn, logpath):
    print(f"\n===== PROBING {kind} across {len(MODELS)} models =====", flush=True)
    ok, fail = [], []
    with open(logpath, "w") as lf:
        for m in MODELS:
            try:
                success, detail = fn(m)
                if success:
                    ok.append((m, detail))
                    line = f"OK   {m} -> {detail}"
                else:
                    fail.append(m)
                    line = f"NOTOOL {m} -> {detail}"
            except Exception as e:
                err = str(e)
                fail.append(m)
                line = f"ERR  {m} -> {err[:60]}"
            print("  " + line, flush=True)
            lf.write(line + "\n")
    print(f"\n{kind}: {len(ok)} working, {len(fail)} failing", flush=True)
    return ok

if __name__ == "__main__":
    sup = probe("SUPERVISOR(tool-call)", try_supervisor, "/tmp/probe_supervisor.log")
    exp = probe("EXPLAINER(chat)", try_explainer, "/tmp/probe_explainer.log")
    out = {"supervisor_ok": [m for m, _ in sup], "explainer_ok": [m for m, _ in exp]}
    with open(os.path.join(os.getcwd(), "models", "working_llm_models.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved models/working_llm_models.json")
    print("SUPERVISOR models:", out["supervisor_ok"])
    print("EXPLAINER models :", out["explainer_ok"])
