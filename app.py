"""Streamlit interface for the existing Hopscotch LangGraph project.

Place this file beside Hopscotch/graph.py, gate.py, config.py, and data.py.
Run: python -m streamlit run streamlit_app.py
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import streamlit as st
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
# Existing modules use absolute imports such as `from Hopscotch.config import llm`.
if str(APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(APP_DIR.parent))
load_dotenv(APP_DIR / ".env")

st.set_page_config(page_title="Hopscotch | Exchange desk", page_icon="↩", layout="wide")
st.markdown("""
<style>
.stApp {background: #fbfaf7;}
[data-testid="stSidebar"] {background: #f1eee8;}
.block-container {max-width: 1250px; padding-top: 2.2rem;}
.desk-eyebrow {font-size: .8rem; letter-spacing: .16em; color: #a4492e;
  font-weight: 700; margin-bottom: .7rem;}
.desk-subtitle {color: #65645f; font-size: 1.05rem; margin-bottom: 1.4rem;}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: #b54e32; border-color: #b54e32; color: white;}
</style>
""", unsafe_allow_html=True)

SAMPLES = {
    "Size exchange": "Order HS2291. The 4-5Y kurta set is too small. Please exchange it and pick it up from 122002.",
    "Damaged item": "Order HS2291 arrived with torn stitching. I am upset and would like a refund. Please collect it from 122002.",
    "Missing details": "I received the wrong size and would like an exchange. Can you help?",
    "Gate rejection": "Do you have any Diwali offers on party wear this week?",
}
LABELS = {
    "extract_claim": "Extract customer claim",
    "read_tone": "Read customer tone",
    "investigator": "Investigate request",
    "tools": "Look up facts",
    "policy_check": "Apply policy rules",
    "escalate": "Flag for human review",
    "respond": "Generate reply draft",
}


def plain(value):
    """Convert graph messages and structured values to exportable JSON."""
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def error_text(exc):
    text = str(exc)
    secret = os.getenv("OPENROUTER_API_KEY", "")
    if secret:
        text = text.replace(secret, "[redacted]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", text)


def load_gate():
    if not (APP_DIR / "gate.py").is_file():
        raise FileNotFoundError("Place streamlit_app.py inside your existing Hopscotch folder, beside gate.py and graph.py.")
    return importlib.import_module("Hopscotch.gate").gate


def load_graph():
    missing = [name for name in ("graph.py", "data.py", "config.py", "tools.py")
               if not (APP_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing existing project files: " + ", ".join(missing))
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise ValueError("Add OPENROUTER_API_KEY to Hopscotch/.env, then restart Streamlit.")
    config = importlib.import_module("Hopscotch.config")
    tools = importlib.import_module("Hopscotch.tools").TOOLS
    build = importlib.import_module("Hopscotch.graph").build_graph
    policies = {name: getattr(config, name) for name in
                ("RETURN_WINDOW_DAYS", "HIGH_VALUE", "VENDOR_SPIKE", "MAX_TRIPS")}
    # A fresh execution is used per request. No conversation state is shared.
    return build(tools), policies


def execute(message, status):
    started = perf_counter()
    verdict = load_gate()(message)
    record = {"time": datetime.now().astimezone().isoformat(timespec="seconds"),
              "message": message, "gate": verdict, "events": [], "state": {}, "policies": {}}
    if verdict != "accepted":
        record["state"] = {"exit": "rejected", "reason": verdict}
        record["seconds"] = round(perf_counter() - started, 2)
        return record

    graph, policies = load_graph()
    record["policies"] = policies
    state = {"message": message, "messages": [], "trips": 0}
    recursion_limit = max(30, int(policies["MAX_TRIPS"]) * 3 + 15)
    for chunk in graph.stream(state, stream_mode="updates",
                              config={"recursion_limit": recursion_limit}):
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            status.write("✓ " + LABELS.get(node, node))
            record["events"].append({"node": node, "update": plain(update)})
            for key, value in update.items():
                if key == "messages":
                    state["messages"] = state.get("messages", []) + list(value)
                else:
                    state[key] = value
    if state.get("exit") not in {"approve", "decline", "escalate"}:
        raise RuntimeError("The graph ended without a policy decision. Check its final node output.")
    record["state"] = plain(state)
    record["seconds"] = round(perf_counter() - started, 2)
    return record


def render_result(record):
    state = record["state"]
    outcome = state.get("exit", "unknown")
    title = {"approve": "Exchange / refund eligible", "decline": "Request declined",
             "escalate": "Human review needed", "rejected": "Message not accepted"}.get(outcome, "Result")
    show = {"approve": st.success, "decline": st.error,
            "escalate": st.warning, "rejected": st.info}.get(outcome, st.info)
    st.subheader("Request outcome")
    show(f"**{title}** — {state.get('reason', 'No reason returned')}")
    st.caption(f"Processed in {record['seconds']:.2f}s · {record['time']}")
    if outcome == "rejected":
        st.caption("The input gate stopped this request before any model call.")
        return

    claim = state.get("claim") or {}
    a, b, c, d = st.columns(4)
    a.metric("Order", claim.get("order_id") or "Not provided")
    b.metric("Customer tone", str(state.get("tone", "unknown")).title())
    c.metric("Requested action", str(claim.get("wants", "unclear")).title())
    d.metric("Tool-call rounds", state.get("trips", 0))

    reply_tab, facts_tab, trace_tab = st.tabs(["Reply draft", "Claim & facts", "Execution trace"])
    with reply_tab:
        if state.get("reply"):
            st.info("Draft only. Any pickup-booked or reply-sent wording comes from the demo backend; no pickup or message has actually been sent.")
            st.text_area("Generated customer reply", value=state["reply"], height=185,
                         disabled=True, key="reply_" + record["time"])
            st.download_button("Download reply", state["reply"], file_name="hopscotch_reply.txt",
                               mime="text/plain", key="download_reply")
        else:
            st.write("This request needs human review. The graph did not generate a reply.")
        if state.get("action"):
            st.caption("Backend action (simulated): " + str(state["action"]))
        if outcome == "escalate":
            st.caption("Flagged in this demo session. Connect a ticketing system for a durable escalation queue.")

    with facts_tab:
        st.write("**Extracted claim**")
        st.json(claim)
        facts = []
        for event in record["events"]:
            for message in event["update"].get("messages", []):
                if isinstance(message, dict) and message.get("type") == "tool":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except (ValueError, TypeError):
                            pass
                    facts.append({"tool": message.get("name"), "result": content})
        if facts:
            for fact in facts:
                st.write("**" + str(fact["tool"]) + "**")
                st.json(fact["result"])
        else:
            st.caption("No tool results returned.")
        st.warning("check_pickup uses postal delivery data. A valid postal delivery location does not confirm reverse courier pickup availability.")
        if record["policies"]:
            st.write("**Policy values used for this run**")
            st.json(record["policies"])

    with trace_tab:
        st.caption("Node outputs and tool calls, in the order received from the graph.")
        for index, event in enumerate(record["events"], start=1):
            with st.expander(f"{index:02d} · {LABELS.get(event['node'], event['node'])}"):
                st.json(event["update"])
        st.download_button("Download request report", json.dumps(record, indent=2, ensure_ascii=False),
                           file_name="hopscotch_request_report.json", mime="application/json",
                           key="download_report")


def main():
    st.session_state.setdefault("desk_history", [])
    st.session_state.setdefault("desk_result", None)
    st.session_state.setdefault("desk_error", None)
    st.session_state.setdefault("customer_message", SAMPLES["Size exchange"])

    with st.sidebar:
        st.title("Hopscotch")
        st.caption("EXCHANGE DESK · DEMO")
        st.divider()
        st.write("**How it works**")
        st.write("1. Check the message\n2. Extract claim & tone\n3. Gather order facts\n4. Apply policy\n5. Prepare a draft or flag for review")
        st.divider()
        st.write("**Setup**")
        key_present = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
        st.caption("API key: " + ("configured" if key_present else "missing"))
        st.caption("Uses your existing graph.py, tools.py, data.py and config.py.")
        st.caption("Model calls use your OpenRouter account and may incur charges.")
        if st.button("Clear session", width="stretch"):
            st.session_state["desk_history"] = []
            st.session_state["desk_result"] = None
            st.session_state["desk_error"] = None
            st.rerun()

    st.markdown('<div class="desk-eyebrow">CUSTOMER CARE / EXCHANGES & RETURNS</div>', unsafe_allow_html=True)
    st.title("A little help. A clear next step.")
    st.markdown('<div class="desk-subtitle">Review a customer request, inspect the facts, and prepare a thoughtful response.</div>', unsafe_allow_html=True)
    history = st.session_state["desk_history"]
    one, two, three = st.columns(3)
    one.metric("Recent requests", len(history))
    two.metric("Eligible · recent", sum(r["state"].get("exit") == "approve" for r in history))
    three.metric("Needs review · recent", sum(r["state"].get("exit") == "escalate" for r in history))
    st.divider()
    st.subheader("Customer request")
    sample = st.selectbox("Try an example", list(SAMPLES))
    if st.button("Use example"):
        st.session_state["customer_message"] = SAMPLES[sample]
    with st.form("request_form"):
        message = st.text_area("Customer message", key="customer_message", height=145,
                               help="Include an order ID, the issue, the requested action and a pickup pincode.")
        st.caption("Examples use HS2291. Actual decisions depend on your data.py and the facts returned by tools.")
        submitted = st.form_submit_button("Review request →", type="primary", width="stretch")
    if submitted:
        st.session_state["desk_result"] = None
        st.session_state["desk_error"] = None
        if not message.strip():
            st.session_state["desk_error"] = "Enter a customer message first."
        else:
            with st.status("Reviewing the request…", expanded=True) as status:
                try:
                    record = execute(message.strip(), status)
                except (Exception, SystemExit) as exc:
                    st.session_state["desk_error"] = error_text(exc)
                    status.update(label="Request could not be completed", state="error")
                else:
                    st.session_state["desk_result"] = record
                    history.append(record)
                    del history[:-25]
                    status.update(label="Review complete", state="complete", expanded=False)
            # Refresh top metrics from the completed run; do not execute it again.
            st.rerun()
    if st.session_state["desk_error"]:
        st.error(st.session_state["desk_error"])
        st.caption("Check your API key, package installation, project files and network connection. Fix the issue, then submit again.")
    if st.session_state["desk_result"]:
        render_result(st.session_state["desk_result"])
    if history:
        with st.expander("Recent requests · this session only"):
            st.dataframe([{"Time": r["time"], "Decision": r["state"].get("exit"),
                           "Order": (r["state"].get("claim") or {}).get("order_id"),
                           "Reason": r["state"].get("reason")} for r in reversed(history)],
                         width="stretch", hide_index=True)
    st.caption("Learning prototype · Results and session history are not a persistent order-management system.")


if __name__ == "__main__":
    main()
