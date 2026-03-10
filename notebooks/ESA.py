import json
import time
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY = "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"
MODEL   = "google/gemini-2.5-flash-lite"

# Other free models to try if flash hits limits — just swap MODEL:
#   "meta-llama/llama-3.3-70b-instruct:free"
#   "deepseek/deepseek-r1:free"
#   "mistralai/mistral-7b-instruct:free"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json"
}

# ── Core call ─────────────────────────────────────────────────────────────────

def ask(prompt: str, system: str = None, temperature: float = 0.0) -> str:
    """
    Send a prompt to OpenRouter and return the response text.
    Retries automatically on rate limits.
    temperature=0.0 for deterministic outputs (recommended for graph problems).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model":       MODEL,
        "messages":    messages,
        "temperature": temperature
    }

    for attempt in range(5):
        response = requests.post(API_URL, headers=HEADERS, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        elif response.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"  [Rate limited] Waiting {wait}s before retry {attempt+1}/5...")
            time.sleep(wait)

        else:
            raise RuntimeError(
                f"API error {response.status_code}: {response.text}"
            )

    raise RuntimeError("Failed after 5 retries. Check your quota at openrouter.ai.")


# ── ESA Prompt Templates ──────────────────────────────────────────────────────

OUTPUT_INTEGRITY_RULE = """
OUTPUT INTEGRITY RULE — Read this before proceeding.

This prompt will produce a long response. That is correct and expected.
Length is not a problem here.

You are explicitly forbidden from:
  - Summarizing steps instead of executing them
  - Writing "continuing similarly..." or "and so on..."
  - Writing "omitting for brevity" or any equivalent
  - Skipping a state block because it resembles a prior one
  - Combining multiple steps into one block

Each step that exists must have its own complete block.
A skipped block is a wrong answer, not a short answer.

If you find yourself wanting to compress, instead ask:
  "Have I printed the state block for this step?"
If no, print it. Then proceed.
"""

PHASE1_PROMPT = """
{output_integrity_rule}

PHASE 1 - ROUNDTABLE ANALYSIS

5 Experts sit in a circle. Before anything else, Expert 1 writes down
the complete formal representation of the problem input:
  - Full edge list with all properties
  - Vertex count and edge count, verified against the problem statement
  - For graphs: full adjacency/edge list + counts

Then each expert makes one remark about a distinct structural property
they observe. No two experts may remark on the same property.

A 6th Judge then:
  - Evaluates each remark
  - Identifies which algorithm or approach to use and why
  - Produces a numbered execution plan specific enough that a machine
    could follow it with no ambiguity
  - States explicitly: "Phase 2 will process exactly N elements"
    where N = total edges to be EXAMINED (not just accepted --
    every edge gets a step block including rejections)
  - Assigns one fault class to each of the 5 experts for Phase 3
    from: ORDER / STATE / CONFLICT / COMPLETENESS / ARITHMETIC
    If a class is not applicable, reassign to the next most relevant
  - Identifies the top 2 execution risks

A Devil's Advocate then:
  - Challenges the algorithm choice
  - Challenges the execution plan for edge cases
  - Challenges the element count declared by the Judge

The Judge resolves all challenges and locks the plan.

PHASE 1 IS NOW CLOSED.
The plan, element count, and fault class assignments are locked.
No new algorithmic decisions may be made after this point.

PROBLEM:
{problem}
"""

PHASE2_PROMPT = """
{output_integrity_rule}

PHASE 2 - ESA EXECUTION

No experts. No discussion. No opinions. Execute the locked plan only.

The Judge declared Phase 2 will process exactly N elements.
You must produce exactly N step blocks. No more, no less.
Count your blocks before closing Phase 2.

BEFORE Step 1: write the complete ordered input the algorithm will
consume. Do not begin Step 1 until this full list is written.

For every single element print this block, then move to the next:

STEP [N]
Processing   : [current element and its weight]
Decision     : [ADD or SKIP]
Reason       : [cite the exact prior step number or rule]
Full state   : [complete set partition of ALL vertices, e.g. {{1,3,9}} {{2,10}} {{4}}]
Unresolved   : [explicit list of vertices not yet in the main component]
Running value: [cumulative total]
---

ANTI-ASSUMPTION RULE:
You may never state that a vertex is in a certain component unless
you can cite the exact Step N where it joined. If you cannot cite
it, recompute from Step 1.

FULL STATE RULE:
Full state must show every vertex in the graph assigned to a component.
Write it as a set partition on one line. Never write "same as above"
or any shorthand.

When the algorithm terminates, print all three audits below:

STEP COUNT AUDIT:
  Declared by Judge : [N]
  Blocks printed    : [count your blocks above]
  Match?            : YES / NO
  If NO             : print the missing step blocks now before continuing

COMPLETENESS AUDIT:
  All vertices in graph    : [list every one]
  Vertices in solution     : [list every one that appears in an accepted edge]
  Missing                  : [anything in row 1 not in row 2, or NONE]
  If missing vertices exist : go back and find the edge that connects them

CONSTRAINT AUDIT:
  [ ] Correct number of edges accepted? (should be V-1 for MST)
  [ ] All elements processed in correct sorted order?
  [ ] Each ADD decision verified legal before accepting?
  [ ] Final value built step-by-step with every intermediate shown?

PHASE 2 IS NOW CLOSED. The execution log is final.

PHASE 1 OUTPUT:
{phase1}
"""

PHASE3_PROMPT = """
PHASE 3 - ROUNDTABLE AUDIT

The 5 experts reconvene with their assigned fault classes from Phase 1.
They are shown the Phase 2 execution log now for the first time.
They have not seen it before this moment.

Each expert audits only their assigned fault class and reports:
  FAULT CLASS : [their class]
  METHOD      : [exactly how they checked it]
  FINDING     : PASS or FAIL
  IF FAIL     : Step [N] — [what is wrong] — [what correct state should be]

Expert 1 [{fault1}]:
  Extract every weight in the order it was processed.
  Write them as a sequence. Verify the sequence is non-decreasing.
  Flag any position where order was violated.

Expert 2 [{fault2}]:
  Select 3 ADD decisions from the log (first, middle, last).
  For each: retrace every prior step from scratch using only the
  log above it. Recompute which component each endpoint belongs to.
  Does your recomputed state match what the log claimed?

Expert 3 [{fault3}]:
  For every ADD decision in the log:
  verify both endpoints were in DIFFERENT components at that exact
  moment by reading only the Full State lines of all prior steps.
  Flag any ADD where endpoints were already in the same component.

Expert 4 [{fault4}]:
  List every vertex in the graph.
  List every vertex that appears in at least one accepted edge.
  Compute the difference. Flag any vertex not covered.

Expert 5 [{fault5}]:
  Recompute the final value from zero using only accepted edges.
  Show every addition on its own line:
    Start : 0
    +[w1] = [total]
    +[w2] = [total]
    ...
  Does your total match the log's final Running value?

The Devil's Advocate then:
  - Reviews all 5 findings
  - If any FAIL: names the earliest faulted step and the correct state
  - If all PASS: states exactly "NO FAULT FOUND IN ANY AUDIT CLASS"

The Judge delivers FINAL VERDICT:
  - If fault found: state which step is wrong and what the fix is
  - If no fault: confirm the answer is valid and proceed

FINAL ANSWER

State the answer in one sentence.
State in one sentence what condition would make this answer wrong,
and confirm that condition does not hold.

Return your answer as JSON on the very last line with no other text
after it:
{{"answer": "number/Boolean"}}

PHASE 1 OUTPUT:
{phase1}

PHASE 2 EXECUTION LOG:
{phase2}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_fault_classes(phase1_text: str) -> dict:
    """
    Extract the 5 fault class assignments from Phase 1 output.
    Falls back to sensible defaults if parsing fails.
    """
    defaults = {
        "fault1": "ORDER",
        "fault2": "STATE",
        "fault3": "CONFLICT",
        "fault4": "COMPLETENESS",
        "fault5": "ARITHMETIC"
    }
    upper = phase1_text.upper()
    classes = ["ORDER", "STATE", "CONFLICT", "COMPLETENESS", "ARITHMETIC"]
    found = [c for c in classes if c in upper]
    if len(found) >= 5:
        return {f"fault{i+1}": found[i] for i in range(5)}
    return defaults


def extract_json_answer(text: str) -> dict:
    """Pull the last JSON object from a response string."""
    try:
        last  = text.rfind("}")
        first = text.rfind("{", 0, last + 1)
        if first != -1 and last != -1:
            return json.loads(text[first:last + 1])
    except (json.JSONDecodeError, ValueError):
        pass
    return {"answer": "could not parse", "raw": text[-300:]}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(problem: str) -> dict:
    """
    3-phase ESA + Roundtable pipeline.

    Phase 1 — Roundtable  : free reasoning, algorithm selection, plan locking
    Phase 2 — ESA         : strict mechanical execution, state blocks per step
    Phase 3 — Audit       : independent fault detection, final answer
    """
    sep = "=" * 64

    # ── Phase 1 ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("PHASE 1 — ROUNDTABLE ANALYSIS")
    print(sep)
    phase1 = ask(PHASE1_PROMPT.format(
        output_integrity_rule=OUTPUT_INTEGRITY_RULE,
        problem=problem
    ))
    print(phase1)

    # ── Phase 2 ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("PHASE 2 — ESA EXECUTION")
    print(sep)
    phase2 = ask(PHASE2_PROMPT.format(
        output_integrity_rule=OUTPUT_INTEGRITY_RULE,
        phase1=phase1
    ))
    print(phase2)

    # ── Phase 3 ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("PHASE 3 — ROUNDTABLE AUDIT")
    print(sep)
    fault_classes = extract_fault_classes(phase1)
    phase3 = ask(PHASE3_PROMPT.format(
        phase1=phase1,
        phase2=phase2,
        **fault_classes
    ))
    print(phase3)

    # ── Extract answer ────────────────────────────────────────────
    result = extract_json_answer(phase3)
    print(f"\n{sep}")
    print(f"FINAL RESULT: {result}")
    print(sep)
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Replace this with any graph problem
    PROBLEM = """
    Find the total weight of the minimum spanning tree (MST).
    Undirected weighted graph with 16 vertices and 20 edges.
    Edges (Node Node Weight):
    1 2 22
    1 3 69
    2 13 13
    1 7 58
    1 14 89
    1 16 91
    13 12 47
    13 5 88
    2 10 62
    10 9 98
    1 6 51
    10 8 48
    14 4 10
    4 15 29
    15 11 52
    9 1 32
    15 10 9
    11 7 71
    8 11 35
    14 11 77
    """

    run_pipeline(PROBLEM)