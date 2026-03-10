import json
import time
import requests
from dotenv import load_dotenv
import os
load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY_1")
MODEL   = "google/gemini-2.0-flash-lite-001"

# Other models to try:
#   "google/gemini-2.0-flash-exp:free"
#   "meta-llama/llama-3.3-70b-instruct:free"
#   "deepseek/deepseek-r1:free"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json"
}


# ── Inject helper ──────────────────────────────────────────────────────────────

def inject(template: str, **kwargs) -> str:
    """
    Substitute {key} placeholders using plain string replacement.
    Unlike str.format(), never chokes on curly braces in values
    such as set notation {1,2} or JSON {"answer": "value"}.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


# ── Streaming call ─────────────────────────────────────────────────────────────

def ask_stream(prompt: str, system: str = None, temperature: float = 0.0) -> str:
    """
    Send a prompt to OpenRouter and stream the response to the console.
    Returns the full response text when complete.
    Retries automatically on rate limits.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model":       MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  16000,
        "stream":      True
    }

    for attempt in range(5):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                stream=True
            )

            if response.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"\n  [Rate limited] Waiting {wait}s before retry {attempt+1}/5...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                raise RuntimeError(f"API error {response.status_code}: {response.text}")

            full_text = ""
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data = decoded[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk   = json.loads(data)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0]["delta"].get("content", "")
                        if delta:
                            print(delta, end="", flush=True)
                            full_text += delta
                    except (json.JSONDecodeError, KeyError):
                        continue

            print()
            return full_text

        except requests.exceptions.RequestException as e:
            if attempt < 4:
                wait = 10 * (attempt + 1)
                print(f"\n  [Connection error] Waiting {wait}s... retry {attempt+1}/5")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Request failed after 5 attempts: {e}")

    raise RuntimeError("Failed after 5 retries. Check your quota at openrouter.ai.")


# ── Shared Rules ───────────────────────────────────────────────────────────────

OUTPUT_INTEGRITY_RULE = """
OUTPUT INTEGRITY RULE

This prompt will produce a long response. That is correct and expected.
Length is not a problem here.

You are explicitly forbidden from:
  - Summarizing steps instead of executing them
  - Writing "continuing similarly..." or "and so on..."
  - Writing "omitting for brevity" or any equivalent
  - Skipping a step block because it resembles a prior one
  - Combining multiple steps into one block

Each step that exists must have its own complete block.
A skipped block is a wrong answer, not a short answer.
"""

ANTI_ASSUMPTION_RULE = """
ANTI-ASSUMPTION RULE:
You may never assert that a vertex or state is in a certain condition
unless you can cite the exact Step N or Checkpoint N where that was
established. If you cannot cite it, recompute from the beginning.

FULL STATE RULE:
All state fields must be written in full at every checkpoint.
Never write "same as above", "unchanged", or any shorthand.
"""


# ══════════════════════════════════════════════════════════════════════════════
# SHARED AGENT PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

AGENT1_PARSER_PROMPT = """
You receive a raw graph problem statement.
Your only job is to extract and reformat the input. Do not solve anything.

Output exactly these fields and nothing else:
  Problem type : [MIS / MST / Max Clique / Shortest Path / etc.]
  Vertices     : [count] -- [list every vertex label]
  Edges        : [count] -- list every edge one per line as (u, v) or (u, v, weight)
  Source/Sink  : [if applicable, else N/A]
  Objective    : [one sentence describing exactly what must be computed]

Verify your edge count matches the problem statement before finishing.

PROBLEM:
{problem}
"""

AGENT_CLASSIFIER_PROMPT = """
You receive a structured graph problem.
Classify it into exactly one execution track.

TRACK A -- MECHANICAL
  The problem has a known polynomial algorithm with a fixed
  execution order. The answer is found by following a defined
  procedure. Deviation produces wrong answers.
  Examples: MST, Shortest Path, BFS/DFS, Topological sort,
  Tree DP, Max Flow.

TRACK B -- EXPLORATORY
  The problem requires search, backtracking, or combinatorial
  reasoning. No single fixed execution order. The model needs
  to explore, prune, and reason about partial solutions.
  Examples: Maximum Clique, Maximum Independent Set (non-tree),
  Graph Coloring, Hamiltonian Path, Travelling Salesman.

  NOTE: Maximum Independent Set on a TREE is Track A (Tree DP).
  Maximum Independent Set on a general/dense graph is Track B.

Write your reasoning first:
  REASONING : [2-3 sentences justifying the classification]
  ALGORITHM : [name of the algorithm to use]
  TRACK     : [A or B]

Then output this JSON as the very last line with nothing after it:
{"track": "A", "algorithm": "algorithm name"}

PARSED PROBLEM:
{parsed}
"""

AGENT7_FAITHFULNESS_PROMPT = """
You are a Logical Faithfulness Judge. You have no prior knowledge of
this problem. Judge whether the final answer is logically faithful
to the reasoning that produced it.

This is NOT a correctness check. You are checking whether the answer
follows from the steps, not whether the steps are mathematically right.

DIMENSION 1 -- ANSWER GROUNDING
  Does the final answer match the last value in the execution log?
  RATING   : GROUNDED / UNGROUNDED
  EVIDENCE : [quote the relevant log line and the final answer]

DIMENSION 2 -- REASONING CONTINUITY
  Does each step's decision follow logically from its stated reason?
  RATING   : CONTINUOUS / BROKEN
  IF BROKEN: Step N -- Decision was [X] but Reason implies [Y]

DIMENSION 3 -- INTERNAL CONSISTENCY
  Do state fields ever change without a decision causing it?
  RATING         : CONSISTENT / INCONSISTENT
  IF INCONSISTENT: Step N -- [what changed without justification]

DIMENSION 4 -- AUDIT FAITHFULNESS
  Were the audit checks sufficient to catch real faults?
  RATING       : FAITHFUL / SUPERFICIAL
  IF SUPERFICIAL: [which class and why it was insufficient]

DIMENSION 5 -- CONCLUSION SUPPORT
  Is the answer supported by evidence or does it rest on one claim?
  RATING   : WELL-SUPPORTED / WEAKLY-SUPPORTED
  REASONING: [one sentence]

DIMENSION 6 -- UNJUSTIFIED LEAPS
  Are there assertions made without citing a prior step or rule?
  RATING  : NONE FOUND / LEAPS FOUND
  IF FOUND: Step N -- [the unjustified assertion]

DIMENSION 7 -- LOOP DETECTION
  Does the log contain repeated steps, restarts, or revisited states?
  RATING : CLEAN / LOOP DETECTED
  IF LOOP: [at which step and what triggered it]

FAITHFULNESS SCORE: [X / 7 dimensions passed]

OVERALL VERDICT:
  HIGH   -- 6/7 or 7/7: answer is logically faithful
  MEDIUM -- 4/7 or 5/7: partially faithful, flag concerns
  LOW    -- 3/7 or below: answer cannot be trusted

RECOMMENDATION:
  HIGH   : Accept the answer.
  MEDIUM : Flag the failed dimensions and note the risk.
  LOW    : Reject. Recommend re-running the pipeline.

EXECUTION LOG:
{final_log}

AUDIT REPORT:
{audit}

FINAL ANSWER:
{answer}
"""


# ══════════════════════════════════════════════════════════════════════════════
# TRACK A -- MECHANICAL (ESA)
# ══════════════════════════════════════════════════════════════════════════════

AGENT2A_PLANNER_PROMPT = """
You receive a structured graph problem. You do not execute anything.

Produce these seven outputs:

1. ALGORITHM
   Name the correct algorithm and justify it.
   Consider at least two alternatives and dismiss them.

2. ALGORITHM CLASS
   Choose one: EDGE-PROCESSING / DYNAMIC-PROGRAMMING / SEARCH / OTHER

3. EXECUTION PLAN
   Numbered steps specific enough for a machine to follow exactly.

4. FULL INPUT REPRESENTATION
   The complete input formatted for the chosen algorithm.
   Label items E1, E2... or V1, V2... No truncation.

5. TERMINATION CONDITION
   The exact logical condition that ends execution. Not a step count.

6. STEP BLOCK FORMAT
   Define the exact fields for each step block:

   EDGE-PROCESSING:
     Processing    : [edge being examined]
     Decision      : [ACCEPT or REJECT]
     Reason        : [cite exact prior step or rule]
     Full state    : [complete set partition of ALL vertices]
     Unresolved    : [vertices not yet in main component]
     Running value : [cumulative total]

   DYNAMIC-PROGRAMMING:
     Vertex/State  : [vertex or state being computed]
     Subproblems   : [sub-state values used]
     Computation   : [exact formula applied]
     dp value      : [resulting dp table entry]
     Running best  : [current best known answer]

   SEARCH:
     Processing    : [node being expanded]
     Queue/Stack   : [current contents]
     Visited       : [all visited nodes]
     Decision      : [EXPAND / SKIP / TERMINATE]
     Running value : [current best]

   Define a custom format if none fit.

7. FAULT CLASS ASSIGNMENTS
   Assign one fault class to each of 5 auditors:
     Auditor 1: [CLASS] -- [what to check]
     Auditor 2: [CLASS] -- [what to check]
     Auditor 3: [CLASS] -- [what to check]
     Auditor 4: [CLASS] -- [what to check]
     Auditor 5: [CLASS] -- [what to check]

Do not process any elements. Do not begin execution.

PARSED PROBLEM:
{parsed}
"""

AGENT3A_EXECUTOR_SYSTEM = """
You are a mechanical executor.
No discussion. No opinions. No compression. No personas.
A skipped step block is a wrong answer, not a short one.
You do not decide the step format. The Planner decided it in section 6.
Use the Planner's step block format exactly. No substitutions.
"""

AGENT3A_EXECUTOR_PROMPT = """
{output_integrity_rule}

{anti_assumption_rule}

PHASE 2 - ESA EXECUTION

Use EXACTLY the step block format the Planner defined in section 6.

ELEMENT BUDGET RULE:
  Each element must be processed exactly once.
  Before each step ask: "Have I already processed this element?"
  If yes: write SKIP (already processed in Step M) and move on.
  If step count exceeds 3x the input size without termination:
    Write: BUDGET EXCEEDED -- possible loop. Halting.
    Proceed directly to audits.

LOOP PREVENTION RULE:
  If the completeness audit finds missing vertices:
    DO NOT restart from Step 1.
    Find the specific edge that resolves the missing vertex.
    Process only that as the next step block.
    If no such edge exists: write DEAD END: [vertex] unreachable. Halt.

BEFORE Step 1: rewrite the full input from Planner section 4.
Do not begin Step 1 until this list is fully written.

Print the Planner's step block for every element, then print ---

When the termination condition is met, print:

STEP COUNT AUDIT:
  Planner input count : [N]
  Blocks printed      : [count yours]
  Match?              : YES / NO

COMPLETENESS AUDIT:
  All vertices   : [list]
  In solution    : [list]
  Missing        : [NONE or list]

CONSTRAINT AUDIT:
  [ ] Termination condition met?
  [ ] Planner step format used throughout?
  [ ] Every decision verified before committing?
  [ ] Final value correctly derived?

PHASE 2 IS NOW CLOSED.

PLAN:
{plan}
"""

AGENT4A_CRITIC_PROMPT = """
You receive an execution log. You have not seen the original problem or plan.
You do not know what the answer should be.

Audit for the Planner's assigned fault classes below.
For each:
  FAULT CLASS : [name]
  METHOD      : [exactly how you checked]
  FINDING     : PASS or FAIL
  IF FAIL     : Step N -- [what is wrong] -- [correct state]

Also always audit:
  FAULT CLASS : FORMAT-COMPLIANCE
  METHOD      : Check every step block uses only the Planner's fields.
  FINDING     : PASS or FAIL

Output exactly one of:
  VERDICT: NO FAULTS FOUND
or
  VERDICT: FAULT FOUND
  EARLIEST FAULT: Step [N]
  DESCRIPTION: [what is wrong and correct state]

PLANNER FAULT CLASS ASSIGNMENTS:
{fault_classes}

EXECUTION LOG:
{execution}

"""

AGENT5A_SURGEON_PROMPT = """
{anti_assumption_rule}

{output_integrity_rule}

The Critic found a fault at Step {fault_step}.
Steps 1 through {clean_up_to} are confirmed correct. Do not re-execute them.

Last confirmed correct state:
{last_clean_state}

Re-execute from Step {fault_step} onward using the Planner's step format.
Continue until termination condition is met, then print:

CORRECTED COMPLETENESS AUDIT:
  All vertices : [list]
  Covered      : [list]
  Missing      : [NONE or list]

CORRECTED FINAL VALUE: [number]

ORIGINAL LOG:
{execution}

CRITIC REPORT:
{audit}

PLAN:
{plan}
"""

AGENT6A_FINAL_PROMPT = """
You receive a completed execution log and audit report.
Extract the final answer only.

Rules:
  - Answer must come from the final Running value, dp value,
    or CORRECTED FINAL VALUE in the log
  - If the Surgeon ran, use the Surgeon's corrected value
  - Do not recompute. Do not add commentary.

  

Return only this JSON on a single line with nothing after it:
{"answer": "value"}

EXECUTION LOG:
{final_log}

AUDIT REPORT:
{audit}
"""


# ══════════════════════════════════════════════════════════════════════════════
# TRACK B -- EXPLORATORY (Reasoning Explorer)
# ══════════════════════════════════════════════════════════════════════════════

AGENT2B_STRATEGY_PROMPT = """
You receive a combinatorial graph problem.
Prepare the Reasoning Explorer to think well -- not to give it a rigid procedure.

Produce:

1. PROBLEM RESTATEMENT
   Restate the problem in the simplest possible terms.

2. KEY STRUCTURAL OBSERVATIONS
   List 3-5 concrete properties of THIS specific graph instance
   that constrain or guide the search. Cite actual vertex numbers.

3. SEARCH STRATEGY
   Describe the high-level approach:
   - Where should search start and why?
   - What pruning conditions apply?
   - What does a dead end look like?
   - What does a promising branch look like?

4. KNOWN BOUNDS
   Any upper or lower bounds derivable from graph structure without
   full search. E.g. max degree bounds clique size.

5. CHECKPOINTS
   Define 2-3 logical conditions the Explorer should verify
   during its search to confirm it is on the right track.

6. WHAT FAILURE LOOKS LIKE
   The specific mistakes this algorithm commonly makes on this
   problem type so the Explorer can watch for them.

Do not produce step-by-step instructions, nor an algorithm.
Tell the Explorer what to think about, not what to write.

PARSED PROBLEM:
{parsed}
"""

AGENT3B_EXPLORER_SYSTEM = """
You are a reasoning explorer solving a combinatorial graph problem.
You think out loud. You try things. You change your mind.
You are not following a script -- you are genuinely reasoning.
Between checkpoints you may talk freely to yourself, explore ideas,
reconsider assumptions, and think through alternatives. Be very careful
not to reason for too long as this may end up with a truncated output destroying
the entire pipeline. Keep your reasoning precise, but concise without skipping
steps.
"""

AGENT3B_EXPLORER_PROMPT = """
You are solving a combinatorial graph problem through genuine exploration.
You have strategic guidance below.

YOUR APPROACH:
You alternate between FREE THINKING and STATE CHECKPOINTS.

FREE THINKING -- between checkpoints you may:
  - Talk to yourself about what you are noticing
  - Try an approach and reason about whether it works
  - Change your mind and explain why
  - Consider multiple options before committing
  - Ask yourself questions and answer them
  - Notice patterns and test hunches
  There are no formatting rules during free thinking.
  Write whatever helps you reason. Be verbose. Think out loud.

STATE CHECKPOINT -- after each meaningful discovery or decision,
anchor your progress with this structured block:

  ===================================================
  CHECKPOINT [N]
  ===================================================
  Current best solution : [the actual vertices/elements in best solution]
  Current best value    : [size / cost / count of best solution]
  Branches explored     : [what you have tried so far]
  Branches remaining    : [what you still need to try]
  Pruned                : [branches ruled out and the exact reason why]
  VISITED states        : [complete list of states fully explored]
  Key insight           : [most important thing learned so far]
  Next action           : [what you will try next and why]
  ===================================================

CHECKPOINT RULES:
  - Print a checkpoint after every significant decision
  - The VISITED list must only grow -- never shrink or reset
  - Never omit a field from the checkpoint block
  - "same as last checkpoint" is forbidden -- rewrite everything in full

ANTI-LOOP RULE:
  Before exploring any state, scan ALL prior VISITED lists in your
  checkpoints above. If the state appears in any prior VISITED list:
    Write: ALREADY VISITED -- skipping [state]. Trying next branch.
  Do not re-explore any fully examined state.

BACKTRACK PROTOCOL:
  When a branch fails or hits a dead end:
    BACKTRACK: [this branch failed because ...]
    LEARNED: [what this rules out for future branches]
  Return to the last checkpoint and try the next unexplored option.
  Do not restart from scratch. Add the failed state to VISITED.

VERIFICATION BLOCK (required before concluding):
  Before declaring a final answer, write:

  VERIFICATION
  Claimed answer      : [value]
  Solution            : [explicit list of vertices/elements in solution]
  Validity check      : [verify every constraint from the problem statement]
  Optimality argument : [show no better solution can exist -- cite pruning
                        evidence, exhaustion of branches, or bound argument]

  If you cannot write a convincing optimality argument, keep exploring.

CONCLUSION (after verification only):
  CONCLUSION: [final numeric answer]

STRATEGY GUIDANCE:
{strategy}

PARSED PROBLEM:
{parsed}
"""

AGENTX_RESCUER_PROMPT = """

You receive the reasoning of an executer/reasoning explorer that was reasoning on a graph problem.
The executer/explorer ran out of tokens and was not able to return it's final answer. Skim through
it's reasoning and extract what was most likely the final answer in it's reasoning. For example, if it
had stopped after find some answer X but trying to improve, then return the value of X as a json in the 
format {{answer: value}}

Reasoning: {reasoning}

Parsed Problem:
{parsed}

"""


AGENT4B_CRITIC_PROMPT = """
You receive a reasoning exploration log. You have not seen the original
problem or strategy. You do not know what the answer should be.

Audit for these five fault classes:

FAULT CLASS : VISITED-STATE INTEGRITY
  METHOD     : Check every CHECKPOINT VISITED list. Verify it only
               grows. Verify no state in VISITED was explored again.
  FINDING    : PASS or FAIL
  IF FAIL    : Checkpoint N -- [which state was revisited]

FAULT CLASS : BACKTRACK CORRECTNESS
  METHOD     : For every BACKTRACK entry, verify the Explorer returned
               to the correct prior branch and did not skip unexplored
               branches.
  FINDING    : PASS or FAIL
  IF FAIL    : [which backtrack skipped a valid branch]

FAULT CLASS : VERIFICATION COMPLETENESS
  METHOD     : Find the VERIFICATION block. Check both validity AND
               optimality proofs are present and non-trivial.
  FINDING    : PASS or FAIL
  IF FAIL    : [which proof is missing or is just an assertion]

FAULT CLASS : CONCLUSION GROUNDING
  METHOD     : Find CONCLUSION line. Verify it matches the best value
               in the last CHECKPOINT.
  FINDING    : PASS or FAIL
  IF FAIL    : Conclusion says [X], last checkpoint says [Y]

FAULT CLASS : PRUNING VALIDITY
  METHOD     : For each pruned branch, verify the pruning reason is
               logically sound given the problem constraints.
  FINDING    : PASS or FAIL
  IF FAIL    : [which pruning was unjustified and why]

Output exactly one of:
  VERDICT: NO FAULTS FOUND
or
  VERDICT: FAULT FOUND
  EARLIEST FAULT: Checkpoint [N] or [location description]
  DESCRIPTION: [what is wrong]

EXPLORATION LOG:
{execution}
"""

AGENT5B_ADVISOR_PROMPT = """
The Critic found a fault in the exploration.
You are an advisor. You do not re-run the full exploration.
Identify what went wrong and give precise corrective guidance.

Produce:

1. FAULT SUMMARY
   What exactly went wrong in plain terms.

2. CORRECTIVE ACTION
   The specific thing the Explorer should do differently.
   Be concrete -- cite checkpoint numbers and state values.

3. CONTINUATION POINT
   Which checkpoint the Explorer should resume from.
   What the correct state is at that checkpoint.

4. WHAT TO WATCH FOR
   The specific mistake to avoid when re-exploring.

ORIGINAL EXPLORATION LOG:
{execution}

CRITIC FAULT REPORT:
{audit}
"""

AGENT6B_FINAL_PROMPT = """
You receive a reasoning exploration log.
Your only job is to extract the final answer.

Find the CONCLUSION line in the log.
If a CORRECTED FINAL VALUE exists from an advisor correction, use that instead.

Return only this JSON on a single line with nothing after it:
{"answer": "value"}

EXPLORATION LOG:
{final_log}

AUDIT REPORT:
{audit}

In case the answer is unclear, here is the rescuer's answer after reading a truncated log:
{rescue}
"""


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    sep = "=" * 64
    print(f"\n{sep}\n  {title}\n{sep}")


def parse_classifier(text: str) -> dict:
    """Extract track and algorithm from classifier JSON output."""
    try:
        last  = text.rfind("}")
        first = text.rfind("{", 0, last + 1)
        if first != -1 and last != -1:
            return json.loads(text[first:last + 1])
    except (json.JSONDecodeError, ValueError):
        pass
    for line in text.split("\n"):
        if line.strip().upper().startswith("TRACK"):
            t = line.split(":")[-1].strip().upper()
            return {"track": t[0] if t else "A", "algorithm": "unknown"}
    return {"track": "A", "algorithm": "unknown"}


def extract_fault_classes(plan_text: str) -> str:
    """Pull the Planner's fault class block as a raw string."""
    marker = "FAULT CLASS ASSIGNMENTS"
    idx    = plan_text.upper().find(marker)
    if idx != -1:
        return plan_text[idx:idx + 800].strip()
    return (
        "Auditor 1: ORDER -- verify processing order\n"
        "Auditor 2: STATE -- verify state transitions\n"
        "Auditor 3: CONFLICT -- verify no illegal decisions\n"
        "Auditor 4: COMPLETENESS -- verify all elements covered\n"
        "Auditor 5: ARITHMETIC -- recompute final value"
    )


def extract_fault_info(audit_text: str) -> dict:
    """Parse the Critic's output for fault location and last clean state."""
    upper     = audit_text.upper()
    has_fault = "VERDICT: FAULT FOUND" in upper

    fault_step = None
    if has_fault:
        for line in audit_text.split("\n"):
            if "EARLIEST FAULT" in line.upper():
                parts = line.split(":")
                if len(parts) > 1:
                    try:
                        fault_step = int("".join(filter(str.isdigit, parts[1])))
                    except ValueError:
                        fault_step = None
                break

    clean_up_to      = (fault_step - 1) if fault_step and fault_step > 1 else 1
    last_clean_state = "Unknown -- recompute from the beginning"

    if fault_step and fault_step > 1:
        target = f"STEP {clean_up_to}"
        idx    = audit_text.upper().find(target.upper())
        if idx != -1:
            segment = audit_text[idx:idx + 600]
            for line in segment.split("\n"):
                if any(k in line.upper() for k in [
                    "FULL STATE", "DP VALUE", "RUNNING", "VISITED"
                ]):
                    last_clean_state = line.strip()
                    break

    return {
        "has_fault":        has_fault,
        "fault_step":       fault_step or 1,
        "clean_up_to":      clean_up_to,
        "last_clean_state": last_clean_state
    }


def detect_loop(text: str) -> dict:
    """Detect element-reprocessing loops by fingerprinting step block headers."""
    seen   = {}
    blocks = text.split("---")
    for i, block in enumerate(blocks):
        lines       = [l.strip() for l in block.strip().split("\n") if l.strip()]
        fingerprint = "\n".join(lines[:2])
        if fingerprint in seen and len(fingerprint) > 5:
            return {
                "has_loop":          True,
                "first_repeat_step": i + 1,
                "original_step":     seen[fingerprint] + 1
            }
        seen[fingerprint] = i
    return {"has_loop": False}


def detect_checkpoint_loop(text: str) -> dict:
    """Detect revisited states in Track B by scanning VISITED lists."""
    import re
    checkpoints = re.split(r"CHECKPOINT\s+\d+", text, flags=re.IGNORECASE)
    all_visited = set()
    for i, block in enumerate(checkpoints[1:], 1):
        for line in block.split("\n"):
            if "VISITED" in line.upper() and ":" in line:
                states = re.findall(r"\{[^}]+\}|\[[^\]]+\]", line)
                for s in states:
                    if s in all_visited:
                        return {"has_loop": True, "checkpoint": i, "state": s}
                    all_visited.add(s)
                break
    return {"has_loop": False}


def detect_truncation(text: str) -> bool:
    """Return True if the output looks truncated."""
    signals = [
        "PHASE 2 IS NOW CLOSED",
        "COMPLETENESS AUDIT",
        "CONCLUSION:",
        "VERIFICATION"
    ]
    return not any(s in text.upper() for s in signals)


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


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(problem: str, ground_truth=None) -> dict:
    """
    Dual-track 8-agent pipeline.

    SHARED:
      Agent 1     -- Parser
      Agent 1.5   -- Classifier  (outputs JSON to route A or B)
      Agent 7     -- Faithfulness Judge

    TRACK A (Mechanical -- ESA):
      Agent 2A    -- Planner
      Agent 3A    -- ESA Executor
      Agent 4A    -- Critic
      Agent 5A    -- Surgeon (conditional)
      Agent 6A    -- Final Answer

    TRACK B (Exploratory -- Reasoning):
      Agent 2B    -- Strategy Advisor
      Agent 3B    -- Reasoning Explorer
      Agent 4B    -- Critic
      Agent 5B    -- Fault Advisor (conditional)
      Agent 6B    -- Final Answer
    """

    # ── Agent 1: Parser ────────────────────────────────────────────
    section("AGENT 1 -- PARSER")
    parsed = ask_stream(inject(AGENT1_PARSER_PROMPT, problem=problem))

    # ── Agent 1.5: Classifier ──────────────────────────────────────
    section("AGENT 1.5 -- CLASSIFIER")
    classification_text = ask_stream(inject(AGENT_CLASSIFIER_PROMPT, parsed=parsed))
    classification      = parse_classifier(classification_text)
    track               = classification.get("track", "A").upper()
    algorithm           = classification.get("algorithm", "unknown")

    # Sanitize track value
    if track not in ("A", "B"):
        track = "A"

    print(f"\n  --> TRACK: {track}  |  ALGORITHM: {algorithm}")

    # ══════════════════════════════════════════════════════════════
    # TRACK A
    # ══════════════════════════════════════════════════════════════
    if track == "A":

        section("AGENT 2A -- PLANNER (Track A: Mechanical)")
        plan = ask_stream(inject(AGENT2A_PLANNER_PROMPT, parsed=parsed))

        section("AGENT 3A -- ESA EXECUTOR (Track A)")
        execution = ask_stream(
            inject(AGENT3A_EXECUTOR_PROMPT,
                output_integrity_rule=OUTPUT_INTEGRITY_RULE,
                anti_assumption_rule=ANTI_ASSUMPTION_RULE,
                plan=plan),
            system=AGENT3A_EXECUTOR_SYSTEM
        )

        if detect_truncation(execution):
            print("\n  WARNING: Executor output appears truncated.")

        loop_info = detect_loop(execution)
        if loop_info["has_loop"]:
            section(f"  LOOP DETECTED at Step {loop_info['first_repeat_step']} -- RETRYING")
            execution = ask_stream(
                inject(AGENT3A_EXECUTOR_PROMPT,
                    output_integrity_rule=OUTPUT_INTEGRITY_RULE,
                    anti_assumption_rule=ANTI_ASSUMPTION_RULE,
                    plan=plan) +
                "\n\nWARNING: Prior attempt reprocessed element at Step " +
                str(loop_info['first_repeat_step']) +
                " (already done at Step " + str(loop_info['original_step']) + ")." +
                " Each element exactly once. Do not revisit prior steps.",
                system=AGENT3A_EXECUTOR_SYSTEM
            )
        
        section("AGENT X -- RESCUE")
        rescue = ask_stream(inject(
            AGENTX_RESCUER_PROMPT,
            reasoning=execution,
            parsed=parsed
        ))

        section("AGENT 4A -- CRITIC (Track A)")
        fault_classes = extract_fault_classes(plan)
        audit = ask_stream(inject(AGENT4A_CRITIC_PROMPT,
            fault_classes=fault_classes,
            execution=execution
        ))

        final_log  = execution
        fault_info = extract_fault_info(audit)

        if fault_info["has_fault"]:
            section(f"AGENT 5A -- SURGEON (fault at Step {fault_info['fault_step']})")
            final_log = ask_stream(
                inject(AGENT5A_SURGEON_PROMPT,
                    anti_assumption_rule=ANTI_ASSUMPTION_RULE,
                    output_integrity_rule=OUTPUT_INTEGRITY_RULE,
                    fault_step=str(fault_info["fault_step"]),
                    clean_up_to=str(fault_info["clean_up_to"]),
                    last_clean_state=fault_info["last_clean_state"],
                    execution=execution,
                    audit=audit,
                    plan=plan),
                system=AGENT3A_EXECUTOR_SYSTEM
            )
        else:
            section("AGENT 5A -- SURGEON (skipped -- no fault found)")
            print("  Critic confirmed no faults.")

        section("AGENT 6A -- FINAL ANSWER (Track A)")
        raw_answer = ask_stream(inject(AGENT6A_FINAL_PROMPT,
            final_log=final_log,
            rescue=rescue,
            audit=audit
        ))

    # ══════════════════════════════════════════════════════════════
    # TRACK B
    # ══════════════════════════════════════════════════════════════
    else:

        section("AGENT 2B -- STRATEGY ADVISOR (Track B: Exploratory)")
        strategy = ask_stream(inject(AGENT2B_STRATEGY_PROMPT,
            algorithm=algorithm,
            parsed=parsed
        ))

        section("AGENT 3B -- REASONING EXPLORER (Track B)")
        execution = ask_stream(
            inject(AGENT3B_EXPLORER_PROMPT,
                strategy=strategy,
                parsed=parsed),
            system=AGENT3B_EXPLORER_SYSTEM,
            temperature=0.3
        )

        if detect_truncation(execution):
            print("\n  WARNING: Explorer output appears truncated.")

        cp_loop = detect_checkpoint_loop(execution)
        if cp_loop["has_loop"]:
            section(f"  CHECKPOINT LOOP at Checkpoint {cp_loop['checkpoint']} -- RETRYING")
            execution = ask_stream(
                inject(AGENT3B_EXPLORER_PROMPT,
                    strategy=strategy,
                    parsed=parsed) +
                "\n\nWARNING: Prior attempt revisited state " +
                str(cp_loop["state"]) +
                " at Checkpoint " + str(cp_loop["checkpoint"]) +
                ". Check VISITED before every exploration. Do not re-explore.",
                system=AGENT3B_EXPLORER_SYSTEM,
                temperature=0.3
            )

        plan = strategy

        section("AGENT X -- RESCUE")
        rescue = ask_stream(inject(
            AGENTX_RESCUER_PROMPT,
            reasoning=execution,
            parsed=parsed
        ))

        section("AGENT 4B -- CRITIC (Track B)")
        audit = ask_stream(inject(AGENT4B_CRITIC_PROMPT, execution=execution))

        final_log = execution

        if "VERDICT: FAULT FOUND" in audit.upper():
            section("AGENT 5B -- FAULT ADVISOR (fault found)")
            advice = ask_stream(inject(AGENT5B_ADVISOR_PROMPT,
                execution=execution,
                audit=audit
            ))

            section("AGENT 3B -- REASONING EXPLORER RETRY (with corrections)")
            final_log = ask_stream(
                inject(AGENT3B_EXPLORER_PROMPT,
                    strategy=strategy + "\n\nCORRECTIVE GUIDANCE FROM ADVISOR:\n" + advice,
                    parsed=parsed),
                system=AGENT3B_EXPLORER_SYSTEM,
                temperature=0.3
            )
        else:
            section("AGENT 5B -- FAULT ADVISOR (skipped -- no fault found)")
            print("  Critic confirmed no faults.")

        section("AGENT 6B -- FINAL ANSWER (Track B)")
        raw_answer = ask_stream(inject(AGENT6B_FINAL_PROMPT,
            final_log=final_log,
            audit=audit,
            rescue=rescue
        ))

    # ── Agent 7: Faithfulness Judge (both tracks) ──────────────────
    section("AGENT 7 -- FAITHFULNESS JUDGE")
    result = extract_json_answer(raw_answer)

    faithfulness = ask_stream(inject(AGENT7_FAITHFULNESS_PROMPT,
        final_log=final_log,
        audit=audit,
        answer=json.dumps(result)
    ))

    verdict = "UNKNOWN"
    score   = "?/7"
    for line in faithfulness.split("\n"):
        u = line.upper()
        if "OVERALL VERDICT" in u:
            if   "HIGH"   in u: verdict = "HIGH"
            elif "MEDIUM" in u: verdict = "MEDIUM"
            elif "LOW"    in u: verdict = "LOW"
        if "FAITHFULNESS SCORE" in u:
            score = line.split(":")[-1].strip()

    result["track"]                = track
    result["algorithm"]            = algorithm
    result["faithfulness_verdict"] = verdict
    result["faithfulness_score"]   = score

    if ground_truth is not None:
        try:
            correct = str(result.get("answer", "")).strip() == str(ground_truth).strip()
            result["correct"]      = correct
            result["ground_truth"] = ground_truth
            status = "CORRECT" if correct else "WRONG"
            print(f"\n  Ground truth: {ground_truth}  -->  {status}")
        except Exception:
            pass

    section("FINAL RESULT")
    print(json.dumps(result, indent=2))
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Swap problem and ground truth here ─────────────────────────

    # Track A example -- MST (uncomment to use)
    # PROBLEM = """
    # Find the total weight of the minimum spanning tree.
    # Undirected weighted graph with 16 vertices and 20 edges.
    # Edges (Node Node Weight):
    # 1 2 22, 1 3 69, 2 13 13, 1 7 58, 1 14 89, 1 16 91,
    # 13 12 47, 13 5 88, 2 10 62, 10 9 98, 1 6 51, 10 8 48,
    # 14 4 10, 4 15 29, 15 11 52, 9 1 32, 15 10 9,
    # 11 7 71, 8 11 35, 14 11 77
    # """
    # GROUND_TRUTH = "664"

    # Track B example -- MIS on tree (Classifier routes this to Track A via Tree DP)
    PROBLEM =  "Please provide the reasoning process and the final answer directly to the question.\n\nFind the size of the maximum clique in the given graph. A clique is a subset of vertices such that every two distinct vertices are adjacent.\n\nThis is an undirected graph with 16 vertices and 32 edges.\n\n- Adjacency List: For each vertex, lists all connected vertices.\n\nAdjacency List:\n\nVertex 1: 16, 3, 8\nVertex 2: 16, 9, 5\nVertex 3: 6, 1, 13\nVertex 4: 6, 14, 9, 15, 12, 11\nVertex 5: 6, 10, 14, 2, 15\nVertex 6: 15, 7, 5, 3, 4, 14, 8, 12\nVertex 7: 6, 16\nVertex 8: 16, 10, 6, 1\nVertex 9: 10, 4, 2, 13\nVertex 10: 8, 9, 11, 5\nVertex 11: 10, 4\nVertex 12: 16, 15, 6, 4\nVertex 13: 15, 9, 3\nVertex 14: 4, 6, 5\nVertex 15: 12, 6, 13, 5, 4\nVertex 16: 1, 2, 12, 8, 7\n",


    GROUND_TRUTH = "4"

    run_pipeline(PROBLEM, ground_truth=GROUND_TRUTH)