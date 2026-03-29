"""Web-search tool-call graders — stub.

Can be used to evaluate web-search result coverage, such as verifying that
expected findings appear in the agent's web-search responses.

See ``CONTRIBUTING_EVALUATION.md`` for guidance on adding graders.
"""
import json
import re
import csv
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
import csv
import argparse
from dotenv import load_dotenv
load_dotenv() 


GEMINI_MODEL = "gemini-2.5-flash"
SEARCH_TOOL = "web_search"
QUERY_FIELD = "query"
API_KEY = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))

client = genai.Client(api_key=API_KEY)

def extract_search_events(log_path:Path):
    """
    Parse a single entity JSON file and return one record per web_search call.
 
    Each record contains everything downstream evaluators need:
      - query : the search string issued by the agent
      - response_text : the full markdown blob returned by the search tool
      - cited_sources : structured list parsed from the CITABLE SOURCES block
      - agent_reasoning : the agent's thought text immediately before the search
      - subject : the entity being investigated 
      - session_id : unique run identifier 
    """
    with open(log_path, encoding="utf-8") as f:
        doc = json.load(f)
 
    subject = doc.get("subject", "")
    session_id = doc.get("session_id", "")
 
    reasoning_by_query = _extract_reasoning_from_llm_calls(doc)
 
    events = []
    for i, call in enumerate(doc.get("tool_calls", [])):
        if call.get("tool") != SEARCH_TOOL:
            continue
 
        query = call.get("args", {}).get(QUERY_FIELD, "").strip()
        response = call.get("response", "")
 
        if not query:
            continue
 
        events.append({
            "source_file": log_path.name,
            "session_id": session_id,
            "subject": subject,
            "timestamp": call.get("timestamp", ""),
            "query": query,
            "response_text": response,
            "cited_sources": _parse_cited_sources(response),
            "agent_reasoning": reasoning_by_query.get(query, ""),
        })
 
    return events
 
 
def _extract_reasoning_from_llm_calls(doc:dict):
    """
    Walk the llm_call_history conversation turns and map each web_search query to the
    agent's reasoning thought that immediately preceded it.
    """
    reasoning = {}
    llm_section = doc.get("llm_call_history", {})
 
    calls = llm_section.get("calls", []) if isinstance(llm_section, dict) else []
 
    for call in calls:
        parts = call.get("response", {}).get("content", {}).get("parts", [])
        thought_text = None
 
        for part in parts:
            if part.get("thought") is True and part.get("type") == "text":
                thought_text = part.get("text", "").strip()
 
            elif part.get("type") == "function_call" and part.get("name") == SEARCH_TOOL:
                query = part.get("args", {}).get(QUERY_FIELD, "")
                if query and thought_text:
                    reasoning[query] = thought_text

                thought_text = None
 
    return reasoning
 
 
def _parse_cited_sources(response_text):
    """
    Extract structured sources from the CITABLE SOURCES block at the bottom
    of the search response.
    Titles may be blank or "&nbsp;". Only lines with a valid http URL are kept.
    Returns a list of {title, url, snippet} dicts.
    """
    sources = []
    block_match = re.search(
        r"CITABLE SOURCES.*?(?=\Z)", response_text, re.DOTALL | re.IGNORECASE
    )
    if not block_match:
        return sources
 
    for line in block_match.group().splitlines():
        line = line.strip().lstrip("- ").strip()
        if "|" not in line:
            continue
 
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            title = parts[0] if parts[0] not in ("", "&nbsp;") else ""
            url = parts[1] if len(parts) > 1 else ""
            snippet = parts[2] if len(parts) > 2 else ""
            if url.startswith("http"):
                sources.append({"title": title, "url": url, "snippet": snippet})
 
    return sources
 
 
def load_all_files(input_path):
    """
    Take in either a single .json file path or a directory.
    If a directory, loads all *.json files within it.
    """
    p = Path(input_path)
    files = [p] if p.is_file() else sorted(p.glob("*.json"))
 
    if not files:
        raise FileNotFoundError(f"No JSON files found at: {input_path}")
 
    all_events = []
    for f in files:
        try:
            events = extract_search_events(f)
            all_events.extend(events)
            print(f"  {f.name}: {len(events)} web_search call(s)")
        except Exception as e:
            print(f"  WARNING: skipping {f.name} — {e}")
 
    return all_events


STOP_WORDS  = {"the","a","an","of","and","or","is","in","to","for",
               "what","how","why","do","does","with","on","at","by"}
VAGUE_TERMS = {"information","details","stuff","things","data",
               "general","overview","about","related","various"}


def rule_based_eval(query: str, cited_sources: list[dict]):
    """
    Rule-base evaluation. 
    Metrics (score starts form 5.0): 
    - how many stop words in the search query. 
    - how many vague words. 
    - whether the search query is too long or too short. 
    - whether or not the entity name is in the search query. 
    - if there's cited url source. 
    - whether any of the url is a link to government or any other official webpage. 
    """
    tokens = query.lower().split()
    n = len(tokens)
    stop_ratio = sum(1 for t in tokens if t in STOP_WORDS) / max(n, 1)
    has_vague = any(t in VAGUE_TERMS for t in tokens)
    has_entity = bool(re.search(r'(?<=\s)[A-Z][a-zA-Z]+', query))
    too_short = n < 2
    too_long = n > 12

    num_sources = len(cited_sources)
    has_gov_source = any(
        re.search(r'\.(gov|justice\.gov|treasury\.gov|ofac)', s.get("url", ""), re.I)
        for s in cited_sources
    )
    no_sources = num_sources == 0

    flags = []
    if stop_ratio > 0.4: flags.append("high_stop_word_ratio")
    if has_vague: flags.append("vague_terms")
    if too_short: flags.append("search_query_too_short")
    if too_long: flags.append("search_query_too_long")
    if no_sources: flags.append("no_cited_sources")

    score = 5.0
    score -= 1.25 * (stop_ratio > 0.4) # too many stop words
    score -= 1.00 * has_vague # too many vague words 
    score -= 1.25 * too_short
    score -= 0.50 * too_long
    score += 0.50 * has_entity
    score -= 1.00 * no_sources # no url sources 
    score += 0.25 * has_gov_source
    score = round(max(0.0, min(5.0, score)), 3)

    return {
        "rule_score": score,
        "num_tokens": n, # length of search query 
        "stop_word_ratio": round(stop_ratio, 3),
        "has_named_entity": has_entity,
        "num_cited_sources": num_sources,
        "has_gov_source": has_gov_source,
        "flags": flags,
    }



def load_ground_truth(gt_path: str):
    """
    Read the ground truth CSV and return a dict keyed by test_case_id.
        {"TC-001": {"expected_findings": ["Entity has a business address in ...", "Entity's legal name is ...",]},...}
    """
    ground_truth = {}

    with open(gt_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tc_id = row.get("test_case_id", "").strip()
            if not tc_id:
                continue

            raw_expected = row.get("expected_open_search_results", "[]").strip()
            try:
                expected_findings = json.loads(raw_expected)
            except json.JSONDecodeError:
                expected_findings = [raw_expected] if raw_expected else []

            ground_truth[tc_id] = {
                "test_case_id": tc_id,
                "expected_findings": expected_findings,
            }

    print(f"  Loaded {len(ground_truth)} ground truth record(s): "
          f"{list(ground_truth.keys())}")
    return ground_truth


def _match_ground_truth(json_filename: str, ground_truth: dict):
    """
    Match a test case JSON filename to its ground truth record
    by checking if any test_case_id is a substring of the filename.

    Returns None if no match found.
    """
    for tc_id, val in ground_truth.items():
        if tc_id in json_filename:
            return val

    return None

## LLM-as-judge 

COMBINED_JUDGE_PROMPT = """\
You are an expert evaluator of agentic web search behaviour for an \
Anti-Money Laundering (AML) due diligence tool.
 
The agent investigates entities for AML risk. It uses web search to find \
adverse media, sanctions exposure, PEP status, financial crime records etc..
 
---
 
## Request for investigation 
{subject}
 
## Agent's reasoning before issuing this search
{agent_reasoning}
 
## Search query issued
{query}
 
## Cited sources extracted from the search response
{sources_block}
 
## Web search response
{response_preview}
 
## Expected findings (ground truth)
These are the facts this search should ideally have surfaced. Use them to \
inform your evaluation of query quality and source relevancy, and assess \
coverage directly in the recall dimension.
{expected_findings_block}
 
---
 
Evaluate three dimensions and return ONLY the JSON below — \
no markdown fences, no extra keys:
 
1. query_quality (1-5)
   Does the query efficiently and specifically target AML-relevant information?
   Evaluate based on how the query was constructed, not solely on whether
   expected findings were returned — a well-formed query may still miss findings
   due to limitations of the search tool.
   5 = precise, well-formed — a skilled AML analyst would write this
   4 = good, minor improvements possible
   3 = reasonable but too broad or missing a key discriminating term
   2 = somewhat relevant but likely to retrieve noise
   1 = vague, off-topic, or not useful for AML purposes
 
2. source_relevancy (1-5)
   Do the cited sources appear credible and directly useful for AML due diligence?
   5 = highly relevant, authoritative (gov, regulators, reputable news)
   4 = mostly relevant with minor noise
   3 = mixed quality or only tangentially related
   2 = mostly irrelevant or low-credibility
   1 = no sources, entirely irrelevant, or misleading
 
3. recall (only if expected findings are provided, otherwise null)
   What fraction of the expected findings does the search response cover,
   even if paraphrased, substring matched, or semantically equivalent?
 
{{
  "query_quality_score": <int 1-5>,
  "query_quality_rationale": "<one concise sentence>",
  "source_relevancy_score": <int 1-5>,
  "source_relevancy_rationale": "<one concise sentence>",
  "recall_score": <float 0.0-1.0, or null if no expected findings provided>,
  "per_finding": [
    {{
      "expected": "<the expected finding text>",
      "covered": <true or false>,
      "evidence": "<one short sentence referencing the part of the response \
that covers it, or 'Not found' if absent>"
    }}
  ]
}}
"""
 
def _format_sources(sources: list[dict]):
    """Format cited sources into a readable block for the judge prompt."""
    if not sources:
        return "No cited sources extracted."
    lines = []
    for i, s in enumerate(sources, 1):
        title = s.get("title") or "(no title)"
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {s.get('url', '')}")
        snippet = s.get("snippet", "")
        if snippet:
            lines.append(f"   {snippet[:200]}")
    return "\n".join(lines)
 
 
def llm_eval(event: dict, expected_findings: list[str], retries: int = 3):
    """
    Run the combined LLM judge for a single search event.
 
    Inputs:
      event — parsed search event dict from extract_search_events()
      expected_findings — list of ground truth finding strings (may be empty)
      retries — number of attempts on transient API / parse errors
 
    Returns a dict with keys:
      query_quality_score, query_quality_rationale,
      source_relevancy_score, source_relevancy_rationale,
      recall_score, per_finding
 
    On unrecoverable failure, returns the same keys with None / empty values
    so downstream code never has to handle a missing key.
    """
    # response_text is scoped exclusively to the web_search tool call by the
    # parser — it contains no KB search or SQL output.
    web_search_response = event["response_text"][:6000].strip()
    if len(event["response_text"]) > 6000:
        web_search_response += "\n... [truncated]"
 
    reasoning = event["agent_reasoning"][:800].strip()
    if len(event["agent_reasoning"]) > 800:
        reasoning += "\n... [truncated]"
 
    if expected_findings:
        findings_block = "\n".join(
            f"{i+1}. {f}" for i, f in enumerate(expected_findings)
        )
    else:
        findings_block = "No ground truth provided for this subject."
 
    # print("subject: ", event["subject"], "\n",)
    prompt = COMBINED_JUDGE_PROMPT.format(
        subject = event["subject"],
        agent_reasoning = reasoning or "Not available.",
        query = event["query"],
        sources_block = _format_sources(event["cited_sources"]),
        response_preview = web_search_response,
        expected_findings_block = findings_block,
    )
 
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            raw = resp.text.strip()
 
            # Strip markdown fences that the model sometimes adds despite instructions
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            result = json.loads(raw)
 
            # Normalise: ensure per_finding is always a list
            if "per_finding" not in result:
                result["per_finding"] = []
 
            # If no ground truth was provided, enforce 0 recall regardless of
            # what the model returned — avoids hallucinated per_finding entries
            if not expected_findings:
                result["recall_score"] = 0
                result["per_finding"]  = []
 
            return result
 
        except Exception as e:
            if attempt == retries - 1:
                # Return a well-shaped failure dict so output writers never KeyError
                return {
                    "query_quality_score":        None,
                    "query_quality_rationale":    f"Eval error: {e}",
                    "source_relevancy_score":     None,
                    "source_relevancy_rationale": "",
                    "recall_score":               None,
                    "per_finding":                [],
                }
            
            time.sleep(2 ** attempt)

 
def evaluate(input_path: str,
             ground_truth_path: str | None = None):
    """
    Run the full evaluation pipeline and return a result dict containing:
      - summary  : aggregate metrics across all search events
      - details  : one record per search event with all eval scores
 
    ground_truth_path is optional. If omitted, or if a session's subject
    name does not match any ground truth row, recall_score will be None
    for that event.
    """
    print(f"\nParsing logs from: {input_path}")
    events = load_all_files(input_path)
    print(f"\nTotal web_search calls to evaluate: {len(events)}\n")
 
    ground_truth = {}
    if ground_truth_path:
        print("Loading ground truth...")
        ground_truth = load_ground_truth(ground_truth_path)
 
    results = []
    for i, ev in enumerate(events):
        print(f"  [{i+1}/{len(events)}]  {ev['subject']}  |  '{ev['query']}'")
 
        # fast rule-based eval (no API cost)
        rule = rule_based_eval(ev["query"], ev["cited_sources"])
 
        # match ground truth by subject name
        gt_record = _match_ground_truth(ev["source_file"], ground_truth)
        expected_findings = gt_record["expected_findings"] if gt_record else []
        test_case_id = gt_record["test_case_id"] if gt_record else None
 
        # combined LLM judge (one API call covers all three dimensions)
        llm = llm_eval(ev, expected_findings)
 
        results.append({
            "task": ev["subject"],
            "source_file": ev["source_file"],
            "test_case_id": test_case_id,
            "session_id": ev["session_id"],
            "search terms": ev["query"],
            "expected_findings": expected_findings,
            "rule_eval": rule,
            "llm_eval": llm,
            "final_score": rule["rule_score"] + llm["query_quality_score"] + llm["source_relevancy_score"] + llm["recall_score"] * 5
        })
 
        time.sleep(1)
 
    return _build_output(results)
 
 
def _build_output(results: list[dict]):
    """Compute summary statistics and package results for output writers."""
 
    valid = [r for r in results if r["llm_eval"]["query_quality_score"] is not None]
    recall_valid = [r for r in results if r["llm_eval"].get("recall_score") is not None]
 
    summary = {
        "total_searches": len(results),
        "llm_eval_failures": len(results) - len(valid),
        "avg_rule_score": _avg(results, lambda r: r["rule_eval"]["rule_score"]),
        "avg_query_quality": _avg(valid, lambda r: r["llm_eval"]["query_quality_score"]),
        "avg_source_relevancy": _avg(valid, lambda r: r["llm_eval"]["source_relevancy_score"]),
        "avg_recall_score": _avg(recall_valid, lambda r: r["llm_eval"]["recall_score"]),
        "sessions_with_gt": len(recall_valid),
        "flagged_count": sum(1 for r in results if r["rule_eval"]["flags"]),
        "flag_breakdown": _flag_breakdown(results),
    }
    return {"summary": summary, "details": results}
 
 
def _avg(items: list, fn):
    """Return the mean of fn(item) over items, or None if items is empty."""
    if not items:
        return None
    return round(sum(fn(r) for r in items) / len(items), 2)
 
 
def _flag_breakdown(results: list[dict]):
    """Count how many times each rule flag appears across all events."""
    counts = {}
    for r in results:
        for flag in r["rule_eval"]["flags"]:
            counts[flag] = counts.get(flag, 0) + 1
    return counts


def write_json(output: dict, path: str = "eval_results.json"):
    """
    Write full eval results to JSON.
    """
    slim = []
    for r in output["details"]:
        slim.append({k: v for k, v in r.items()
                     if k not in ("response_text", "agent_reasoning")})
    out = {**output, "details": slim}
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  JSON  →  {path}")
 
 
def write_csv(output: dict, path: str = "eval_results.csv"):
    """
    Write one row per search event to CSV.
    cited_source_urls collapses all source URLs into a pipe-separated string.
    """
    fieldnames = [
        "test_case_id",
        "source_file", "session_id", "task", 
        # Search query
        "search_terms",
        # Rule-based eval
        "rule_score", "num_tokens", "stop_word_ratio",
        "has_named_entity", 
        "num_cited_sources", "cited_source_urls", "has_gov_source", "flags",
        # LLM judge — query quality
        "query_quality_score", "query_quality_rationale",
        # LLM judge — source relevancy
        "source_relevancy_score", "source_relevancy_rationale",
        # LLM judge — ground truth recall
        "recall_score", "num_expected_findings", "num_covered_findings",
        # Final score - sum of scores from rule-based and LLM judge
        "final_score", 
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in output["details"]:
            urls = " | ".join(s.get("url", "") for s in r.get("cited_sources", []))
            pf = r["llm_eval"].get("per_finding", [])
            n_exp = len(pf)
            n_cover = sum(1 for f in pf if f.get("covered"))
            w.writerow({
                "test_case_id": r.get("test_case_id") or "",
                "source_file": r["source_file"],
                "session_id": r["session_id"],
                "task": r["task"],
                "search_terms": r["search terms"],
                "rule_score": r["rule_eval"]["rule_score"],
                "num_tokens": r["rule_eval"]["num_tokens"],
                "stop_word_ratio": r["rule_eval"]["stop_word_ratio"],
                "has_named_entity": r["rule_eval"]["has_named_entity"],
                "num_cited_sources": r["rule_eval"]["num_cited_sources"],
                "cited_source_urls": urls,
                "has_gov_source": r["rule_eval"]["has_gov_source"],
                "flags": "; ".join(r["rule_eval"]["flags"]),
                "query_quality_score": r["llm_eval"]["query_quality_score"],
                "query_quality_rationale": r["llm_eval"]["query_quality_rationale"],
                "source_relevancy_score": r["llm_eval"]["source_relevancy_score"],
                "source_relevancy_rationale": r["llm_eval"]["source_relevancy_rationale"],
                "recall_score": r["llm_eval"].get("recall_score"),
                "num_expected_findings": n_exp,
                "num_covered_findings": n_cover,
                "final_score": r["final_score"], }
            )
    print(f"  CSV   →  {path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AML search results.")
    parser.add_argument("input_path", help="Path to input directory or file")
    parser.add_argument("gt_path", help="Path to ground truth CSV")
    parser.add_argument("--out-json", default="eval_results.json", help="Output JSON path")
    parser.add_argument("--out-csv", default="eval_results.csv", help="Output CSV path")
    args = parser.parse_args()

    output = evaluate(args.input_path, args.gt_path)

    print("\nWriting outputs...")
    write_json(output, path=args.out_json)
    write_csv(output, path=args.out_csv)
 
    s = output["summary"]
    print(f"\n{'─' * 42}")
    print(f"  Searches evaluated:   {s['total_searches']}")
    print(f"  Avg rule score:       {s['avg_rule_score']}")
    print(f"  Avg query quality:    {s['avg_query_quality']} / 5")
    print(f"  Avg source relevancy: {s['avg_source_relevancy']} / 5")
    print(f"  Avg recall score:     {s['avg_recall_score']}")
    print(f"  Flagged queries:      {s['flagged_count']}")
    print(f"{'─' * 42}")
 