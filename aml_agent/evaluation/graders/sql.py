"""SQL tool-call graders — stub.

Can be used to evaluate SQL quality, such as query validity, read-only
adherence, or time-window filtering.

See ``CONTRIBUTING_EVALUATION.md`` for guidance on adding graders.
"""

from typing import Any
from .llm_judge import run_llm_judge_structured, build_judge_error_evaluation, LLMJudgeConfig
from aml_agent.evaluation.types import Evaluation
import re
from dataclasses import replace

formatted_sql_template = """
[SQL]
─────────────────────────────────────────────────────────
{}

"""

SQL_EVALUATOR_USER_PROMPT = """

════════════════════════════════════════
INPUT
════════════════════════════════════════

USER INTENT (free text):
"{intent}"

DATABASE SCHEMA:
{schema}

List of GENERATED SQL {total_sql_statements}:
{formatted_generated_sql}
"""

SQL_EVALUATOR_PROMPT = """
You are a strict SQL evaluation judge. Your job is to evaluate whether 
a generated SQL query correctly fulfills a user's free-text search intent, 
given the database schema that was used to produce it.

No reference SQL is provided. You must reason from first principles using 
only the schema and the user's intent.

════════════════════════════════════════
EVALUATION INSTRUCTIONS
════════════════════════════════════════

Before scoring, reason step by step through these questions:

STEP 1 — Entity Resolution
  - What real-world entity is the user searching for? 
    (e.g. "customers", "products", "orders")
  - Which table in the schema best represents that entity?
  - Does the generated SQL query that table? If not, this is a 
    critical failure.

STEP 2 — Column Resolution
  - What attribute is the user filtering on? 
    (e.g. name, description, city, email, status)
  - Which column in that table best maps to that attribute?
  - Does the generated SQL filter on that column?

STEP 3 — LIKE Pattern Correctness
  Determine the match type from the user's intent:
    • "contains X"       → LIKE '%X%'
    • "starts with X"    → LIKE 'X%'
    • "ends with X"      → LIKE '%X'
    • "is exactly X"     → = 'X'  (not LIKE)
  - Does the generated SQL use the correct pattern?
  - Is case sensitivity handled? (LOWER(), UPPER(), or ILIKE preferred)
  - Are wildcards placed correctly?

STEP 4 — SQL Validity
  - Do all referenced tables exist in the schema?
  - Do all referenced columns exist in those tables?
  - Is the SQL syntax valid (no missing keywords, malformed clauses)?
  - Is there exactly one SELECT statement?

STEP 5 — Quality & Safety
  - Does it use explicit column names rather than SELECT *?
  - Is there a LIMIT clause? (absence is a warning, not a failure)
  - Is the LIKE value hardcoded safely (no concatenation risk)?
  - Is the query readable and free of unnecessary complexity?

════════════════════════════════════════
SCORING RUBRIC
════════════════════════════════════════

Score each dimension from 1 to 10 using these anchors:

intent_alignment (weight: highest)
  10 — Correct table, correct column, query fully captures intent
   7 — Correct table, minor column ambiguity but defensible
   4 — Wrong column but right table
   1 — Wrong table entirely

like_pattern (weight: high)
  10 — Correct pattern type, correct wildcards, case handled
   7 — Correct pattern type, wildcards correct, no case handling
   4 — Pattern type wrong (e.g. starts-with used for contains)
   1 — No LIKE used at all, or pattern is inverted/broken

sql_correctness (weight: medium)
  10 — All tables/columns valid, syntax clean, no errors
   6 — Minor issue (e.g. unnecessary join, redundant clause)
   2 — References non-existent column or table
   1 — Query would not execute
   0 — if no query is present

safety_quality (weight: low)
  10 — Explicit columns, LIMIT present, clean pattern
   7 — Explicit columns, no LIMIT
   4 — SELECT * used
   1 — SELECT * and no LIMIT and complex unnecessary logic

════════════════════════════════════════
PASS / FAIL THRESHOLD
════════════════════════════════════════

PASS requires ALL of the following:
  • intent_alignment  >= 7
  • like_pattern      >= 6
  • sql_correctness   >= 6
  • safety_quality    >= 4

If ANY condition is not met → verdict is FAIL.

════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════

Respond ONLY with a valid JSON object for the sql statements provided. No markdown. No explanation 
outside the JSON. No preamble.

If there are mutliple SQL statments, then summarize all the results into a single json with scores averaged for all. 

{
  "reasoning": {
    "entity_resolved":  "<what entity the user wants, e.g. 'customers'>",
    "table_chosen":     "<correct table from schema>",
    "column_chosen":    "<correct column for the filter>",
    "match_type":       "<contains | starts_with | ends_with | exact>",
    "expected_pattern": "<the ideal LIKE pattern, e.g. '%anna%'>",
    "case_handling":    "<required | not_required>",
    "agent_table":      "<table the agent actually queried>",
    "agent_column":     "<column the agent actually filtered on>",
    "agent_pattern":    "<LIKE pattern the agent used>"
  },
  "scores": {
    "intent_alignment": <1-10>,
    "like_pattern":     <1-10>,
    "sql_correctness":  <1-10>,
    "safety_quality":   <1-10>
  },
  "verdict": "<PASS|FAIL>",
  "verdict_reason": "<one sentence explaining the verdict>",
  "findings": [
    { "level": "<good|warn|bad>", "text": "<specific, actionable finding>" }
  ]
}
"""


def sql_quality_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,   
):
    text_input = input.get("test_case_info_input")

    get_schema_calls = list(
        filter(
            lambda x: x['tool'] == "get_schema_info"
            , output.get("tool_calls", [])
        )
    )
    latest_schema_response = get_schema_calls[-1] if len(get_schema_calls) > 0 else None

    sql_tool_calls = list(
        filter(lambda x: x["tool"] == "execute",
               output.get("tool_calls", [])
               )
    )
    if latest_schema_response is None or len(sql_tool_calls)==0:
        return [Evaluation(
            name="sql_quality",
            value=round(0, 2),
            comment="schema or sql not returned by agent",
        )]

    schema = latest_schema_response.get("response", None)
    sqls = list(map(lambda x: x.get("args", {}).get("query", None), sql_tool_calls))

    try:
        judge_response = run_llm_judge_structured(
                metric_name="SQL_QUALITY",
                system_prompt=SQL_EVALUATOR_PROMPT,
                user_prompt=SQL_EVALUATOR_USER_PROMPT.format(
                    intent=text_input,
                    schema=schema,
                    total_sql_statements=len(sqls),
                    formatted_generated_sql="\n".join(map(lambda x: formatted_sql_template.format(x), sqls))
                ),
                config=replace(LLMJudgeConfig(), max_output_tokens=8096)
        )
        scores = judge_response.get("scores", {"key": -1})
        score = sum(scores.values()) / len(scores)
        comment = judge_response.get("verdict", "") +" "+ judge_response.get("verdict_reason", "")
        metadata = judge_response.get("reasoning")
    except Exception as e:
        return [
            build_judge_error_evaluation(metric_name="sql_quality", error=e)
        ]

    return [
        Evaluation(
            name="sql_quality",
            value=round(score, 2),
            comment=comment,
            metadata=metadata
        ),
    ]

ALTERING_RULES = [
    {
        "id": "R01",
        "name": "INSERT statement",
        "pattern": r"\bINSERT\b",
        "severity": "critical",
    },
    {
        "id": "R02",
        "name": "UPDATE statement",
        "pattern": r"\bUPDATE\b",
        "severity": "critical",
    },
    {
        "id": "R03",
        "name": "DELETE statement",
        "pattern": r"\bDELETE\b",
        "severity": "critical",
    },
    {
        "id": "R04",
        "name": "DROP statement",
        "pattern": r"\bDROP\b",
        "severity": "critical",
    },
    {
        "id": "R05",
        "name": "TRUNCATE statement",
        "pattern": r"\bTRUNCATE\b",
        "severity": "critical",
    },
    {
        "id": "R06",
        "name": "ALTER statement",
        "pattern": r"\bALTER\b",
        "severity": "critical",
    },
    {
        "id": "R07",
        "name": "CREATE statement",
        "pattern": r"\bCREATE\b",
        "severity": "critical",
    },
    {
        "id": "R08",
        "name": "REPLACE statement",
        "pattern": r"\bREPLACE\b",
        "severity": "high",
    },
    {
        "id": "R09",
        "name": "MERGE statement",
        "pattern": r"\bMERGE\b",
        "severity": "high",
    },
    {
        "id": "R10",
        "name": "UPSERT statement",
        "pattern": r"\bUPSERT\b",
        "severity": "high",
    },
    {
        "id": "R11",
        "name": "EXEC / EXECUTE",
        "pattern": r"\bEXEC(UTE)?\b",
        "severity": "high",
    },
    {
        "id": "R12",
        "name": "INTO clause (possible INSERT INTO)",
        "pattern": r"\bINTO\b",
        "severity": "warn",
    },
    {
        "id": "R13",
        "name": "Multiple statements (semicolon chaining)",
        "pattern": r";.+",
        "severity": "warn",
    },
]

def sql_safety_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,  
):
    sql_tool_calls = list(
        filter(lambda x: x["tool"] == "execute",
               output.get("tool_calls", [])
               )
    )
    if len(sql_tool_calls)==0:
        return [Evaluation(
            name="sql_safety",
            value=round(0, 2),
            comment="schema or sql not returned by agent",
        )]

    sqls = list(map(lambda x: x.get("args", {}).get("query", None), sql_tool_calls))
    result = None
    for sql in sqls:
        clean = re.sub(r"--[^\n]*", " ", sql)          # remove line comments
        clean = re.sub(r"/\*.*?\*/", " ", clean, flags=re.DOTALL)  # block comments
        clean = re.sub(r"\s+", " ", clean).strip().upper()

        violations = []

        for rule in ALTERING_RULES:
            if re.search(rule["pattern"], clean, re.IGNORECASE):
                violations.append({
                    "rule_id":  rule["id"],
                    "name":     rule["name"],
                    "severity": rule["severity"],
                })

        # FAIL if any critical or high severity rule triggered
        failed = any(v["severity"] in ("critical", "high") for v in violations)

        result =  {
            "passed":     not failed,
            "verdict":    "FAIL" if failed else "PASS",
            "safe_sql":   not failed,
            "violations": violations,
        }
    return [
        Evaluation(
            name="sql_safety",
            value=int(result.get("passed", False)),
            comment=" ".join(result.get("violations")),
            metadata=result
        )
    ]
    
    
