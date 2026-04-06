"""Run-level aggregate graders — stub.

Can be used to compute cross-test-case aggregate metrics, such as mean
accuracy or overall recall across the full evaluation dataset.

See ``CONTRIBUTING_EVALUATION.md`` for guidance on adding graders.
"""

from aml_agent.evaluation.types import Evaluation
import re
from typing import Any


EXPECTED_TOOLS = [
    "search_knowledgebase",
    "execute",
    "web_search",
    "get_schema_info"
]

def tool_completeness_grader(
    input: dict,  # noqa: A002
    output: dict,
    expected_output: dict,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
):
    "Tool invocation accuracy"

    tools_called = output.get("tool_calls", [])
    correct_tool_call = 0
    correct_tool_used = 0
    uncalled_tool=list()

    for tool_name in EXPECTED_TOOLS:
        cur_tool = list(filter(lambda x: x['tool'] == tool_name, tools_called))
        if len(cur_tool) > 0:
            correct_tool_used += 1
        else:
            uncalled_tool.append(tool_name)
            continue

        for tool in cur_tool:
            TC_FLAG = True
            match tool.get("tool", ""):
                case "get_schema_info":
                    valid_arg = True
                    valid_response = re.search(r"table", tool.get("response", ""), re.IGNORECASE) is not None
                case "execute":
                    valid_arg = "query" in tool.get("args", {})
                    valid_response = tool.get("response", "") != ""
                case "web_search":
                    valid_arg = "query" in tool.get("args", {})
                    valid_response = tool.get("response", "") != ""
                case "search_knowledgebase":
                    valid_arg = "keyword" in tool.get("args", {})
                    valid_response = re.search(r"Knowledge Base Search Results", tool.get("response", ""), re.IGNORECASE) is not None
                case _:
                    "unexpected or deprecated tool call"       
                    valid_arg = False
                    valid_response = False    
            TC_FLAG =  TC_FLAG and valid_arg and valid_response    
            correct_tool_call += 1 if TC_FLAG else 0 


    total_tool_call = len(tools_called)
    hallucinated_call = list(filter(lambda x: x.get("tool", "") not in EXPECTED_TOOLS, tools_called))
    
    tool_success_rate = correct_tool_call / total_tool_call if total_tool_call>0 else 0
    tool_hallucination_rate = len(hallucinated_call) / total_tool_call if total_tool_call>0 else 0
    tool_correctness_rate = correct_tool_used / len(EXPECTED_TOOLS)

    
    return [
        Evaluation(
            name="tool_success_rate",
                value=round(tool_success_rate, 2),
                comment="Successful tool calls",
                metadata={
                    "success_tool_call": correct_tool_call,
                    "total_tool_call": total_tool_call,
                },
        ),
        Evaluation(
            name="tool_non_hallucination_rate",
            value=round(1.0 - tool_hallucination_rate, 2),
            comment=f"tools hallucinated: {hallucinated_call}",
            metadata = {
                "hallucinated_tool_call": hallucinated_call,
                "total_tool_call": total_tool_call
            }
        ),
        Evaluation(
            name="tool_correctness_rate",
            value=round(tool_correctness_rate, 2),
            comment=f"Uncalled tools: {uncalled_tool}",
            metadata={
                "expected_tool_call":correct_tool_used,
                "total_tool_call":total_tool_call
            }
        )
        ]

    

    



