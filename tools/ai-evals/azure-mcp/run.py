import os
import asyncio
import pathlib
import json
from typing import Any

# set before azure.ai.evaluation import to make PF output less noisy
os.environ["PF_LOGGING_LEVEL"] = "CRITICAL"

import dotenv
from openai import AzureOpenAI
from openai.types import chat
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from mcp.types import CallToolResult
from azure.ai.evaluation import evaluate, ToolCallAccuracyEvaluator, AzureAIProject
from azure.identity import (
    AzurePipelinesCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)


dotenv.load_dotenv()

# for best results, model judge should always be a different model from the one we are evaluating
MODEL = "gpt-4.1"
MODEL_JUDGE = "o3-mini"
API_VERSION = "2025-03-01-preview"
SCORE_THRESHOLD = 0.8


def in_ci() -> bool:
    return os.getenv("TF_BUILD") is not None


if in_ci():
    service_connection_id = os.environ["AZURESUBSCRIPTION_SERVICE_CONNECTION_ID"]
    client_id = os.environ["AZURESUBSCRIPTION_CLIENT_ID"]
    tenant_id = os.environ["AZURESUBSCRIPTION_TENANT_ID"]
    system_access_token = os.environ["SYSTEM_ACCESSTOKEN"]
    CREDENTIAL = AzurePipelinesCredential(
        service_connection_id=service_connection_id,
        client_id=client_id,
        tenant_id=tenant_id,
        system_access_token=system_access_token,
    )
else:
    CREDENTIAL = DefaultAzureCredential()


# Monkeypatch AsyncPrompty.load to accept token_credential
# https://github.com/Azure/azure-sdk-for-python/issues/41295
from azure.ai.evaluation._legacy.prompty import AsyncPrompty

original_load = AsyncPrompty.load


def patched_load(cls, source, **kwargs):
    """Patched version of AsyncPrompty.load that accepts token_credential parameter"""
    return original_load(source=source, token_credential=CREDENTIAL, **kwargs)


AsyncPrompty.load = classmethod(patched_load)


client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=API_VERSION,
    azure_ad_token_provider=get_bearer_token_provider(CREDENTIAL, "https://cognitiveservices.azure.com/.default"),
)

server_params = StdioServerParameters(
    command="npx", args=["-y", "@azure/mcp@latest", "server", "start"], env=None
)


def reshape_tools(
    tools: list[chat.ChatCompletionMessageToolCall],
) -> list[dict[str, Any]]:
    return [
        {
            "type": "tool_call",
            "tool_call_id": tool.id,
            "name": tool.function.name,
            "arguments": json.loads(tool.function.arguments),
        }
        for tool in tools
    ]


def reshape_tool_definitions(
    available_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools = [tool["function"] for tool in available_tools]
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
            "type": "function",
        }
        for tool in tools
    ]


async def call_mcp_tool(tool_call: chat.ChatCompletionMessageToolCall, function_args: dict[str, Any]) -> CallToolResult:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.list_tools()

            result = await session.call_tool(tool_call.function.name, function_args)

    return result


async def make_request(
    messages: list,
    available_tools: list[dict[str, Any]],
    tool_calls_made: list[chat.ChatCompletionMessageToolCall],
) -> tuple[list, list[chat.ChatCompletionMessageToolCall]]:
    
    # Execute all tool calls
    tool_messages = []
    for tool_call in tool_calls_made:
        function_args = json.loads(tool_call.function.arguments)
        result = await call_mcp_tool(tool_call, function_args)

        # Handle different content types
        content = ""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            # Try to get text content, fallback to string representation
            try:
                content = getattr(content_item, 'text', str(content_item))
            except:
                content = str(content_item)

        tool_messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": content,
            }
        )
    
    # Add tool results to messages
    messages.extend(tool_messages)
    
    return messages, tool_calls_made


def is_final_answer(query: str, content: str) -> bool:
    """Use LLM to determine if the response is a final answer or asking for more information."""
    judge_prompt = f"""
    Determine if the following response is a final answer to a user's question or if it's asking for more information.

    User's question: "{query}"
    Response: "{content}"

    Reply with only "FINAL" if this is a final answer that addresses the user's request, or "QUESTION" if it's asking for SPECIFIC information necessary to answer the question.
    """
    
    judge_response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=10
    )
    
    response_content = judge_response.choices[0].message.content
    return "FINAL" in (response_content.upper() if response_content else "")


def evaluate_azure_mcp(query: str, expected_tool_calls: list):
    messages = [{"role": "user", "content": query}]
    available_tools = asyncio.run(get_tools())

    followup_answers = pathlib.Path(__file__).parent / "followup.json"
    with open(str(followup_answers), "r") as f:
        options = json.load(f)

    attempts = 0
    max_attempts = 10
    all_tool_calls_made = []

    while attempts < max_attempts:
        attempts += 1
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=available_tools)  # type: ignore

        response_message = response.choices[0].message
        messages.append(response_message)  # type: ignore

        # If tool calls were made, execute them
        if response_message.tool_calls:
            all_tool_calls_made.extend(response_message.tool_calls)
            messages, _ = asyncio.run(make_request(messages, available_tools, response_message.tool_calls))  # type: ignore
            
            # After tool execution, continue the conversation to see if assistant has follow-up questions
            continue

        # If no tool calls and the assistant has content (might be follow-up question or final answer)
        if response_message.content and not response_message.tool_calls:
            # Use LLM judge to determine if this is a final answer
            if is_final_answer(query, response_message.content):
                # This is a final answer, break the loop
                break
            else:
                followup_question = response_message.content
                user_prompt = (
                    f"You are the user. Given the following options: {json.dumps(options)}, "
                    f"answer the assistant's question as concisely as possible: {followup_question}."
                )
                followup_response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful user."},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                messages.append({"role": "user", "content": followup_response.choices[0].message.content})  # type: ignore
                continue

        # If we get here, the assistant has finished (no tool calls, no questions)
        break

    tool_calls_made = reshape_tools(all_tool_calls_made)
    tool_defs = reshape_tool_definitions(available_tools)

    return {
        "tool_calls": tool_calls_made,
        "response": [msg.model_dump() if not isinstance(msg, dict) else msg for msg in messages],
        "tool_definitions": tool_defs,
        "num_tool_calls_actual": len(tool_calls_made),
        "num_tool_calls_expected": len(expected_tool_calls),
    }


async def get_tools() -> list[dict[str, Any]]:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tools.tools
            ]
    return available_tools


class MCPEval:

    def __init__(self): ...

    def get_failure_reasons(
        self, tool_calls, correct_params, called_expected, num_tool_calls_actual, num_tool_calls_expected
    ):
        reasons = []

        if len(tool_calls) == 0:
            return "No tool calls were made"

        if not called_expected:
            reasons.append("Expected tool was not called")

        if not correct_params:
            reasons.append("Some parameters are missing or invalid")

        if num_tool_calls_actual != num_tool_calls_expected:
            reasons.append(
                f"Number of tool calls mismatch (expected {num_tool_calls_expected}, got {num_tool_calls_actual})"
            )

        return "; ".join(reasons) if reasons else "Passed successfully"

    def __call__(
        self, tool_calls, tool_definitions, expected_tool_calls, num_tool_calls_actual, num_tool_calls_expected
    ):

        correct_params = False
        for tool_call in tool_calls:
            arguments = tool_call.get("arguments", {})
            tool_def = [d for d in tool_definitions if d["name"] == tool_call["name"]]
            if not tool_def:
                break

            tool_def = tool_def[0]
            parameters = tool_def.get("parameters", {})
            required = parameters.get("required", [])
            properties = parameters.get("properties", {})

            all_required_present = all(arg in arguments and arguments[arg] is not None for arg in required)
            # TODO tool definitions changed - part of first MCP call now
            # all_args_valid = all(arg in required or arg in properties for arg in arguments)
            # correct_params = all_required_present and all_args_valid
            correct_params = all_required_present

        called_expected = (
            any(actual["name"] == expected for actual in tool_calls for expected in expected_tool_calls)
        )
        reason = self.get_failure_reasons(
            tool_calls, correct_params, called_expected, num_tool_calls_actual, num_tool_calls_expected
        )
        score = (
            (0.5 if called_expected else 0.0)
            + (0.3 if correct_params else 0.0)
            + (0.2 if num_tool_calls_actual == num_tool_calls_expected else 0.0)
        )
        return {
            "tool_call_accuracy": "Pass" if score >= SCORE_THRESHOLD else "Fail",
            "reason": reason,
            "score": score,
            "score_threshold": SCORE_THRESHOLD,
        }


if __name__ == "__main__":
    azure_ai_project: AzureAIProject = {
        "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
        "resource_group_name": os.environ["AZURE_FOUNDRY_RESOURCE_GROUP"],
        "project_name": os.environ["AZURE_FOUNDRY_PROJECT_NAME"],
    }

    model_config: dict[str, str] = {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "azure_deployment": MODEL_JUDGE,
        "api_version": API_VERSION,
    }

    tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config, is_reasoning_model=True)
    custom = MCPEval()

    test_file = pathlib.Path(__file__).parent / "data.jsonl"
    result = evaluate(
        data=str(test_file),
        evaluators={
            "mcp": custom,
        },
        target=evaluate_azure_mcp,
        azure_ai_project=azure_ai_project,
        credential=CREDENTIAL,
    )
    print(f"Overall score: {result['metrics']['mcp.score']}")
    if 'studio_url' in result:
        print(f"Evaluation result: {result['studio_url']}")
    else:
        print("Evaluation completed (no studio URL available)")
