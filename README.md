# `llmscope`

[![PyPI](https://img.shields.io/pypi/v/llmscope.svg)](https://pypi.org/project/llmscope/)
[![Documentation](https://readthedocs.org/projects/llm/badge/?version=latest)](https://yuantianding.github.io/llmscope/)

`llmscope` is a Python library designed to simplify interactions with Large Language Models (LLMs) by providing a stateful, fluent interface for managing conversation history, tool usage, and response parsing. It leverages libraries like `mirascope` for LLM calls and `pydantic` for data validation and parsing.

## A Taste of `llmscope`

```python
from pydantic import BaseModel
import llmscope

class CodeItemDoc(BaseModel):
    summary: str = Field("A concise description of this function/struct/enum... ")
    example: str = Field("Provide an example of how this item is used. ")

@llmscope.fn("openai", model="gpt-4o")
def generate_doc(llm, code: str, item_type: Literal["function", "struct"]) -> CodeItemDoc:
    # Organize your prompts like `print`
    llm.system("You are a professional software engineer writing documentation. ")
    if item_type == "function":
        llm.system("* For functions, start with a verb and describe the functionality.")
    else:
        llm.system("* For structs, start with a noun phrase summarizing this type.")
    
    llm.user(code)

    # Elegant structured output selectors.
    # Using `os.fork` to collect different json schemas in different places.
    # Equilvalent to mirascope.llm.call(tools=[ViewCodeSpace], response_model=CodeItemDoc).
    while tool := llm.try_tool(ViewCodeSpace):
        llm.assistant(tool.call())

    return llm.parse(CodeItemDoc) 
```


## Installation

```bash
pip install llmscope[openai,anthropic,...] 
```
*(Note: Similar to Mirascope, `llmscope` also relies upon different dependencies for different LLM providers, see full list of providers: [https://mirascope.com/api/llm/call/](https://mirascope.com/api/llm/call/))*

## Usage

Use the `@fn` decorator to wrap a function that defines the agent's behavior. This decorator automatically injects an `LLMState` instance.

```python
from llmscope import fn, BaseTool, Field
# Assuming necessary imports for Provider, BaseMessageParam, etc.

# Define a tool (using mirascope's BaseTool)
class EmotionTool(BaseTool):
    """Tool to represent a chosen emotion."""
    emotion: str = Field(..., description="The name of the emotion.")
    reason: str = Field(..., description="A brief reason for choosing this emotion.")

    def call(self):
        print(f"Tool Call: Emotion={self.emotion}, Reason={self.reason}")
        return f"Emotion {self.emotion} acknowledged."

# Define the agent function
@fn(provider="openai", model="gpt-4o") # Configure provider and model
def emotion_agent(llm, initial_prompt: str):
    llm.system("You are an llm that chooses emotions when asked.")
    llm.user(initial_prompt)

    # Loop while the LLM decides to use the EmotionTool
    while tool_call := llm.try_tool(EmotionTool):
        result = tool_call.call() # Execute the tool
        print(f"Tool Result: {result}")
        # Add tool execution result back to the conversation
        llm.assistant(f"Okay, I chose {tool_call.emotion}.") # Or use llm.tool(tool_call=..., content=result) with mirascope
        llm.user("Okay, choose another different emotion and explain why.")

    # If no tool is called, get the final text response
    try:
        final_response = llm.generate()
        print(f"Agent's final text response: {final_response.content}")
    except Exception as e:
        # Handle cases where generate might fail or is used incorrectly (e.g., after try_parse)
        print(f"Could not generate final response: {e}")

    return "Agent finished."

# Run the llm
result = emotion_agent("Choose an emotion and explain why.")
print(result)
```

### 2. Key `LLMState` Methods

*   `config(provider=..., model=..., call_params=...)`: Sets the LLM provider, model, and optional call parameters.
*   `msg(role, *message)`: Adds a message to the history.
*   `system(*message)`, `user(*message)`, `assistant(*message)`: Convenience methods for `msg`.
*   `try_tool(ToolClass)`: Attempts to get the LLM to use the specified tool. Returns the tool instance if successful, `None` otherwise. Can be used in a loop.
*   `try_tools(*ToolClasses)`: Similar to `try_tool` but for multiple possible tools. Returns a list of successful tool calls.
*   `try_parse(PydanticModel)`: Attempts to parse the LLM response into the given Pydantic model *without* finalizing the request. Useful for checking intermediate structured outputs.
*   `parse(PydanticModel)`: **Finalizes** the request and parses the LLM response into the given Pydantic model. Raises `ValidationError` on failure.
*   `generate()`: **Finalizes** the request and returns the raw LLM response (`BaseCallResponse` from mirascope), typically used when no specific parsing or tool use is expected at the end.

**Important:** Methods like `try_tool`, `try_parse`, `parse`, and `generate` trigger internal state management and potentially LLM calls. Avoid calling `config` or `msg` *between* a `try_` call and its corresponding `parse` or `generate` call within the same logical block.



## License

[MIT](https://mit-license.org/)
