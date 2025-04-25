from dataclasses import dataclass, field

import os
import textwrap
from typing import Any, Union
from pydantic import BaseModel, Field, ValidationError
from mirascope import BaseMessageParam, BaseTool, Provider, llm
from mirascope.core.base import BaseCallResponse
from pydantic import BaseModel, TypeAdapter
import pickle
from pydantic import Field

@dataclass
class LLMRequest(BaseException):
    """Represents a request to the language model."""
    provider: Provider
    model: str
    messages: list[BaseMessageParam]
    tools: list[BaseTool]
    call_params: dict | None = None
    response_model: BaseModel | None = None

    def generate(self):
        """Generates a response from the language model based on the request."""
        kwargs = {}
        if self.call_params is not None:
            kwargs["call_params"] = self.call_params
        if self.response_model is not None:
            kwargs["response_model"] = self.response_model
        return llm.call(provider=self.provider, model=self.model, tools=self.tools, **kwargs)(lambda: self.messages)()

class LLMState:
    """
    Manages the state and configuration for an llm's interaction with a language model.
    This class provides a fluent interface for building LLM requests, including setting
    the provider and model, adding messages, defining tools, and specifying response
    parsing models. It uses an internal mechanism involving subprocesses (`_command`)
    to collect tool and response schema definitions across chained calls (`try_tools`,
    `try_parse`) before finally executing the LLM request (`generate`, `parse`).
    Attributes:
        provider (Provider | None): The LLM provider instance (e.g., OpenAI, Anthropic).
        model (str | None): The specific model name to use (e.g., "gpt-4o").
        messages (list[BaseMessageParam]): A list of messages constituting the
            conversation history.
    """
    provider: Provider | None = None
    model: str | None = None
    messages: list[BaseMessageParam] = []
    call_params: dict | None = None
    _tools: list[BaseTool] = []
    _response_model: list[type] = []
    _response_value: Any = None
    _child_writeback: Any = None

    def __init__(self):
        pass
    
    def config(self, provider: Provider | None = None, model: str | None = None, call_params: dict | None = None):
        """Configure the llm's provider, model, and call parameters.

        This method updates the llm's configuration. It includes an assertion
        to prevent configuration changes between specific method calls like
        `try_tools`, `try_tool`, `try_parse`, `parse`, and `generate`.

        Args:
            provider (Provider | None, optional): The API provider to set for the llm.
                If None, the current provider remains unchanged. Defaults to None.
            model (str | None, optional): The model name to set for the llm.
                If None, the current model remains unchanged. Defaults to None.
            call_params (dict | None, optional): The call parameters to set for the llm's
                API calls. If None, the current call parameters remain unchanged.
                Defaults to None.

        Returns:
            self: The llm instance itself, allowing for method chaining.

        Raises:
            AssertionError: If called between `try_tools`, `try_tool`, `try_parse`,
                `parse` and `generate` methods.
        """
        assert self._child_writeback is None and self._response_value is None, "Called `llm.config` between `try_tools`, `try_tool`, `try_parse`, `parse` and `generate`."
        if provider is not None:
            self.provider = provider
        if model is not None:
            self.model = model
        if call_params is not None:
            self.call_params = call_params
        return self

    def msg(self, role: str, *message: list[Any]):
        """Append a message to the llm's memory.

        This method takes a role and one or more message strings, processes
        them, and adds them to the llm's message history (`self.messages`).
        If the last message in the history has the same role, the new
        content is appended to it. Otherwise, a new message object is created.
        It also use `textwrap.dedent` to remove common leading whitespace from the
        message content.

        Args:
            role (str): The role of the message sender (e.g., "user",
                "assistant", "system").
            *message (list[Any]): One or more message parts (typically strings)
                to be combined into a single message content. Each part is
                dedented before joining.

        Returns:
            LLMState: The llm instance itself, allowing for method chaining.

        Raises:
            AssertionError: If called between specific asynchronous operations
                like `try_tools`, `try_tool`, `try_parse`, `parse`, or
                `generate`, indicating improper usage.
        """
        message = " ".join(textwrap.dedent(m) for m in message) + "\n"
        assert self._child_writeback is None and self._response_value is None, "Called `llm.msg` between `try_tools`, `try_tool`, `try_parse`, `parse` and `generate`."
        if len(self.messages) > 0 and self.messages[-1].role == role:
            self.messages[-1].content += " ".join(message) + "\n"
        else:
            self.messages.append(BaseMessageParam(role=role, content=" ".join(message) + "\n"))
        return self
    
    def system(self, *message: list[Any]):
        """Send a system message. See `msg` for more details.

        Args:
            *message: A list of messages to be sent as system messages.

        Returns:
            LLMState: The llm instance itself, allowing for method chaining.
        """
        return self.msg("system", *message)
        
    def user(self, *message: list[Any]):
        """Send a user message. See `msg` for more details.

        Args:
            *message: A list of messages to be sent as system messages.

        Returns:
            LLMState: The llm instance itself, allowing for method chaining.
        """
        return self.msg("user", *message)
    
    def assistant(self, *message: list[Any]):
        """Send a assistant message. See `msg` for more details.

        Args:
            *message: A list of messages to be sent as system messages.

        Returns:
            LLMState: The llm instance itself, allowing for method chaining.
        """
        return self.msg("assistant", *message)
    
    def _command(self, update_state: callable, validate: callable, final: bool = False) -> Any | None:
        # If it is subprcess, update current state and return to the main process later
        if self._child_writeback is not None:
            update_state()
            if final:
                with os.fdopen(self._child_writeback, "wb") as write_pipe:
                    pickled_types = pickle.dumps((self._tools, self._response_model))
                    write_pipe.write(pickled_types)
                os._exit(0)
            else:
                return None
        # If it is main process, we need to check if the response value is already exists and validate it.
        if self._response_value is not None:
            resp = validate(self._response_value)
            if resp is not None:
                self._response_value = None
                self._tools = []
                self._response_model = []
                return resp
            else:
                if final:
                    raise ValidationError(f"Invalid response value: {type(self._response_value)}. Expected: {self._response_model + self._tools} ")
                return None
        # Otherwise 1. Create a subprocess to collect the _tools and response schema, 2. Generate a value and check.
        else:
            if final:
                assert self._tools == [], "Unknown Error"
                assert self._response_model == [], "Unknown Error"
                update_state()
            else:
                r, w = os.pipe()
                pid = os.fork()
                if pid == 0:
                    os.close(r)
                    self._child_writeback = w
                    return self._command(update_state, validate, final)
                os.close(w)
                os.waitpid(pid, 0)
                with os.fdopen(r, "rb") as read_pipe:
                    pickled_types = read_pipe.read()
                (self._tools, self._response_model) = pickle.loads(pickled_types)
            assert all(issubclass(m, BaseTool) for m in self._tools)
            assert all(issubclass(m, BaseModel) for m in self._response_model)
            self._work_value()
            return self._command(update_state, validate, final)
    
    def try_tools(self, *tools: list[type]) -> list[BaseTool] | None:
        """Attempts to let the llm use a specific set of tools.

        This method uses `os.fork` to create a subprocess that collects the
        tools and response schema and assemble all these type information into a
        request to the language model. 
        Args:
            *tools (list[type]): A variable number of tool types (classes) to
                try. Each type must be a subclass of `BaseTool`.
        Returns:
            list[BaseTool] | None: A list containing the validated tool call
                objects from the response if the validation is successful for
                at least one of the provided tools. Returns `None` if the
                response does not contain a valid call to any of the specified
                tools.
        """
        assert all(issubclass(t, BaseTool) for t in tools), "All tools must be subclass of `BaseTool`."
        def validate(value: BaseCallResponse):
            if not hasattr(value, 'tools') or type(value.tools) is not list:
                return None
            
            v = [v for v in value.tools if any(_validate(t, v.model_dump()) and v.tool_call.function.name == t.__name__ for t in tools)]
            if len(v) == 0:
                return None
            return v
        return self._command(
            update_state=lambda: self._tools.extend(tools),
            validate=validate,
        )
    
    def try_tool(self, tool: type) -> BaseTool | None:
        """Attempts to let the llm use a specific tool.
        This method is a convenience wrapper around `try_tools` for a single
        tool type. It calls `try_tools` with the provided tool type and
        returns the first successfully initialized tool instance if any,
        otherwise None. See `try_tools` for more details on the underlying
        mechanism and potential error handling.
        Args:
            tool (type): The class type of the tool to attempt to use.
        Returns:
            BaseTool | None: An instance of the tool if successfully
                initialized and added, otherwise `None`.
        """
        result = self.try_tools(tool)
        if type(result) == list and len(result) > 0:
            return result[0]
        else:
            return None
    

    def try_parse(self, ty: type, final: bool = False) -> Any:
        """Attempts to parse and validate a value against the specified type.

        This method uses `os.fork` to create a subprocess that collects the
        tools and response schema and assemble all these type information into a
        request to the language model. 

        Args:
            ty (type): The expected type for the value. Must be a type
                supported by Pydantic's TypeAdapter for validation.

        Returns:
            Any: The parsed and validated value conforming to the type `ty`.
                 The exact behavior on validation failure depends on the
                 implementation of the `_command` method.

        Raises:
            AssertionError: If the provided type `ty` is not validatable
                by Pydantic.
        """
        assert TypeAdapter(ty), "Type must be validable through pydantic. "
        return self._command(
            update_state=lambda: self._response_model.append(ty),
            validate=lambda value: value if _validate(ty, value) else None,
            final=final
        )

    def parse(self, ty: type) -> Any:
        """Parse the content into the given type.

        Note that when this method is often used in conjunction with `try_parse` and `try_tools`.
        `try_parse` or `try_tools` uses `os.fork` to create a subprocess that collects the
        tools and response schema and assemble all these type information into a
        request to the language model. The forked process will be ended here, returning
        all type information (tools and response schemas) into the main process.

        This method can also be used alone, in which case it will not use `os.fork` to create a subprocess.
        It will directly generate the response and parse it into the given type.

        Args:
            ty (type): The target type to parse the content into.

        Returns:
            Any: An instance of the specified type `ty` representing the
                 parsed content.

        Raises:
            ValueError: If the content cannot be successfully parsed into the
                        specified type `ty`.
        """
        return self.try_parse(ty, final=True)
    
    def generate(self) -> BaseCallResponse:
        """Generate a response from the language model.

        Note that when this method is often used in conjunction with `try_tools` but not `try_parse`.
        `try_tools` uses `os.fork` to create a subprocess that collects the
        tools and response schema and assemble all these type information into a
        request to the language model. The forked process will be ended here, returning
        all type information (tools and response schemas) into the main process.

        This method can also be used alone, in which case it will not use `os.fork` to create a subprocess.
        It will directly generate the response and parse it into the given type.

        The method will return AssertionError when used with `try_parse`.


        Returns:
            BaseCallResponse: The generated response from the language model, from `mirascope`.

        Raises:
            AssertionError: when used with `try_parse`.
        """
        assert len(self._response_model) == 0, "Must not use `generate` together with `try_parse`."
        return self._command(
            update_state=lambda: None,
            validate=lambda value: BaseCallResponse.model_validate(value),
            final=True
        )

    def _work_value(self):
        assert self.provider is not None, "Provider must be set. See https://mirascope.com/api/llm/call/."
        assert self.model is not None, "Model must be set. See https://mirascope.com/api/llm/call/. "
        if len(self._response_model) == 0:
            response_model = None
        elif len(self._response_model) == 1:
            response_model = self._response_model[0]
        else:
            response_model = Union[*self._response_model]

        response = LLMRequest(
            provider=self.provider,
            model=self.model,
            messages=self.messages,
            tools=self._tools,
            call_params=self.call_params,
            response_model=response_model,
        ).generate()
        if response is None:
            raise RuntimeError("No response from LLM.")
        self._response_value = response
        self._tools = []
        self._response_model = []

def _validate(ty: type, value: Any) -> bool:
    try:
        TypeAdapter(ty).validate_python(value)
        return True
    except ValidationError:
        return False

def fn(provider: Provider | None = None, model: str | None = None) -> callable:
    """Create a decorator to initialize and inject LLMState into a function.

    This function acts as a factory that generates a decorator. When this
    decorator is applied to a function, it modifies the function's behavior.
    The decorated function will receive an `LLMState` object as its first
    argument. This `LLMState` object can be used for LLM generation,
    managing messages, and handling tools and response schemas.

    The decorated function is expected to accept `LLMState` as its first
    positional argument, followed by its original arguments (`*args`, `**kwargs`),
    and should return a `runner` object.

    Args:
        provider (Provider | None, optional): The provider instance to assign
            to `state.provider`. Defaults to None.
        model (str | None, optional): The model identifier string to assign
            to `state.model`. Defaults to None.

    Returns:
        callable: A decorator function that wraps the target function,
                  injects the configured `LLMState`, and returns the
                  `runner` produced by the target function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            state = LLMState()
            state.provider = provider
            state.model = model
            runner = func(state, *args, **kwargs)
            return runner
        return wrapper
    return decorator


if __name__ == "__main__":
    class EmotionTool(BaseTool):
        """Tool to represent a chosen emotion."""
        emotion: str = Field(..., description="The name of the emotion.")
        reason: str = Field(..., description="A brief reason for choosing this emotion.")

        def call(self):
            print(f"Tool Call: Emotion={self.emotion}, Reason={self.reason}")
            return f"Emotion {self.emotion} acknowledged."


    @fn(provider="openai", model="gpt-4o")
    def example(llm: LLMState):
        llm.system("You are an llm that chooses emotions when asked.")
        llm.user("Choose an emotion and explain why.")

        while tool := llm.try_tool(EmotionTool):
            result = tool.call()
            print(result)
            llm.assistant(f"Okay, I chose {tool.emotion}.")
            llm.user("Okay, choose another different emotion and explain why.")

        try:
            final_response = llm.generate()
            print(f"Agent's final text response: {final_response}")
        except Exception as e:
            print(f"Could not generate final response: {e}")

        return True

    print(example())