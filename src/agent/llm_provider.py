"""
llm_provider.py -- pluggable LLM backend behind a vendor-neutral interface.

Design goal: NO provider is privileged. There is one neutral tool
specification (ToolSpec) that WE define, and each provider adapter converts
that neutral spec into its own vendor format in complete isolation. Adding a
new model = adding one self-contained adapter (a ToolSpec->request converter
+ a response->neutral-result extractor) and registering it. Nothing else in
the app, and no other adapter, needs to change.

    ToolSpec (our neutral format)
       |
       +--> AnthropicAdapter : ToolSpec -> Anthropic request -> neutral result
       +--> GeminiAdapter    : ToolSpec -> Gemini request    -> neutral result
       +--> (future adapters, each isolated)

The provider is chosen by the LLM_PROVIDER env var:
    LLM_PROVIDER=anthropic   (default)
    LLM_PROVIDER=gemini

Each provider needs its own key in .env:
    ANTHROPIC_API_KEY=...   (for anthropic)
    GEMINI_API_KEY=...      (for gemini)

Public entry point: call_tool(system_prompt, tool: ToolSpec, user_prompt)
returns (tool_input: dict, usage: dict). tool_input is the arguments the
model chose (same dict shape regardless of provider). usage has token counts
+ cost_usd.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


# --- The vendor-neutral tool specification --------------------------------

@dataclass
class ToolSpec:
    """A single callable tool, described in OUR OWN terms -- not any vendor's.

    - name:        the tool's identifier.
    - description: what the tool does / how the model should use it.
    - parameters:  a plain JSON-Schema object describing the arguments. JSON
                   Schema is an open standard (not a vendor format), and every
                   provider we target accepts a JSON-Schema parameter object,
                   so this stays as a neutral dict rather than being modelled
                   field-by-field. Adapters pass it through to whatever key
                   their vendor expects.
    """
    name: str
    description: str
    parameters: dict


def _neutral_usage(input_tokens: int, output_tokens: int, cost_usd: float,
                   cache_write_tokens: int = 0, cache_read_tokens: int = 0) -> dict:
    """The single usage shape every adapter returns, so callers never see
    vendor-specific token field names."""
    return {
        "input_tokens": input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }


# --- Adapter interface ----------------------------------------------------

class LLMAdapter(ABC):
    """One isolated provider. An adapter's ONLY responsibilities are:
    convert a neutral ToolSpec into its vendor's request, force the tool
    call, call the vendor, and convert the vendor's response back into the
    neutral (tool_input dict, usage dict) result. It knows nothing about any
    other provider."""

    MODEL: str = ""

    def model_name(self) -> str:
        """The model to call. Resolved at CALL TIME (not class-definition
        time) so adapters that read a model from an env var see it after
        .env has been loaded. Base returns the class attribute; adapters
        whose model is configurable override this."""
        return self.MODEL

    @abstractmethod
    def call_tool(self, system_prompt: str, tool: ToolSpec, user_prompt: str) -> tuple[dict, dict]:
        ...


# --- Anthropic adapter (isolated) -----------------------------------------

class AnthropicAdapter(LLMAdapter):
    MODEL = "claude-haiku-4-5-20251001"
    # $/MTok: fresh input, cache write, cache read, output
    PRICES = (1.00, 1.25, 0.10, 5.00)

    def _to_request_tool(self, tool: ToolSpec) -> dict:
        """Neutral ToolSpec -> Anthropic tool dict. Note this adapter, like
        every other, converts EXPLICITLY -- Anthropic is not privileged just
        because it happens to nest parameters under 'input_schema'."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def call_tool(self, system_prompt: str, tool: ToolSpec, user_prompt: str) -> tuple[dict, dict]:
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")
        client = Anthropic(api_key=api_key)

        # Prompt caching: tool schema (with its ~150-tag vocabulary) and system
        # prompt are large and static, so mark them cacheable -- repeat calls
        # within the 5-minute window mostly pay cache-read rates.
        request_tool = {**self._to_request_tool(tool), "cache_control": {"type": "ephemeral"}}
        system_blocks = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

        response = client.messages.create(
            model=self.model_name(),
            max_tokens=600,
            system=system_blocks,
            tools=[request_tool],
            tool_choice={"type": "tool", "name": tool.name},
            messages=[{"role": "user", "content": user_prompt}],
        )

        tool_use = next(b for b in response.content if b.type == "tool_use")
        tool_input = tool_use.input

        fresh = response.usage.input_tokens
        cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        out = response.usage.output_tokens
        p_fresh, p_cw, p_cr, p_out = self.PRICES
        cost = (fresh * p_fresh + cache_write * p_cw + cache_read * p_cr + out * p_out) / 1_000_000

        return tool_input, _neutral_usage(fresh, out, cost, cache_write, cache_read)


# --- Gemini adapter (isolated) --------------------------------------------

class GeminiAdapter(LLMAdapter):
    MODEL = "gemini-2.5-flash"
    # $/MTok: input, output. Free tier = 0; set real prices if you go paid.
    PRICES = (0.0, 0.0)

    def _to_request_tool(self, tool: ToolSpec) -> dict:
        """Neutral ToolSpec -> Gemini function_declarations. The JSON-Schema
        parameters pass straight through to Gemini's 'parameters' key."""
        return {
            "function_declarations": [{
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }]
        }

    def call_tool(self, system_prompt: str, tool: ToolSpec, user_prompt: str) -> tuple[dict, dict]:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in .env")
        client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[self._to_request_tool(tool)],
            # Force a function call (Gemini's equivalent of tool_choice).
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
            temperature=0,
        )

        response = client.models.generate_content(
            model=self.model_name(),
            contents=user_prompt,
            config=config,
        )

        function_call = None
        for part in response.candidates[0].content.parts:
            if getattr(part, "function_call", None):
                function_call = part.function_call
                break
        if function_call is None:
            raise RuntimeError("Gemini did not return a function call as required (mode=ANY).")

        # Gemini's args is already a structured object -- no json.loads needed.
        tool_input = dict(function_call.args)

        um = getattr(response, "usage_metadata", None)
        in_tokens = getattr(um, "prompt_token_count", 0) or 0
        out_tokens = getattr(um, "candidates_token_count", 0) or 0
        p_in, p_out = self.PRICES
        cost = (in_tokens * p_in + out_tokens * p_out) / 1_000_000

        return tool_input, _neutral_usage(in_tokens, out_tokens, cost)


# --- OpenAI-compatible adapter (Groq, Cerebras, Mistral, OpenRouter, ...) --

class OpenAICompatibleAdapter(LLMAdapter):
    """Adapter for any provider that speaks the OpenAI Chat Completions API.
    A large family of free/cheap providers (Groq, Cerebras, Mistral,
    OpenRouter, Together, ...) are all OpenAI-compatible -- same request
    shape, differing only by base_url + api_key + model. So this one adapter
    covers all of them; concrete providers below just set those three.

    Subclasses set BASE_URL, API_KEY_ENV, MODEL (and optionally PRICES)."""
    BASE_URL: str = ""
    API_KEY_ENV: str = ""
    MODEL: str = ""
    PRICES = (0.0, 0.0)  # (input, output) $/MTok -- free tiers = 0

    def _to_request_tool(self, tool: ToolSpec) -> dict:
        """Neutral ToolSpec -> OpenAI function-tool shape."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def call_tool(self, system_prompt: str, tool: ToolSpec, user_prompt: str) -> tuple[dict, dict]:
        from openai import OpenAI

        api_key = os.getenv(self.API_KEY_ENV)
        if not api_key:
            raise EnvironmentError(f"{self.API_KEY_ENV} not found in .env")
        client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

        response = client.chat.completions.create(
            model=self.model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[self._to_request_tool(tool)],
            # Force the specific tool (OpenAI's named-tool form -- the
            # equivalent of Anthropic tool_choice / Gemini mode=ANY).
            tool_choice={"type": "function", "function": {"name": tool.name}},
            temperature=0,
        )

        message = response.choices[0].message
        if not message.tool_calls:
            raise RuntimeError(
                f"{self.__class__.__name__} did not return a tool call as required."
            )
        import json as _json
        # OpenAI-compatible providers return arguments as a JSON STRING here
        # (unlike Anthropic/Gemini which give a dict), so parse it.
        tool_input = _json.loads(message.tool_calls[0].function.arguments)

        usage_obj = getattr(response, "usage", None)
        in_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
        out_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
        p_in, p_out = self.PRICES
        cost = (in_tokens * p_in + out_tokens * p_out) / 1_000_000

        return tool_input, _neutral_usage(in_tokens, out_tokens, cost)


class GroqAdapter(OpenAICompatibleAdapter):
    BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY_ENV = "GROQ_API_KEY"
    PRICES = (0.0, 0.0)  # free tier

    def model_name(self) -> str:
        # REQUIRE GROQ_MODEL to be set explicitly -- no silent default.
        # Groq churns its lineup, so a hardcoded default tends to 404, and a
        # silent fallback masks the real problem ("I set GROQ_MODEL but it's
        # ignored") behind a confusing model-not-found error. Failing loudly
        # here surfaces the actual issue. Run list_groq_models() to see what
        # your account has, then set GROQ_MODEL in .env to one of them.
        model = os.getenv("GROQ_MODEL")
        if not model:
            raise EnvironmentError(
                "GROQ_MODEL is not set. Add it to your .env, e.g.\n"
                "    GROQ_MODEL=llama-3.3-70b-versatile\n"
                "Run `python src/agent/llm_provider.py` to list the models your "
                "Groq API key can actually access, then use one of those."
            )
        return model


# --- Registry + public entry point ----------------------------------------

# To add a new model later: write an isolated Adapter class above and add one
# line here. Nothing else changes. (OpenAI-compatible providers are even
# easier -- subclass OpenAICompatibleAdapter with a base_url/key/model.)
_ADAPTERS: dict[str, type[LLMAdapter]] = {
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "groq": GroqAdapter,
}


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


def call_tool(system_prompt: str, tool: ToolSpec, user_prompt: str) -> tuple[dict, dict]:
    """Force the configured LLM to call the given tool; return (tool_input,
    usage). `tool` is a vendor-neutral ToolSpec."""
    provider = get_provider()
    adapter_cls = _ADAPTERS.get(provider)
    if adapter_cls is None:
        known = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Set it to one of: {known}."
        )
    return adapter_cls().call_tool(system_prompt, tool, user_prompt)


# --- Utilities ------------------------------------------------------------

def list_groq_models() -> list[str]:
    """Return the model IDs your Groq API key can actually access.

    Groq changes its lineup often, so a hardcoded model name can 404. Run
    this to discover valid names, then set GROQ_MODEL in .env to one of them:

        python src/agent/llm_provider.py

    (Requires GROQ_API_KEY in the environment / .env.)
    """
    import os
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found in .env")
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    resp.raise_for_status()
    return sorted(m["id"] for m in resp.json().get("data", []))


if __name__ == "__main__":
    # Convenience: list the Groq models available to your key.
    print("Groq models available to your API key:\n")
    for model_id in list_groq_models():
        print(f"  {model_id}")
    print("\nSet GROQ_MODEL in your .env to one of these (ideally a tool-use\n"
          "capable model like a llama-3.x or qwen variant).")