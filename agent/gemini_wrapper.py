"""
Gemini wrapper using direct HTTP REST API calls.
No external SDK needed — only 'requests' (already in requirements.txt).
This avoids all google namespace conflicts caused by firebase-admin.
"""
import json
import requests


GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiAnthropicWrapper:
    """
    Drop-in replacement for anthropic.Anthropic() client.
    Uses Gemini REST API directly — no SDK imports required.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.messages = self  # mimic anthropic.client.messages

    # ------------------------------------------------------------------
    # Anthropic-compatible entry point
    # ------------------------------------------------------------------
    def create(self, model, max_tokens, system, tools, messages):
        # Build Gemini function declarations from Anthropic tool schemas
        fn_declarations = []
        for t in tools:
            props = t["input_schema"].get("properties", {})
            required = t["input_schema"].get("required", [])
            # Gemini expects simple type strings
            gemini_props = {}
            for k, v in props.items():
                gemini_props[k] = {
                    "type": v.get("type", "string").upper(),
                    "description": v.get("description", ""),
                }
            fn_declarations.append({
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "OBJECT",
                    "properties": gemini_props,
                    "required": required,
                },
            })

        # Convert Anthropic messages → Gemini contents
        contents = self._convert_messages(messages)

        payload = {
            "system_instruction": {
                "parts": [{"text": system}]
            },
            "tools": [{"function_declarations": fn_declarations}],
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.1,
            },
        }

        url = GEMINI_REST_URL.format(model=DEFAULT_MODEL, key=self.api_key)
        resp = requests.post(url, json=payload, timeout=60)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Gemini API error {resp.status_code}: {resp.text[:500]}"
            )

        return self._parse_response(resp.json())

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------
    def _convert_messages(self, messages):
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            content = m["content"]
            parts = []

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        if btype == "tool_result":
                            try:
                                result_data = json.loads(block.get("content", "{}"))
                            except Exception:
                                result_data = {"raw": block.get("content", "")}
                            parts.append({
                                "functionResponse": {
                                    "name": block.get("tool_use_id", "tool"),
                                    "response": result_data,
                                }
                            })
                        elif btype == "text":
                            parts.append({"text": block.get("text", "")})
                        elif btype == "tool_use":
                            parts.append({
                                "functionCall": {
                                    "name": block.get("name", ""),
                                    "args": block.get("input", {}),
                                }
                            })
                    elif isinstance(block, str):
                        parts.append({"text": block})
                    else:
                        # SDK objects from previous turns
                        t = getattr(block, "type", None)
                        if t == "text":
                            parts.append({"text": getattr(block, "text", "")})
                        elif t == "tool_use":
                            parts.append({
                                "functionCall": {
                                    "name": getattr(block, "name", ""),
                                    "args": getattr(block, "input", {}),
                                }
                            })
            else:
                parts.append({"text": str(content)})

            if parts:
                contents.append({"role": role, "parts": parts})

        return contents

    # ------------------------------------------------------------------
    # Response parsing → Anthropic-style response
    # ------------------------------------------------------------------
    def _parse_response(self, data):
        class Block:
            def __init__(self, type_val, **kwargs):
                self.type = type_val
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class AnthropicResponse:
            def __init__(self, content, stop_reason):
                self.content = content
                self.stop_reason = stop_reason

        blocks = []
        stop_reason = "end_turn"

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return AnthropicResponse(
                    [Block("text", text=data.get("error", {}).get("message", "No response"))],
                    "end_turn"
                )

            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    blocks.append(Block("text", text=part["text"]))
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    blocks.append(Block(
                        "tool_use",
                        name=fc.get("name", ""),
                        id=fc.get("name", ""),
                        input=fc.get("args", {}),
                    ))
                    stop_reason = "tool_use"

            # Check finish reason
            finish = candidates[0].get("finishReason", "STOP")
            if finish in ("MAX_TOKENS",):
                stop_reason = "max_tokens"

        except Exception as exc:
            blocks = [Block("text", text=f"Parse error: {exc}")]

        if not blocks:
            blocks = [Block("text", text="")]

        return AnthropicResponse(blocks, stop_reason)
