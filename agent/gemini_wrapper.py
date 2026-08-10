import google.generativeai as genai
import json

class GeminiAnthropicWrapper:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.messages = self

    def create(self, model, max_tokens, system, tools, messages):
        gemini_tools = []
        for t in tools:
            # Convert Anthropic schema to Gemini schema
            gemini_tools.append({
                "function_declarations": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": {
                            "type": "OBJECT",
                            "properties": t["input_schema"].get("properties", {}),
                            "required": t["input_schema"].get("required", [])
                        }
                    }
                ]
            })

        gen_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system,
            tools=gemini_tools
        )

        gemini_messages = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            content = m["content"]
            
            # Handle tool results (Anthropic format -> Gemini format)
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        parts.append({
                            "functionResponse": {
                                "name": block["tool_use_id"], # We store tool name in tool_use_id for Gemini
                                "response": json.loads(block["content"])
                            }
                        })
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append({"text": block["text"]})
                    elif isinstance(block, str):
                        parts.append({"text": block})
                gemini_messages.append({"role": role, "parts": parts})
            else:
                gemini_messages.append({"role": role, "parts": [{"text": str(content)}]})

        response = gen_model.generate_content(gemini_messages)

        # Convert Gemini response to Anthropic format
        class Block:
            def __init__(self, type_val, text=None, name=None, id_val=None, input_val=None):
                self.type = type_val
                if text: self.text = text
                if name: self.name = name
                if id_val: self.id = id_val
                if input_val: self.input = input_val

        class AnthropicResponse:
            def __init__(self, content, stop_reason):
                self.content = content
                self.stop_reason = stop_reason

        blocks = []
        stop_reason = "end_turn"
        
        if response.parts:
            for part in response.parts:
                if hasattr(part, "text") and part.text:
                    blocks.append(Block("text", text=part.text))
                elif hasattr(part, "function_call") and part.function_call:
                    args = type(part.function_call).to_dict(part.function_call)["args"]
                    blocks.append(Block("tool_use", name=part.function_call.name, id_val=part.function_call.name, input_val=args))
                    stop_reason = "tool_use"
        
        return AnthropicResponse(blocks, stop_reason)
