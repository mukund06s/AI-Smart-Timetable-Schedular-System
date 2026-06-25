"""
Plain-English repair explanations via LLM (Phase 4 UI polish).
"""

from typing import Any, Dict, Optional


def build_explain_prompt(repair_entry: Dict[str, Any]) -> str:
    return (
        "Explain the following timetable repair action in simple plain English "
        "for a college administrator. Keep it under 120 words.\n\n"
        f"Action Type: {repair_entry.get('action_type', 'unknown')}\n"
        f"From Slot: {repair_entry.get('from_slot', {})}\n"
        f"To Slot: {repair_entry.get('to_slot', {})}\n"
        f"Reason: {repair_entry.get('reason', '')}\n"
        f"Success: {repair_entry.get('success', repair_entry.get('result', True))}\n"
    )


def explain_repair_plain_english(
    repair_entry: Dict[str, Any],
    llm_client: Optional[Any] = None,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-5",
) -> str:
    """Return a plain-English explanation of one repair action."""
    prompt = build_explain_prompt(repair_entry)

    if llm_client is not None:
        response = llm_client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                return block.text.strip()

    if api_key:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                return block.text.strip()

    from_slot = repair_entry.get("from_slot", {})
    to_slot = repair_entry.get("to_slot", {})
    action = repair_entry.get("action_type", "repair")
    reason = repair_entry.get("reason", "resolve a scheduling clash")
    return (
        f"The agent performed a {action} by moving a class from "
        f"{from_slot.get('day', '?')} {from_slot.get('slot', '?')} to "
        f"{to_slot.get('day', '?')} {to_slot.get('slot', '?')} "
        f"because {reason}. This was intended to remove a timetable clash "
        f"while keeping faculty and room constraints valid."
    )
