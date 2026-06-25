"""
Shared test helpers for agent layer tests.
"""

import copy


def build_sample_schedule(with_faculty_clash=False, with_room_clash=False):
    schedule = {
        "BTECH": {
            "Sem_2_Section_A": {
                "Monday": {
                    "09:00-10:00": {
                        "subject": "Physics",
                        "faculty": "Dr. Mehta",
                        "room": "LH-101",
                        "type": "THEORY",
                    },
                    "10:00-11:00": None,
                    "11:00-12:00": {
                        "subject": "Math",
                        "faculty": "Dr. Singh",
                        "room": "LH-102",
                        "type": "THEORY",
                    },
                },
                "Tuesday": {
                    "09:00-10:00": None,
                    "10:00-11:00": None,
                    "11:00-12:00": None,
                },
            },
            "Sem_2_Section_B": {
                "Monday": {
                    "09:00-10:00": {
                        "subject": "Chemistry",
                        "faculty": "Dr. Mehta" if with_faculty_clash else "Dr. Rao",
                        "room": "LH-103" if not with_room_clash else "LH-101",
                        "type": "THEORY",
                    },
                    "10:00-11:00": None,
                    "11:00-12:00": None,
                },
                "Tuesday": {
                    "09:00-10:00": None,
                    "10:00-11:00": None,
                    "11:00-12:00": None,
                },
            },
        }
    }
    return schedule


class MockFirebase:
    """FirebaseManager-compatible mock for agent Firebase tests."""

    def __init__(self):
        self.saved = {}

    class _Collection:
        def __init__(self, outer, name):
            self.outer = outer
            self.name = name

        def document(self, doc_id):
            outer = self.outer
            name = self.name

            class _Doc:
                def set(self, payload, merge=False):
                    key = f"{name}/{doc_id}"
                    if merge and key in outer.saved:
                        outer.saved[key].update(payload)
                    else:
                        outer.saved[key] = payload

                def get(self):
                    key = f"{name}/{doc_id}"

                    class _Snapshot:
                        def __init__(self, data):
                            self._data = data

                        @property
                        def exists(self):
                            return self._data is not None

                        def to_dict(self):
                            return self._data

                    return _Snapshot(outer.saved.get(key))

            return _Doc()

        def where(self, field, op, value):
            return self

        def limit(self, count):
            return self

        def stream(self):
            return []

    @property
    def db(self):
        return self

    def collection(self, name):
        return self._Collection(self, name)

    def save_agent_session(self, session_id, session_data):
        self.saved[f"agent_sessions/{session_id}"] = copy.deepcopy(session_data)
        return True, session_id

    def save_repair_history(self, repair_id, repair_data):
        self.saved[f"repair_history/{repair_id}"] = copy.deepcopy(repair_data)
        return True, repair_id

    def get_agent_config(self, config_id="default"):
        key = f"agent_config/{config_id}"
        if key in self.saved:
            return self.saved[key]
        return {
            "max_turns": 10,
            "llm_model": "claude-sonnet-4-5",
            "enabled": True,
            "fallback_to_random_repair": True,
        }

    def save_agent_config(self, config_id, config_data):
        self.saved[f"agent_config/{config_id}"] = copy.deepcopy(config_data)
        return True, config_id

    def get_repair_history(self, session_id=None, limit=50):
        prefix = "repair_history/"
        items = []
        for key, value in self.saved.items():
            if not key.startswith(prefix):
                continue
            if session_id and value.get("session_id") != session_id:
                continue
            item = copy.deepcopy(value)
            item["id"] = key.split("/", 1)[1]
            items.append(item)
        return items[:limit]


class MockToolUseBlock:
    def __init__(self, block_id, name, input_data):
        self.type = "tool_use"
        self.id = block_id
        self.name = name
        self.input = input_data


class MockTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class MockAnthropicResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class MockAnthropicMessages:
    def __init__(self, steps):
        self.steps = steps
        self.call_count = 0

    def create(self, **kwargs):
        if self.call_count >= len(self.steps):
            return MockAnthropicResponse(
                [MockTextBlock("Repair complete.")],
                "end_turn",
            )

        step = self.steps[self.call_count]
        self.call_count += 1

        if step is None:
            return MockAnthropicResponse(
                [MockTextBlock("Repair complete.")],
                "end_turn",
            )

        return MockAnthropicResponse(
            [MockToolUseBlock(f"tool_{self.call_count}", step["name"], step["input"])],
            "tool_use",
        )


class MockAnthropicClient:
    def __init__(self, steps):
        self.messages = MockAnthropicMessages(steps)
