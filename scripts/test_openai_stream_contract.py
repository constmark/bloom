#!/usr/bin/env python3
"""Regression checks for tool-call SSE framing used by the native smoke gate."""

import json
import unittest

from openai_compat_smoke import collect_candidate_stream


def frame(delta, finish=None):
    return json.dumps({
        "id": "chatcmpl-tools", "created": 123, "model": "fixture",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })


def usage():
    return json.dumps({
        "id": "chatcmpl-tools", "created": 123, "model": "fixture",
        "object": "chat.completion.chunk", "choices": [],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    })


class ToolStreamContractTests(unittest.TestCase):
    def test_single_call_accepts_combined_or_separate_terminal_and_argument_fragments(self):
        call = {"index": 0, "id": "call_weather", "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
        variants = [
            [frame({"tool_calls": [call]}, "tool_calls")],
            [frame({"tool_calls": [call]}), frame({}, "tool_calls")],
            [frame({"tool_calls": [dict(call, function={"name": "get_weather", "arguments": '{"city":'})]}),
             frame({"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
             frame({}, "tool_calls")],
        ]
        for middle in variants:
            with self.subTest(frames=len(middle)):
                result = collect_candidate_stream(
                    [frame({"role": "assistant", "content": ""}), *middle, usage(), "[DONE]"],
                    "fixture", 1, chat=True,
                )
                choice = result["choices"][0]
                self.assertEqual(choice["finish_reason"], "tool_calls")
                self.assertEqual(choice["message"]["content"], "")
                received = choice["message"]["tool_calls"][0]
                self.assertEqual(received["id"], "call_weather")
                self.assertEqual(received["function"]["name"], "get_weather")
                self.assertEqual(json.loads(received["function"]["arguments"]), {"city": "Paris"})

    def test_partial_or_duplicated_lifecycles_are_rejected(self):
        start = frame({"role": "assistant", "content": ""})
        end = frame({}, "tool_calls")
        for events in [
            [start, usage(), "[DONE]"],
            [start, end, end, usage(), "[DONE]"],
            [start, end, usage(), "[DONE]", "[DONE]"],
            [start, end, usage(), frame({"content": "late"}), "[DONE]"],
        ]:
            with self.subTest(events=events), self.assertRaises(AssertionError):
                collect_candidate_stream(events, "fixture", 1, chat=True)


if __name__ == "__main__":
    unittest.main()
