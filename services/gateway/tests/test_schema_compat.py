import copy
import unittest

from services.gateway.schema_compat import apply_xgrammar_schema_compat, sanitize_json_schema


SPATIAL_HOME_RESPONSE_SCHEMA = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "enum": ["reply"],
                    "description": (
                        "reply ONLY for ordinary conversation that needs a direct answer "
                        "and does NOT confirm, approve, reject, or request any home action, "
                        "schedule, or pending device proposal."
                    ),
                },
                "reply": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 2400,
                    "description": "Natural answer in the latest user's language for route=reply.",
                },
            },
            "required": ["route", "reply"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "enum": ["recent_events", "home_task"],
                    "description": (
                        "recent_events for general recent home updates; home_task for home work, "
                        "actions, schedules, or confirmations/approvals of ANY pending device "
                        "proposal (standard OR high-risk, including locks)."
                    ),
                },
                "reply": {
                    "type": "string",
                    "enum": [""],
                    "description": "Must be empty string for recent_events and home_task.",
                },
            },
            "required": ["route", "reply"],
            "additionalProperties": False,
        },
    ]
}


class SchemaCompatTest(unittest.TestCase):
    def test_strips_min_max_length_when_pattern_or_format_present_across_all_slots(self):
        schema = {
            "type": "object",
            "$defs": {
                "Slug": {
                    "type": "string",
                    "pattern": "^[a-z0-9_-]{1,40}$",
                    "minLength": 1,
                    "maxLength": 40,
                }
            },
            "properties": {
                "optional_email": {
                    "anyOf": [
                        {"type": "string", "format": "email", "minLength": 3, "maxLength": 120},
                        {"type": "null"},
                    ]
                },
                "plain_str": {"type": "string", "minLength": 2, "maxLength": 10},
                "items_list": {"type": "array", "uniqueItems": True, "items": {"type": "number", "multipleOf": 0.5}},
            },
        }
        for slot_builder in (
            lambda s: {"response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": copy.deepcopy(s)}}},
            lambda s: {"structured_outputs": {"json": copy.deepcopy(s)}},
            lambda s: {"text": {"format": {"type": "json_schema", "name": "t", "schema": copy.deepcopy(s)}}},
        ):
            payload = slot_builder(schema)
            apply_xgrammar_schema_compat(payload)
            if "response_format" in payload:
                out = payload["response_format"]["json_schema"]["schema"]
            elif "structured_outputs" in payload:
                out = payload["structured_outputs"]["json"]
            else:
                out = payload["text"]["format"]["schema"]

            slug = out["$defs"]["Slug"]
            self.assertEqual(slug["pattern"], "^[a-z0-9_-]{1,40}$")
            self.assertNotIn("minLength", slug)
            self.assertNotIn("maxLength", slug)

            opt = out["properties"]["optional_email"]["anyOf"][0]
            self.assertEqual(opt["format"], "email")
            self.assertNotIn("minLength", opt)
            self.assertNotIn("maxLength", opt)

            # Plain string without pattern/format keeps minLength/maxLength.
            self.assertEqual(out["properties"]["plain_str"], {"type": "string", "minLength": 2, "maxLength": 10})
            # Defensive stripping outside literal values.
            self.assertNotIn("uniqueItems", out["properties"]["items_list"])
            self.assertNotIn("multipleOf", out["properties"]["items_list"]["items"])

    def test_spatial_home_response_schema_passes_through_unchanged(self):
        payload = {"text": {"format": {"type": "json_schema", "name": "conv", "strict": True, "schema": copy.deepcopy(SPATIAL_HOME_RESPONSE_SCHEMA)}}}
        apply_xgrammar_schema_compat(payload)
        self.assertEqual(payload["text"]["format"]["schema"], SPATIAL_HOME_RESPONSE_SCHEMA)

    def test_structural_tag_passes_through_unchanged(self):
        rf_tag = {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": [{"begin": "<tool_call>", "content": {"type": "json_schema", "json_schema": {"type": "object"}}, "end": "</tool_call>"}],
                "at_least_one": False,
                "stop_after_first": False,
            },
        }
        payload = {
            "response_format": copy.deepcopy(rf_tag),
            "structured_outputs": {"structural_tag": copy.deepcopy(rf_tag["format"])},
        }
        expected = copy.deepcopy(payload)
        apply_xgrammar_schema_compat(payload)
        self.assertEqual(payload, expected)

    def test_property_names_matching_schema_keywords_are_preserved(self):
        schema = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "minLength": 1},
                "minLength": {"type": "integer"},
                "maxLength": {"type": "integer"},
                "multipleOf": {"type": "integer"},
                "uniqueItems": {"type": "boolean"},
            },
            "required": ["pattern", "minLength"],
        }
        out = sanitize_json_schema(schema)
        self.assertEqual(out, schema)
