"""Lightweight JSON Schema compatibility helpers for vLLM xgrammar."""
import copy

_VALUE_KEYWORDS = frozenset({"enum", "const", "default", "examples"})
_SCHEMA_MAP_KEYWORDS = frozenset(
    {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
)
_SCHEMA_LIST_KEYWORDS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
_SUBSCHEMA_KEYWORDS = frozenset(
    {
        "items",
        "additionalProperties",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
    }
)
_DEFENSIVE_UNSUPPORTED = frozenset({"multipleOf", "uniqueItems"})


def _is_string_compatible(node):
    node_type = node.get("type")
    if node_type is None or node_type == "string":
        return True
    return isinstance(node_type, list) and "string" in node_type


def _sanitize_schema_node(node):
    if not isinstance(node, dict):
        return copy.deepcopy(node)

    drop_bounds = (
        _is_string_compatible(node)
        and ("pattern" in node or "format" in node)
        and ("minLength" in node or "maxLength" in node)
    )

    cleaned = {}
    for key, value in node.items():
        if key in _DEFENSIVE_UNSUPPORTED:
            continue
        if drop_bounds and key in ("minLength", "maxLength"):
            continue
        if key in _VALUE_KEYWORDS:
            cleaned[key] = copy.deepcopy(value)
        elif key in _SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            cleaned[key] = {
                prop_name: _sanitize_schema_node(sub_schema)
                for prop_name, sub_schema in value.items()
            }
        elif key in _SCHEMA_LIST_KEYWORDS and isinstance(value, list):
            cleaned[key] = [
                _sanitize_schema_node(item) if isinstance(item, dict) else copy.deepcopy(item)
                for item in value
            ]
        elif key in _SUBSCHEMA_KEYWORDS:
            if isinstance(value, dict):
                cleaned[key] = _sanitize_schema_node(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    _sanitize_schema_node(item) if isinstance(item, dict) else copy.deepcopy(item)
                    for item in value
                ]
            else:
                cleaned[key] = copy.deepcopy(value)
        else:
            cleaned[key] = copy.deepcopy(value)
    return cleaned


def sanitize_json_schema(schema):
    """Return a copy of schema safe for vLLM xgrammar backend."""
    if not isinstance(schema, dict):
        return copy.deepcopy(schema)
    return _sanitize_schema_node(schema)


def apply_xgrammar_schema_compat(payload):
    """Sanitize unsupported string pattern+length combinations in gateway payloads."""
    if not isinstance(payload, dict):
        return payload

    text = payload.get("text")
    if isinstance(text, dict):
        fmt = text.get("format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema" and isinstance(fmt.get("schema"), dict):
            fmt["schema"] = sanitize_json_schema(fmt["schema"])

    rf = payload.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        js = rf.get("json_schema")
        if isinstance(js, dict) and isinstance(js.get("schema"), dict):
            js["schema"] = sanitize_json_schema(js["schema"])

    so = payload.get("structured_outputs")
    if isinstance(so, dict) and isinstance(so.get("json"), dict):
        so["json"] = sanitize_json_schema(so["json"])

    return payload
