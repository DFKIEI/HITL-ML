import dataclasses
import json
import os
from typing import List

import torch

from helpers_latent import (
    build_global_summary_semantic,
    build_pair_summary_semantic,
    compute_centroids,
    compute_latents,
    compute_thresholds,
    pairwise_distances,
)


@dataclasses.dataclass
class LLMConstraint:
    class_i: int
    class_j: int
    target: str  # "close", "medium", "far"
    reason: str


@dataclasses.dataclass
class LLMState:
    t_very_close: float
    t_close: float
    t_far: float
    t_very_far: float
    constraints: List[LLMConstraint]
    num_classes: int
    class_names: List[str]
    suggestion_count: int = 0


def extract_response_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    if hasattr(response, "output_parsed") and response.output_parsed:
        return json.dumps(response.output_parsed)
    if hasattr(response, "output_json") and response.output_json:
        return json.dumps(response.output_json)
    if hasattr(response, "output") and response.output:
        for item in response.output:
            if isinstance(item, dict):
                if item.get("type") == "output_text" and item.get("text"):
                    return item["text"]
                for key in ["parsed", "json"]:
                    if key in item and item[key] is not None:
                        return json.dumps(item[key])
                content_list = item.get("content")
            else:
                if getattr(item, "type", None) == "output_text" and getattr(item, "text", None):
                    return item.text
                content_list = getattr(item, "content", None)

            if content_list:
                for content in content_list:
                    if isinstance(content, dict):
                        if content.get("type") == "output_text" and content.get("text"):
                            return content["text"]
                        for key in ["parsed", "json", "text"]:
                            if key in content and content[key] is not None:
                                return json.dumps(content[key]) if key in {"parsed", "json"} else content[key]
                    else:
                        if getattr(content, "type", None) == "output_text" and getattr(content, "text", None):
                            return content.text
                        if getattr(content, "text", None):
                            return content.text
    raise ValueError("Could not extract text output from OpenAI response.")


def save_prompt_dump(prompt_dump_path: str, prompt: str) -> None:
    payload = {
        "prompt_lines": prompt.splitlines(),
    }
    os.makedirs(os.path.dirname(prompt_dump_path), exist_ok=True)
    with open(prompt_dump_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_response_dump(response_dump_path: str, raw_text: str) -> None:
    payload = None
    try:
        payload = json.loads(raw_text)
    except Exception:
        payload = {"response_lines": raw_text.splitlines()}

    os.makedirs(os.path.dirname(response_dump_path), exist_ok=True)
    with open(response_dump_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def call_openai_llm(
    prompt: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    max_constraints: int,
    debug: bool,
    num_classes: int,
) -> tuple[List[LLMConstraint], str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it to use --llm-mode openai.")

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai") from exc

    suggestions_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "class_i": {"type": "integer"},
                "class_j": {"type": "integer"},
                "suggestion": {"type": "string"},
            },
            "required": ["class_i", "class_j", "suggestion"],
            "additionalProperties": False,
        },
    }
    if max_constraints:
        suggestions_schema["maxItems"] = max_constraints

    global_schema = {
        "type": "object",
        "properties": {
            "issue": {"type": "string"},
            "strategy": {"type": "string"},
        },
        "required": ["issue", "strategy"],
        "additionalProperties": False,
    }

    schema = {
        "name": "latent_suggestions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "global": global_schema,
                "suggestions": suggestions_schema,
            },
            "required": ["global", "suggestions"],
            "additionalProperties": False,
        },
    }

    reasoning = None
    if model.startswith("gpt-5"):
        temperature = None
        reasoning = {"effort": "low"}

    client = OpenAI()

    def _call_chat_completion(max_tokens_override: int | None = None) -> str:
        if not (hasattr(client, "chat") and hasattr(client.chat, "completions")):
            raise RuntimeError("OpenAI SDK is missing Chat Completions API.")
        messages = [
            {"role": "system", "content": "Return only JSON that matches the requested schema."},
            {"role": "user", "content": prompt},
        ]
        temp = temperature
        max_tokens = max_tokens_override or max_output_tokens
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "temperature" in str(exc).lower():
                temp = None
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                }
                if temp is not None:
                    kwargs["temperature"] = temp
                response = client.chat.completions.create(**kwargs)
            except Exception as exc2:
                if "temperature" in str(exc2).lower():
                    temp = None
                kwargs = {"model": model, "messages": messages}
                if temp is not None:
                    kwargs["temperature"] = temp
                response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    if hasattr(client, "responses"):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning=reasoning,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "latent_suggestions",
                        "schema": schema["schema"],
                    }
                },
            )
        except Exception as exc:
            if "temperature" in str(exc).lower():
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=max_output_tokens,
                    reasoning=reasoning,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "latent_suggestions",
                            "schema": schema["schema"],
                        }
                    },
                )
            else:
                raise

        if getattr(response, "status", None) == "incomplete":
            if debug and hasattr(response, "model_dump"):
                print(f"[LLM] response model_dump: {response.model_dump()}")
            incomplete = getattr(response, "incomplete_details", None)
            reason = None
            if incomplete is not None:
                reason = getattr(incomplete, "reason", None)
                if reason is None and isinstance(incomplete, dict):
                    reason = incomplete.get("reason")
            if reason == "max_output_tokens":
                retry_tokens = min(max_output_tokens * 2, 1200)
                if debug:
                    print(f"[LLM] retrying with max_output_tokens={retry_tokens}")
                try:
                    response = client.responses.create(
                        model=model,
                        input=prompt,
                        max_output_tokens=retry_tokens,
                        reasoning=reasoning,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "latent_suggestions",
                                "schema": schema["schema"],
                            }
                        },
                    )
                except Exception:
                    response = None

            if response is not None:
                try:
                    raw_text = extract_response_text(response)
                except Exception:
                    raw_text = ""
            if response is None or getattr(response, "status", None) == "incomplete" or not raw_text:
                raw_text = _call_chat_completion(max_tokens_override=min(max_output_tokens * 2, 1200))
        else:
            try:
                raw_text = extract_response_text(response)
            except Exception as exc:
                if debug:
                    if hasattr(response, "model_dump"):
                        print(f"[LLM] response model_dump: {response.model_dump()}")
                    else:
                        print(f"[LLM] response repr: {response}")
                    print(f"[LLM] extract_response_text failed: {exc}")
                raw_text = _call_chat_completion()
    else:
        raw_text = _call_chat_completion()

    if not raw_text or not raw_text.strip():
        raw_text = _call_chat_completion(max_tokens_override=min(max_output_tokens * 2, 1200))

    if debug:
        preview = raw_text if len(raw_text) <= 2000 else raw_text[:2000] + " ...[truncated]"
        print(f"[LLM] raw response: {preview}")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        if debug:
            print(f"[LLM] JSON decode failed: {exc}")
        return [], raw_text
    _ = payload.get("suggestions", [])
    # Freeform mode: keep training constraints disabled for now.
    return [], raw_text


def update_llm_state(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int,
    q_very_close: float,
    q_close: float,
    q_far: float,
    q_very_far: float,
    llm_model: str,
    llm_temperature: float,
    llm_max_output_tokens: int,
    llm_max_constraints: int,
    llm_max_pairs: int,
    llm_debug: bool,
    num_classes: int,
    class_names: List[str],
    dataset_name: str,
    prompt_dump_path: str | None = None,
    response_dump_path: str | None = None,
) -> LLMState:
    z, y = compute_latents(model, loader, device, max_samples)
    centroids = compute_centroids(z, y, num_classes)
    distances = pairwise_distances(centroids)
    t_very_close, t_close, t_far, t_very_far = compute_thresholds(
        distances, q_very_close, q_close, q_far, q_very_far
    )

    pairs_summary = build_pair_summary_semantic(
        z=z,
        y=y,
        centroids=centroids,
        distances=distances,
        t_very_close=t_very_close,
        t_close=t_close,
        t_far=t_far,
        t_very_far=t_very_far,
        max_pairs=llm_max_pairs,
        class_names=class_names,
    )
    global_summary = build_global_summary_semantic(
        z=z,
        y=y,
        centroids=centroids,
        distances=distances,
        t_very_close=t_very_close,
        t_close=t_close,
        t_far=t_far,
        t_very_far=t_very_far,
    )
    prompt = (
        f"You are assisting with latent space optimization for a {dataset_name} classifier.\n"
        "We provide only 5-level categorical latent signals (no numeric values):\n"
        "Global semantic metrics:\n"
        "- overall_overlap_level: very_low | low | medium | high | very_high\n"
        "- overall_spread_level: very_compact | compact | medium | spread | very_spread\n"
        "- spread_imbalance_level: very_low | low | medium | high | very_high\n"
        "- outlier_burden_level: very_low | low | medium | high | very_high\n"
        "- separation_health: very_poor | poor | medium | good | very_good\n"
        "Pairwise semantic metrics:\n"
        "- distance_relation: very_close | close | medium | far | very_far\n"
        "- overlap_level: very_low | low | medium | high | very_high\n"
        "- spread_i_level, spread_j_level: very_compact | compact | medium | spread | very_spread\n"
        "- outlier_i_level, outlier_j_level: very_low | low | medium | high | very_high\n"
        f"Return up to {llm_max_constraints} freeform suggestions to improve the latent space.\n"
        "Output MUST be strict JSON with exactly this shape and no extra keys:\n"
        "{ \"global\": { \"issue\": \"string\", \"strategy\": \"string\" }, "
        "\"suggestions\": [ { \"class_i\": int, \"class_j\": int, \"suggestion\": \"string\" } ] }\n"
        "The \"global.issue\" should summarize the main latent-space problem overall.\n"
        "The \"global.strategy\" should give one overall latent-space movement strategy (not implementation details).\n"
        "The \"suggestion\" field must include both:\n"
        "1) a latent-space movement intention for this pair (e.g., move closer, move farther, tighten or separate overlap region), and\n"
        "2) the reason why this movement is needed based on the provided signals.\n"
        "Do NOT provide algorithmic or implementation instructions. "
        "Keep suggestions focused on semantic movement in latent space.\n\n"
        "Data (JSON):\n"
        f"{json.dumps({'global': global_summary, 'pairs': pairs_summary}, indent=2)}\n"
    )
    if llm_debug:
        preview = prompt if len(prompt) <= 4000 else prompt[:4000] + " ...[truncated]"
        print(f"[LLM] prompt: {preview}")
    if prompt_dump_path:
        save_prompt_dump(prompt_dump_path, prompt)
    try:
        constraints, raw_text = call_openai_llm(
            prompt=prompt,
            model=llm_model,
            temperature=llm_temperature,
            max_output_tokens=llm_max_output_tokens,
            max_constraints=llm_max_constraints,
            debug=llm_debug,
            num_classes=num_classes,
        )
        if response_dump_path:
            save_response_dump(response_dump_path, raw_text)
        suggestion_count = 0
        try:
            suggestion_count = len(json.loads(raw_text).get("suggestions", []))
        except Exception:
            suggestion_count = 0
    except Exception as exc:
        print(f"[LLM] OpenAI call failed: {exc}")
        constraints = []
        suggestion_count = 0

    return LLMState(
        t_very_close=t_very_close,
        t_close=t_close,
        t_far=t_far,
        t_very_far=t_very_far,
        constraints=constraints,
        num_classes=num_classes,
        class_names=class_names,
        suggestion_count=suggestion_count,
    )
