# QwenFlash API

OpenAI-compatible: **Chat Completions** and **Responses**. Tested with the official `openai` Python SDK: model
list, chat, streaming, `reasoning_effort`, a full tool-call round trip, and the Responses API (plain, streamed, tools,
`previous_response_id`).

```
Base URL:  http://127.0.0.1:30000/v1                   (this PC)
           http://127.0.0.1:11434/v1                   (optional front door on Ollama's port, off by default)
           http://<tailnet-host>:11434/v1              (tailnet, via tailscale serve)
API key:   not checked (send any non-empty string if your client insists)
Model:     qwen3.8-flash-next   (any name works: Ollama names such as qwen3.8 are answered by QwenFlash too)
```

Port 11434 and the tailnet URL are the optional "front door" for replacing an Ollama install
([manager guide](manager.md#replacing-ollama-on-a-tailnet-windows-example)). It serves the same API, but never the
dashboard/control API, and `GET /` returns `QwenFlash is running` for health checks.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/chat/completions` | Main endpoint. Streaming or not, tools, reasoning control |
| POST | `/v1/responses` | OpenAI's newer Responses API, on the same model: streaming or not, function tools, reasoning, `previous_response_id`. See [Responses API](#responses-api) |
| POST | `/v1/completions` | Legacy raw-prompt completion. Takes a single prompt; only the first of a list is used |
| GET | `/v1/models` | Answered by the switcher without loading the model |

Not available: `/v1/embeddings`, images, audio, files, batches, assistants, Anthropic `/v1/messages`, and
Ollama's native `/api/*`. These return 404 with a message saying what is supported. With the front door on and
Ollama running alongside on `127.0.0.1:11435`, these are passed through to it instead.

## Request fields (`/v1/chat/completions`)

| Field | Supported | Notes |
|---|---|---|
| `messages` | yes | `system`, `user`, `assistant` (incl. `tool_calls`), `tool`. **Text only**: the model has a vision encoder, but this server has no image handling |
| `stream` | yes | Server-sent events (SSE), see below |
| `max_tokens` / `max_completion_tokens` | yes | Default: everything left in the context window |
| `temperature`, `top_p` | yes | Defaults depend on the thinking mode, see below |
| `top_k`, `min_p` | yes | Non-standard (vLLM-style). SDK: pass via `extra_body` |
| `stop` | yes | String or list |
| `tools` | yes | OpenAI function schema. Calls come back as standard `tool_calls` |
| `tool_choice` | partial | `"none"` disables tools. Any other value behaves like `"auto"` |
| `reasoning_effort` | yes | The model has `low`, `medium`, `xhigh` (default). OpenAI values are mapped: `minimal`→`low`, `high`/`max`→`xhigh`, `none`→thinking off; unknown values are dropped. `reasoning.effort` is accepted too. The engine's server does this itself (`qwen38-flash-offload/qwen38/openai_compat.py`), so it also applies without the manager. (Unmapped, `high` made the model's template fail with HTTP 500, as Hermes Agent hit.) |
| `chat_template_kwargs.enable_thinking` | yes | Non-standard (vLLM/SGLang-style). Default `true`. SDK: `extra_body` |
| `n` | **ignored** | Always one choice |
| `response_format` | **ignored** | No JSON mode or structured outputs. Ask for JSON in the prompt instead |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias`, `logprobs`, `seed`, `parallel_tool_calls` | **ignored** | |

**Sampling defaults when you send nothing** (these come from the model card; anything you send overrides them):

| Mode | temperature | top_p | top_k | min_p |
|---|---|---|---|---|
| Thinking on (default) | 1.0 | 0.95 | 20 | 0 |
| Thinking off | 0.7 | 0.80 | 20 | 0 |

The model card also recommends `presence_penalty=1.5` for thinking-off mode. This server can't apply it, so
watch for repetition in long thinking-off outputs.

## Responses

- **Body:** a standard `chat.completion`. The answer is in `message.content`; the model's thinking is in
  `message.reasoning_content` (the DeepSeek/vLLM convention). Clients that don't know that field just ignore it.
- **`finish_reason`:** `stop`, `length`, or `tool_calls`.
- **Output cut off while thinking:** `content` is `""` and the partial thought is in `reasoning_content`. Give
  thinking requests enough `max_tokens`.
- **Streaming:**
  - It's a series of `chat.completion.chunk` events.
  - `delta.reasoning_content` streams first, then `delta.content`.
  - Tool calls arrive whole, in one delta at the end; arguments are not streamed piece by piece.
  - The last chunk carries `finish_reason` and `usage`, followed by `data: [DONE]`.

## Responses API

`POST /v1/responses` runs on the same pipeline as chat, so the model, sampling, tools and reasoning behave the same.
The engine translates (`qwen38-flash-offload/qwen38/openai_responses.py`), so it also works without the manager.

| Field | Supported | Notes |
|---|---|---|
| `input` | yes | A string, or items: messages (`user`, `assistant`, `system`, `developer`), `function_call`, `function_call_output`. `reasoning` items are skipped. **Text only**: images, files and audio return 400 |
| `instructions` | yes | Becomes the system message, merged with leading `developer`/`system` messages |
| `stream` | yes | Standard Responses events, see below |
| `tools` | partial | `function` tools. Built-in tools (web search, file search, computer use, code interpreter, MCP, custom/freeform) are left out |
| `tool_choice` | partial | `"none"` disables tools; anything else behaves like `"auto"` |
| `previous_response_id` | yes | The server remembers the last 64 responses in memory while it runs. After a restart, send the whole conversation as `input` instead. `store: false` skips remembering |
| `reasoning.effort` | yes | Mapped like chat's `reasoning_effort` (`minimal`, `low`, `medium`, `high`, `max`, `none`) |
| `max_output_tokens`, `temperature`, `top_p` | yes | As in chat |
| `text.format`, `include`, `truncation`, `parallel_tool_calls`, `metadata` | **accepted, ignored** | No structured outputs; `metadata` is echoed back |

- **Output:** a `reasoning` item with the model's thinking as `reasoning_text` content (the convention of other
  open-weight reasoning models, e.g. gpt-oss on vLLM), a `message` item with the answer, and one `function_call`
  item per tool call. The SDK's `response.output_text` gives the answer.
- **Streaming events:** `response.created`, `response.in_progress`, then per item `response.output_item.added`,
  `response.content_part.added`, `response.reasoning_text.delta` / `response.output_text.delta`, the matching
  `.done` events, `response.output_item.done`; tool calls arrive whole as `response.function_call_arguments.delta`
  and `.done`. The last event is `response.completed`, or `response.incomplete` when `max_output_tokens` cut it off.
- **Usage:** `input_tokens` and `output_tokens`; cached and reasoning token counts are reported as 0.

```python
r = client.responses.create(model="qwen3.8-flash-next", input="Explain B-trees in two sentences.",
                            reasoning={"effort": "medium"})
print(r.output_text)
follow_up = client.responses.create(model="qwen3.8-flash-next", input="Now in one.", previous_response_id=r.id)
```

## QwenFlash-specific behaviour

- **One request generates at a time.** Others wait in a queue of 8. Set agent tools to run one request at a time
  if they let you.
- **Cold start:** if the model is paused, the first request waits while it loads (about 35-45 s). Nothing is sent
  in that time, so client timeouts should be **at least 120 s**.
- **Prefix cache:** repeated prompt prefixes are reused. Measured: a 94K-token prompt took 15 s the first time and
  0.14 s when repeated. Agent conversations benefit automatically.
- **Errors** use OpenAI's error shape, `{"error": {"message", "type", "code"}}`:

  | Status | Meaning |
  |---|---|
  | `503 server_busy` | The model couldn't be started: other apps hold the GPU/RAM for longer than the configured wait, wake-on-request is off, or a setup step is running |
  | `502` | The model server is unreachable |
  | `400` | The prompt leaves no room in the context window |

## Recommended client settings

| Setting | Value | Why |
|---|---|---|
| Context window | `262144` | Matches the server. The model was benchmarked at 256K, and 256K costs only ~2.3 GB more VRAM than 16K |
| Max output tokens | 32K or more (or unset) | Thinking can be long. A low cap gives an empty answer with a truncated thought |
| Temperature / top_p | **unset** | The server applies the model card's values. If a client forces a low "coding" temperature, set it to 1.0 for thinking mode |
| `reasoning_effort` | `medium` for agent loops; default (`xhigh`) for hard problems; `low` for quick questions | Lower effort means less thinking and faster turns. Quality per level wasn't measured here |
| Thinking off | for trivial jobs (commit messages, renames, autocomplete) | Fastest; replies with no thinking at all |
| Request timeout | ≥ 120 s | Covers a cold start |
| Parallel requests / subagents | 1 | The server generates one request at a time |

## Examples

```bash
curl http://127.0.0.1:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "Explain B-trees in two sentences."}],
  "reasoning_effort": "medium"
}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="local")

# thinking on (default), streamed
for chunk in client.chat.completions.create(
        model="qwen3.8-flash-next", stream=True, reasoning_effort="medium",
        messages=[{"role": "user", "content": "Write a Python LRU cache."}]):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# thinking off, extra sampling knobs
r = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[{"role": "user", "content": "Commit message for: fix off-by-one in pager"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20})
print(r.choices[0].message.content)
```

## Dashboard / control API (not OpenAI)

Under `http://127.0.0.1:30000/_switcher/api/`. This is what the dashboard uses, and you can script it. When the
manager listens on a non-loopback address, these need the dashboard token (`Authorization: Bearer <token>`).

| Method | Path | Body / result |
|---|---|---|
| GET | `status` | State, profile (full/lean) and VRAM budget, RAM cache, plans that fit now, pressure, GPU/RAM, request counters, speed summary, learned limits, engine sync, recommendations, events |
| GET | `metrics` | The last 100 requests: tokens in/out, wait, time to first token, generation and prompt speed |
| GET | `hardware[?refresh=1]` | Full detection: platform, WSL and `.wslconfig`, GPUs (VRAM, architecture, PCIe link), CPU, RAM, drives, engine side, recommendations |
| GET / POST | `config` | POST `{"config": {"quant": "4.05", "watchlist": ["game.exe"]}}`. Keys are the names in `config.json`; 0 means automatic for the VRAM/RAM sizes |
| POST | `action` | `{"action": "start_full"}`. Also `start_lean`, `pause`, `restart`, `auto_on`, `auto_off`, `wake_on`, `wake_off`, `autostart_on`, `autostart_off`, `forget_learned`, `quit` (stops the model too), `quit_keep_model` |
| GET | `log` | Tails of the model server's log and the manager's log |
| GET | `setup?since=N` | Platform, setup steps with status (`ok`, `missing`, `outdated`, `running`, `failed`) and output lines |
| POST | `setup/check` | Re-run detection |
| POST | `setup/run` | `{"step": "build"}`. Step ids: `wsl`, `wslconfig`, `distro` (Windows), `driver` (Linux), `base`, `engine`, `build`, `m305`, `m405` |

The model server's own extras (`/health`, `/stats`, `/server_info`, `/tokenize`, `/generate`) exist only on its
internal port 30001. `/stats` includes the expert cache (slots, hit rate) and where expert picks were served from
(VRAM, RAM, NVMe). Its FastAPI `/docs` page lists routes but has no request schemas.
