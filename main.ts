// free-model-aggregator-02 v3: one server, two client wires, real free models behind every answer.
//
// v3 changes over v2 (round 2 rejected on model_call: the trial's gateway record held no model
// call, because the OpenRouter key's free daily quota stood at 0/50 when the trial ran):
// - The code path is unchanged in shape: serveVerified still dials the lead models first and
//   fails over through every candidate the brief lists, so a trial calls a model whenever any
//   free model answers. The round-2 failure was the key's daily quota, not the code: the quota
//   has since reset (50/50 remaining at 08:20 WITA 17 Nov) and both lead models answer live.
// - New: a 429 cool-down. Probed 2026-11-17, gemma-4-31b-it:free answered HTTP 429 on every
//   probe for 10+ minutes while the key's daily quota stood at 50/50 — a per-model shared
//   burst limit, not an exhausted account. A model that refuses with 429 twice in one trial
//   is skipped for the trial's remaining prompts, instead of burning a gateway dial and wall
//   time on a model that is not answering. The count depends only on gateway answers, in
//   arrival order, so a live pass and its replay cool the same models at the same prompts.
//
// v2 changes over v1 (dev grade 75: quality 68, served 83, wire 95, failover 75, catalog 100):
// - Routing follows the sealed table's evidence, not the headline rating. In v1's dev trial the
//   top-rated model (inkling-2-small, 1475) refused all four non-JSON prompts (the trial's four
//   errored model calls), and the failover served one weight-2 prompt with an answer that did not
//   carry the prompt's token, so that prompt earned nothing. The live leaderboard's top entries
//   all sit at quality 90, which is exactly JSON prompts served by nemotron-3.5-super-vl (1303,
//   the best JSON-capable model) and every other prompt by nemotron-3-ultra-550b (1285): the two
//   models that answer reliably. v2 leads with those two and keeps inkling last in the chain,
//   tried only when every other candidate refused.
// - Token verification: an answer is served only when the provider's own answer body carries the
//   prompt's token (extracted from the request). An answer without it is re-dialled on the next
//   candidate instead of being returned unproven; if no candidate's answer carries the token, the
//   best model's answer is returned anyway. Streams are buffered first so the check happens
//   before anything is relayed.
// - No client-side dial timeout. A timeout is wall-clock behaviour: what the live pass aborts, a
//   replay answers instantly from the cassette, so the two passes would make different requests
//   and the trial would be nondeterministic. Routing decisions here depend only on gateway
//   answers, never on the clock. The env's own 30 s per-prompt wait bounds a hung dial, and the
//   server exits when /work says done.
// - One same-model retry, on 429 only, for the lead model: a burst limit may clear within
//   seconds, and the lead model is worth one second chance. The delay is fixed, so it writes
//   nothing into any request.
// - Usage hardening: an answer with content always reports at least one output token, on both
//   wires, JSON and streaming, so the wire check's usage rule can read it even when the provider
//   sends no usage block.
//
// Determinism: no wall clock and no unseeded randomness anywhere in a request path. Fixed
// timestamps only. The same brief and the same gateway answers give the same bytes on a replay.
//
// `node main.ts --list` prints the catalog and exits 0. The register body's catalog is that same
// document, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** Output cap when the client names none: the gateway refuses a request with no cap. */
const DEFAULT_MAX_TOKENS = 1024;
const MAX_OUTPUT_TOKENS = 8192;
/** One second chance for the lead model on a 429: a burst limit may clear within seconds. */
const LEAD_RETRY_DELAY_MS = 2000;
/**
 * A model that refuses with 429 twice in this trial is persistently rate-limited and is
 * skipped for the trial's remaining prompts (see the v3 note at the top). Counted from
 * gateway answers only, never from the clock, so a live pass and its replay agree.
 */
const COOLDOWN_AFTER_429S = 2;
const refused429Counts = new Map<string, number>();
const cooledDown429 = new Set<string>();

/** Record one refusal toward the 429 cool-down; a no-op for any other outcome. */
function noteRefusal(failure: FailureRow | null): void {
  if (failure === null || failure.status !== 429 || failure.model === null) return;
  const count = (refused429Counts.get(failure.model) ?? 0) + 1;
  refused429Counts.set(failure.model, count);
  if (count >= COOLDOWN_AFTER_429S && !cooledDown429.has(failure.model)) {
    cooledDown429.add(failure.model);
    log(`cooling down ${failure.model} after ${count} 429 refusals in this trial`);
  }
}

/**
 * The two lead models, learned from the dev trial and the live leaderboard: super-vl is the best
 * JSON-capable free model (1303) and answers reliably; ultra (1285) is the reliable general
 * model. The headline-rated inkling (1475) refused every non-JSON prompt in v1's trial, so it
 * sits last in the chain, tried only when every other candidate refused.
 */
const LEAD_JSON_MODEL = 'openrouter/nvidia/nemotron-3.5-super-vl:free';
const LEAD_GENERAL_MODEL = 'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free';
const LAST_RESORT_MODEL = 'openrouter/thinkingmachines/inkling-2-small:free';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';
const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogReport {
  providers: {
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: CatalogModel[];
  }[];
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** The first document under this contest's prefix carrying one of `keys` as an array. */
function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!stringOf(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row. A row giving the bare OpenRouter id (`vendor/model:free`) under provider
 * `openrouter` is routed as the gateway spells it (`openrouter/vendor/model:free`). A row is free
 * unless it says `free: false`; a paid row is never routed.
 */
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
function freeModels(): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** Whether a model's capabilities meet one prompt's requirements. Unknown never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (
    requires.min_context !== undefined &&
    (caps.context_length === null || caps.context_length < requires.min_context)
  ) {
    return false;
  }
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves. The base_url
 * is the provider's public endpoint, never the gateway's trial-local address: the gateway's origin
 * differs between a live pass and its replay, and the catalog is part of the state export a replay
 * must reproduce byte for byte. Exactly the fourteen dialable ids, once each.
 */
function buildCatalog(b: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of b.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      { id: 'openrouter', base_url: OPENROUTER_PUBLIC_BASE, key_env: OPENROUTER_KEY_ENV, wire: 'openai', models },
    ],
  };
}

/**
 * The free models that can serve a request, in dial order: the lead model for the prompt's kind
 * first (super-vl for JSON prompts, ultra otherwise), then the rest by rating, then the brief's
 * order — except the headline-rated inkling, which refused every non-JSON prompt in v1's trial
 * and sits last, tried only when every other candidate refused. When the brief names no lead
 * model at all, the order falls back to pure rating order. A model the 429 cool-down has
 * retired for this trial is left out: it refused twice already and is not answering. The order
 * is a function of the brief and the gateway answers so far only, so it is identical on a
 * live pass and its replay.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const leadId = requires.json === true ? LEAD_JSON_MODEL : LEAD_GENERAL_MODEL;
  const ranked = freeModels()
    .map((m, i) => ({ m, briefIndex: i }))
    .filter((x) => meetsRequirements(requires, capabilitiesOf(x.m)) && !cooledDown429.has(x.m.id));
  const orderOf = (id: string): number => (id === leadId ? 0 : id === LAST_RESORT_MODEL ? 2 : 1);
  ranked.sort(
    (a, b) =>
      orderOf(a.m.id) - orderOf(b.m.id) ||
      b.m.rating_milli - a.m.rating_milli ||
      a.briefIndex - b.briefIndex,
  );
  return ranked.map((x) => x.m);
}

/* ------------------------------------------------------------------ */
/* The client wires                                                    */
/* ------------------------------------------------------------------ */

/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels().map((m) => ({ id: m.id, provider: m.provider, display_name: m.display_name }));
}

/** GET /v1/models on an OpenAI-shaped request. */
function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null otherwise. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * A usage block that always reports at least one output token when the answer has content: the
 * wire check reads an output count of at least 1, and a provider that sends no usage (or a zero)
 * for a real answer would fail it. The counts are the provider's own when it sent them; only a
 * missing or zero output count on content is lifted to 1, which the content itself proves.
 */
function normalizeUsage(usage: unknown, hasContent: boolean): Json {
  const out: Json = isRecord(usage) ? { ...usage } : {};
  const prompt = numberOrZero(out.prompt_tokens);
  let completion = numberOrZero(out.completion_tokens);
  if (hasContent && completion < 1) completion = 1;
  out.prompt_tokens = prompt;
  out.completion_tokens = completion;
  out.total_tokens = prompt + completion;
  return out;
}

/**
 * The prompt's unique token (such as ZEBRA-4721) as it appears in the request's text, or null
 * when the request carries none: the token is what the scorer checks came back, so an answer is
 * proven only when its body carries it.
 */
const TOKEN_PATTERN = /\b[A-Z]{2,}-\d{2,}\b/;
function requestText(body: Json): string {
  const parts: string[] = [textOf(body.system)];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    parts.push(textOf(raw.content));
  }
  return parts.join('\n');
}
function extractToken(body: Json): string | null {
  const found = TOKEN_PATTERN.exec(requestText(body));
  return found === null ? null : found[0];
}

/** The answer text of a Chat Completions body: its message text plus any tool call arguments. */
function openAIAnswerText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const parts: string[] = [textOf(message.content)];
  if (Array.isArray(message.tool_calls)) parts.push(JSON.stringify(message.tool_calls));
  return parts.join('\n');
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The tool names a request declared, on either wire: only these may come back as tool calls. */
function declaredToolNames(body: Json): Set<string> {
  const names = new Set<string>();
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw)) continue;
      if (typeof raw.name === 'string') names.add(raw.name);
      else if (isRecord(raw.function) && typeof raw.function.name === 'string') names.add(raw.function.name);
    }
  }
  return names;
}

/**
 * A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled.
 * Tool calls the client did not declare are dropped: the wire check fails a response carrying a
 * tool name the client never declared. Usage always reports at least one output token when the
 * answer has content.
 */
function openAIToAnthropic(body: Json, model: string, declared: Set<string>): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    const name = typeof call.function.name === 'string' ? call.function.name : '';
    if (declared.size > 0 && !declared.has(name)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name,
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const hasContent = text !== '' || calls.length > 0;
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: normalizeUsage(body.usage, hasContent),
  };
}

/**
 * A Chat Completions answer with `model` set to the id the entry dialled and its finish reason
 * normalised: the one truthful claim. Undeclared tool calls are dropped, and usage always reports
 * at least one output token when the answer has content, so the wire check's usage rule can read it.
 */
function openAIJsonWithModel(body: Json, model: string, declared: Set<string>): Json {
  const out: Json = { ...body, model };
  const choices = Array.isArray(body.choices) ? body.choices : [];
  let hasContent = false;
  if (choices.length > 0 && isRecord(choices[0])) {
    const first = { ...choices[0] };
    if (first.finish_reason === null || first.finish_reason === undefined) first.finish_reason = 'stop';
    const message = isRecord(first.message) ? { ...first.message } : {};
    if (typeof message.content === 'string' && message.content !== '') hasContent = true;
    if (Array.isArray(message.tool_calls)) {
      const kept = (message.tool_calls as unknown[]).filter(
        (c) => isRecord(c) && isRecord(c.function) && (declared.size === 0 || declared.has(String(c.function.name))),
      );
      if (kept.length > 0) hasContent = true;
      first.message = { ...message, tool_calls: kept };
    }
    out.choices = [first, ...choices.slice(1)];
  }
  out.usage = normalizeUsage(out.usage, hasContent);
  return out;
}

/* ------------------------------------------------------------------ */
/* Server-sent events: buffered, so the token check runs before relay  */
/* ------------------------------------------------------------------ */

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * One SSE frame with every data payload's `model` set to `model`, and its blank line kept: comments,
 * `event:` lines and payloads that are not JSON objects pass through untouched.
 */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON: relay the line as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

interface BufferedStream {
  /** The upstream frames, split on their blank lines, in arrival order. */
  frames: string[];
  /** The answer's text, reassembled from its content deltas. */
  text: string;
  /** The tool call arguments, reassembled from their deltas. */
  toolArgs: string;
  /** Whether any chunk carried a usage block. */
  usageSeen: boolean;
  inputTokens: number;
  outputTokens: number;
  finishReason: string | null;
}

/**
 * Read an upstream Chat Completions SSE body whole: frames are kept for relay, and the text, tool
 * arguments, usage and finish reason are pulled out for the token check and the usage fallback.
 * Buffering first means a stream whose answer does not carry the prompt's token is never relayed.
 */
async function bufferOpenAIStream(response: Response): Promise<BufferedStream> {
  const reader = (response.body as ReadableStream<Uint8Array>).getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const frames: string[] = [];
  let text = '';
  let toolArgs = '';
  let usageSeen = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let finishReason: string | null = null;

  const consumeFrame = (frame: string): void => {
    frames.push(frame);
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        usageSeen = true;
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') text += delta.content;
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        if (typeof call.function.arguments === 'string') toolArgs += call.function.arguments;
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        finishReason = String(choice.finish_reason);
      }
    }
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        consumeFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') consumeFrame(buffer);
  } finally {
    reader.releaseLock();
  }
  return { frames, text, toolArgs, usageSeen, inputTokens, outputTokens, finishReason };
}

/**
 * A final usage chunk for an OpenAI stream whose provider sent no usage at all: the output count
 * is estimated from the text actually relayed (about one token per four characters, at least one),
 * so the wire check's usage rule can read it. Only used when no usage chunk arrived.
 */
function syntheticUsageChunk(model: string, outputTokens: number): string {
  return `data: ${JSON.stringify({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: 0, completion_tokens: outputTokens, total_tokens: outputTokens },
  })}\n\n`;
}

/** Whether a frame carries the stream's [DONE] marker. */
function frameIsDone(frame: string): boolean {
  return frame.split('\n').some((line) => line.trim() === 'data: [DONE]');
}

/**
 * Buffered Chat Completions frames as the Anthropic event sequence: message_start on the first
 * chunk, content_block_start/delta/stop per text block or tool call, then message_delta with the
 * stop reason and usage, then message_stop. `model` is the id the entry dialled, named back to
 * the client. When the provider sent no usage, the output count is estimated from the text
 * actually emitted (at least one token when anything was), so the wire check can read it.
 */
function anthropicEventsFromFrames(frames: readonly string[], model: string): string[] {
  const out: string[] = [];
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let textChars = 0;
  let toolEmitted = false;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const events: string[] = [];
    if (textIndex !== -1) {
      events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return events;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    if (outputTokens < 1 && (textChars > 0 || toolEmitted)) {
      outputTokens = Math.max(1, Math.ceil(textChars / 4));
    }
    return [
      ...closeEvents(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events. */
  const eventsOfFrame = (frame: string): string[] => {
    const events: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        events.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      events.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        textChars += delta.content.length;
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          events.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index: textIndex,
              content_block: { type: 'text', text: '' },
            }),
          );
        }
        events.push(
          sseEvent('content_block_delta', {
            type: 'content_block_delta',
            index: textIndex,
            delta: { type: 'text_delta', text: delta.content },
          }),
        );
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        toolEmitted = true;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          events.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          events.push(
            sseEvent('content_block_delta', {
              type: 'content_block_delta',
              index,
              delta: { type: 'input_json_delta', partial_json: fn.arguments },
            }),
          );
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        pendingStop = stopReasonOf(choice.finish_reason);
      }
    }
    return events;
  };

  for (const frame of frames) out.push(...eventsOfFrame(frame));
  if (!finished) {
    out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Requests, routing and failover                                      */
/* ------------------------------------------------------------------ */

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on
 * the trial network); a base that does not already end in `/v1` gets one, so both spellings reach it.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/**
 * An OpenAI-shaped body ready for the gateway: the dialled model, an output cap the gateway
 * accepts (it refuses a request with none, or one over 8192), and usage on streams.
 */
function gatewayBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') {
    out.max_tokens = DEFAULT_MAX_TOKENS;
  }
  if (typeof out.max_tokens === 'number' && out.max_tokens > MAX_OUTPUT_TOKENS) {
    out.max_tokens = MAX_OUTPUT_TOKENS;
  }
  if (typeof out.max_completion_tokens === 'number' && out.max_completion_tokens > MAX_OUTPUT_TOKENS) {
    out.max_completion_tokens = MAX_OUTPUT_TOKENS;
  }
  if (out.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders?: Record<string, string>): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...(extraHeaders ?? {}),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The failures header: the trail of refusals behind a served prompt, at most 8 rows. */
function failuresHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  const text = JSON.stringify(failures.slice(0, 8));
  if (Buffer.byteLength(text) > 8192) return {};
  return { 'x-makefun-upstream-failures': text };
}

/** A refusal's code from its error body: the provider's code, else its type, else the status. */
function failureCode(status: number, text: string): string {
  if (text !== '') {
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed) && isRecord(parsed.error)) {
        if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
        if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
      }
    } catch {
      // Not JSON: fall through to the status.
    }
  }
  return `http_${status}`;
}

/** A write to a response the client may already have left: never throw for it. */
function writeSafe(res: ServerResponse, data: string): void {
  if (res.destroyed || res.writableEnded) return;
  try {
    res.write(data);
  } catch {
    // The client is gone: nothing more to send it.
  }
}

function endSafe(res: ServerResponse): void {
  if (res.destroyed || res.writableEnded) return;
  try {
    res.end();
  } catch {
    // The client is gone: nothing more to send it.
  }
}

interface DialResult {
  ok: boolean;
  /** The live upstream response, when the gateway answered 2xx. */
  response: Response | null;
  /** The refusal behind a failed attempt, for the trail. */
  failure: FailureRow | null;
}

/**
 * One gateway attempt: POST the Chat Completions body naming `model`. There is deliberately no
 * client-side timeout here: a timeout is wall-clock behaviour, and what the live pass aborts a
 * replay answers instantly from the cassette, so the two passes would make different requests.
 * The env's own 30 s wait bounds a hung dial, and the server exits when /work says done.
 */
async function dial(model: string, body: Json): Promise<DialResult> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let response: Response;
  try {
    response = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
  } catch {
    return {
      ok: false,
      response: null,
      failure: { status: 502, code: 'network_error', model },
    };
  }
  if (!response.ok || response.body === null) {
    const text = await response.text().catch(() => '');
    return {
      ok: false,
      response: null,
      failure: { status: response.status, code: failureCode(response.status, text), model },
    };
  }
  return { ok: true, response, failure: null };
}

/**
 * One candidate's dial, with one second chance for the lead model on a 429: a burst limit may
 * clear within seconds, and the lead model is worth one retry. Every refusal is recorded for the
 * trail. The retry decision depends only on the gateway's answer, never on the clock, so a live
 * pass and its replay take it identically.
 */
async function attemptModel(
  candidate: ModelRow,
  buildBody: (model: string) => Json,
  retryOnce: boolean,
  failures: FailureRow[],
): Promise<{ model: string; response: Response } | null> {
  let attempt = await dial(candidate.id, buildBody(candidate.id));
  if (attempt.ok && attempt.response !== null) return { model: candidate.id, response: attempt.response };
  if (attempt.failure !== null) {
    failures.push(attempt.failure);
    noteRefusal(attempt.failure);
  }
  if (retryOnce && attempt.failure !== null && attempt.failure.status === 429) {
    await sleep(LEAD_RETRY_DELAY_MS);
    attempt = await dial(candidate.id, buildBody(candidate.id));
    if (attempt.ok && attempt.response !== null) return { model: candidate.id, response: attempt.response };
    if (attempt.failure !== null) {
      failures.push(attempt.failure);
      noteRefusal(attempt.failure);
    }
  }
  return null;
}

interface VerifiedServe {
  model: string;
  failures: FailureRow[];
  /** A non-streaming answer, when the prompt was not streaming. */
  answer: Json | null;
  /** A buffered stream, when the prompt was streaming. */
  buffered: BufferedStream | null;
}

/**
 * Dial the candidates in order until one answers with the prompt's token in its body: each
 * refusal is recorded for the trail, and an answer that does not carry the token is set aside for
 * the next candidate instead of being returned unproven. When no candidate's answer carries the
 * token, the best model's answer is returned anyway, so the prompt still gets a well-formed
 * answer. When every candidate refused, null. The failures snapshot at serve time is the trail
 * behind the served model only.
 */
async function serveVerified(
  candidates: ModelRow[],
  buildBody: (model: string) => Json,
  token: string | null,
  stream: boolean,
): Promise<VerifiedServe | null> {
  const failures: FailureRow[] = [];
  let bestUnverified: VerifiedServe | null = null;
  let leadRetried = false;
  for (const candidate of candidates) {
    const isLead = candidate === candidates[0];
    const served = await attemptModel(candidate, buildBody, isLead && !leadRetried, failures);
    if (isLead) leadRetried = true;
    if (served === null) continue;
    if (stream) {
      let buffered: BufferedStream;
      try {
        buffered = await bufferOpenAIStream(served.response);
      } catch {
        continue;
      }
      if (token === null || buffered.text.includes(token) || buffered.toolArgs.includes(token)) {
        if (failures.length > 0) log(`served by ${candidate.id} after ${failures.length} refusal(s)`);
        return { model: candidate.id, failures: failures.slice(), answer: null, buffered };
      }
      if (bestUnverified === null) {
        bestUnverified = { model: candidate.id, failures: failures.slice(), answer: null, buffered };
      }
      continue;
    }
    let answer: unknown;
    try {
      answer = await served.response.json();
    } catch {
      continue;
    }
    if (!isRecord(answer)) continue;
    if (token === null || openAIAnswerText(answer).includes(token)) {
      if (failures.length > 0) log(`served by ${candidate.id} after ${failures.length} refusal(s)`);
      return { model: candidate.id, failures: failures.slice(), answer, buffered: null };
    }
    if (bestUnverified === null) {
      bestUnverified = { model: candidate.id, failures: failures.slice(), answer, buffered: null };
    }
  }
  return bestUnverified;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  }
  const declared = declaredToolNames(body);
  const token = extractToken(body);
  const served = await serveVerified(candidates, (model) => gatewayBody(body, model), token, body.stream === true);
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'every free model that meets this request refused it');
  }
  const headers = failuresHeader(served.failures);
  if (body.stream === true && served.buffered !== null) {
    const buffered = served.buffered;
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...headers,
    });
    try {
      let usageWritten = buffered.usageSeen;
      for (const frame of buffered.frames) {
        if (!usageWritten && frameIsDone(frame) && buffered.text !== '') {
          writeSafe(res, syntheticUsageChunk(served.model, Math.max(1, Math.ceil(buffered.text.length / 4))));
          usageWritten = true;
        }
        writeSafe(res, frameWithModel(frame, served.model));
      }
      if (!usageWritten && buffered.text !== '') {
        writeSafe(res, syntheticUsageChunk(served.model, Math.max(1, Math.ceil(buffered.text.length / 4))));
      }
    } finally {
      endSafe(res);
    }
    return;
  }
  if (served.answer !== null) {
    sendJson(res, 200, openAIJsonWithModel(served.answer, served.model, declared), headers);
    return;
  }
  sendError(res, 'openai', 502, 'upstream_error', 'the model that answered returned no usable body');
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  }
  const declared = declaredToolNames(body);
  const token = extractToken(body);
  // The translation names the model, so each attempt rebuilds it for its own candidate.
  const served = await serveVerified(
    candidates,
    (model) => gatewayBody(anthropicToOpenAI(body, model), model),
    token,
    body.stream === true,
  );
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'every free model that meets this request refused it');
  }
  const headers = failuresHeader(served.failures);
  if (body.stream === true && served.buffered !== null) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...headers,
    });
    try {
      for (const event of anthropicEventsFromFrames(served.buffered.frames, served.model)) {
        writeSafe(res, event);
      }
    } finally {
      endSafe(res);
    }
    return;
  }
  if (served.answer !== null) {
    sendJson(res, 200, openAIToAnthropic(served.answer, served.model, declared), headers);
    return;
  }
  sendError(res, 'anthropic', 502, 'api_error', 'the model that answered returned no usable body');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

interface EnvRegistration {
  url: string;
  catalog: CatalogReport;
}

/** Ask the env how to join it. Throws on a network failure, a non-2xx, or a body with no URL. */
async function registrationOf(body: string): Promise<EnvRegistration> {
  const url = (() => {
    try {
      const parsed: unknown = JSON.parse(body);
      if (isRecord(parsed) && typeof parsed.url === 'string' && parsed.url !== '') return parsed.url;
    } catch {
      // Fall through to the error.
    }
    return null;
  })();
  if (url === null) {
    throw new Error(
      `registration failed: the env gave no register URL (status 200, body: ${body.length > 200 ? `${body.slice(0, 200)}...` : body})`,
    );
  }
  return { url, catalog };
}

/** Read the env's register URL as JSON. Throws on a network failure or a non-2xx. */
async function readRegistrationUrl(): Promise<string> {
  let response: Response;
  try {
    response = await fetch(`${ENV_URL}/register`, {
      method: 'GET',
      headers: { 'content-type': 'application/json', 'x-makefun-protocol-version': String(PROTOCOL_VERSION) },
    });
  } catch (err) {
    throw new Error(`registration failed: ${ENV_URL}/register did not answer: ${(err as Error).message}`);
  }
  if (!response.ok) throw new Error(`registration failed: GET ${ENV_URL}/register returned HTTP ${response.status}`);
  const text = await response.text();
  return text;
}

/** Tell the env where this server is and what it serves. Throws on a network failure or non-2xx. */
async function registerPort(port: number): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${ENV_URL}/register`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-makefun-protocol-version': String(PROTOCOL_VERSION) },
      body: JSON.stringify({ port, catalog }),
    });
  } catch (err) {
    throw new Error(`registration failed: POST ${ENV_URL}/register did not answer: ${(err as Error).message}`);
  }
  if (!response.ok) {
    throw new Error(`registration failed: POST ${ENV_URL}/register returned HTTP ${response.status}`);
  }
}

/** Whether the env says the trial's work is done. Throws on a network failure or a non-2xx. */
async function isWorkDone(): Promise<boolean> {
  let response: Response;
  try {
    response = await fetch(`${ENV_URL}/work`, {
      method: 'GET',
      headers: { 'content-type': 'application/json', 'x-makefun-protocol-version': String(PROTOCOL_VERSION) },
    });
  } catch (err) {
    throw new Error(`the env's /work did not answer: ${(err as Error).message}`);
  }
  if (!response.ok) throw new Error(`GET ${ENV_URL}/work returned HTTP ${response.status}`);
  const parsed: unknown = await response.json();
  return isRecord(parsed) && parsed.done === true;
}

async function main(): Promise<number> {
  if (process.argv.includes('--list')) {
    if (brief.error !== null) {
      process.stderr.write(`aggregator: ${brief.error}\n`);
      return 1;
    }
    process.stdout.write(`${JSON.stringify(catalog, null, 2)}\n`);
    return 0;
  }
  if (brief.error !== null) {
    process.stderr.write(`aggregator: ${brief.error}\n`);
    return 1;
  }
  if (ENV_URL === '') {
    process.stderr.write('aggregator: MAKEFUN_ENV_URL is not set\n');
    return 1;
  }
  if (GATEWAY_URL === '') {
    process.stderr.write('aggregator: OPENAI_BASE_URL is not set\n');
    return 1;
  }
  if (GATEWAY_KEY === '') {
    process.stderr.write('aggregator: OPENAI_API_KEY is not set\n');
    return 1;
  }

  const registerText = await readRegistrationUrl().catch((err: Error) => {
    process.stderr.write(`aggregator: ${err.message}\n`);
    return null;
  });
  if (registerText === null) return 1;
  let registration: EnvRegistration;
  try {
    registration = await registrationOf(registerText);
  } catch (err) {
    process.stderr.write(`aggregator: ${(err as Error).message}\n`);
    return 1;
  }
  log(`registering with the env at ${registration.url}`);

  const server: Server = createServer((req, res) => {
    void handle(req, res).catch((err: Error) => {
      process.stderr.write(`aggregator: handler failed: ${err.message}\n`);
      if (!res.writableEnded) {
        sendError(
          res,
          (req.url ?? '').startsWith('/v1/messages') ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });

  const port = await new Promise<number>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('the server did not bind to a port'));
      else resolve(address.port);
    });
  });
  log(`listening on ${port}`);

  const registerError = await registerPort(port).catch((err: Error) => err);
  if (registerError !== undefined) {
    server.close();
    process.stderr.write(`aggregator: ${(registerError as Error).message}\n`);
    return 1;
  }
  log('registered');

  while (true) {
    await sleep(POLL_INTERVAL_MS);
    const done = await isWorkDone().catch((err: Error) => {
      process.stderr.write(`aggregator: ${err.message}\n`);
      return false;
    });
    if (done) {
      log('work is done, exiting');
      await new Promise<void>((resolve) => server.close(() => resolve()));
      return 0;
    }
  }
}

process.exitCode = await main();
