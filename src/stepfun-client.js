const DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";
const DEFAULT_OPENROUTER_MODEL = "openrouter/free";
const DEFAULT_STEPFUN_BASE_URL = "https://api.stepfun.com/step_plan/v1";
const DEFAULT_STEPFUN_MODEL = "step-3.7-flash";
const DEFAULT_TTS_MODEL = "stepaudio-2.5-tts";
const JSON_RETRY_PROMPT = [
  "上一次输出为空、被截断、不是有效 JSON，或字段结构不符合要求。",
  "请重新作答，只输出一个完整的 JSON 对象。",
  "不要输出推理过程、Markdown 代码块或任何解释，并严格遵守前文要求的字段与类型。"
].join("");
let keyCursor = 0;

function withTimeout(timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort("timeout"), timeoutMs);
  return {
    signal: controller.signal,
    clear: () => clearTimeout(timer)
  };
}

function extractJson(text) {
  const source = String(text || "").trim();
  if (!source) throw new Error("StepFun 返回了空内容");
  try {
    return JSON.parse(source);
  } catch {
    const fenced = source.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fenced) return JSON.parse(fenced[1]);
    const start = Math.min(
      ...["{", "["].map((mark) => {
        const index = source.indexOf(mark);
        return index < 0 ? Number.POSITIVE_INFINITY : index;
      })
    );
    if (!Number.isFinite(start)) throw new Error("StepFun 返回内容不是 JSON");
    const endObject = source.lastIndexOf("}");
    const endArray = source.lastIndexOf("]");
    const end = Math.max(endObject, endArray);
    if (end <= start) throw new Error("StepFun 返回 JSON 不完整");
    return JSON.parse(source.slice(start, end + 1));
  }
}

function isTruncated(choice) {
  const reason = String(choice?.finish_reason || "").trim().toLowerCase();
  return reason === "length" || reason === "max_tokens" || reason === "max_token";
}

async function validateJsonResult(result, options) {
  const requiredKeys = Array.isArray(options.requiredKeys) ? options.requiredKeys : [];
  const missingKeys = requiredKeys.filter((key) => !Object.prototype.hasOwnProperty.call(result, key));
  if (missingKeys.length) {
    throw new Error(`StepFun JSON 缺少字段: ${missingKeys.join(", ")}`);
  }

  const validator = options.validateJson || options.validate;
  if (typeof validator !== "function") return;
  const verdict = await validator(result);
  if (verdict === false) throw new Error("StepFun JSON 结构校验失败");
  if (typeof verdict === "string" && verdict.trim()) {
    throw new Error(`StepFun JSON 结构校验失败: ${verdict.trim()}`);
  }
}

export class StepFunClient {
  constructor(env = {}, options = {}) {
    const requestedProvider = String(env.AI_PROVIDER || env.TEXT_LLM_PROVIDER || "").trim().toLowerCase();
    const openRouterKeys = env.OPENROUTER_API_KEYS || env.OPENROUTER_API_KEY || "";
    const stepFunKeys = env.STEPFUN_API_KEYS || env.STEPFUN_API_KEY || env.step_API_KEY || "";
    // Production sets AI_PROVIDER=openrouter. The legacy StepFun path remains
    // available for local tests and older deployments, but is never selected
    // when OpenRouter is explicitly required.
    this.provider = requestedProvider || (String(openRouterKeys).trim() ? "openrouter" : "stepfun");
    const rawKeys = this.provider === "openrouter" ? openRouterKeys : stepFunKeys;
    this.apiKeys = String(rawKeys).split(/[\s,;]+/).map((item) => item.trim()).filter(Boolean);
    this.baseUrl = this.provider === "openrouter"
      ? String(env.OPENROUTER_BASE_URL || DEFAULT_OPENROUTER_BASE_URL).replace(/\/+$/, "")
      : String(env.STEPFUN_BASE_URL || DEFAULT_STEPFUN_BASE_URL).replace(/\/+$/, "");
    this.textModel = this.provider === "openrouter"
      ? env.OPENROUTER_MODEL || DEFAULT_OPENROUTER_MODEL
      : env.STEPFUN_MODEL || DEFAULT_STEPFUN_MODEL;
    this.ttsModel = env.STEPFUN_TTS_MODEL || DEFAULT_TTS_MODEL;
    this.siteUrl = String(env.OPENROUTER_SITE_URL || env.APP_ORIGIN || "").trim();
    this.appName = String(env.OPENROUTER_APP_NAME || "ShopValidator").trim();
    // Cloudflare's native fetch is an invocation-sensitive host function.
    // Keeping a bare reference and later calling `this.fetch(...)` changes its
    // receiver and throws "Illegal invocation"; a wrapper preserves a direct
    // host call while still allowing tests to inject their own fetch.
    this.fetch = options.fetch
      ? (...args) => options.fetch(...args)
      : (...args) => globalThis.fetch(...args);
  }

  get configured() {
    return this.apiKeys.length > 0;
  }

  nextKey() {
    if (!this.apiKeys.length) return "";
    const key = this.apiKeys[keyCursor % this.apiKeys.length];
    keyCursor += 1;
    return key;
  }

  async chatJson(messages, options = {}) {
    if (!this.configured) throw new Error(`${this.provider === "openrouter" ? "OPENROUTER_API_KEY" : "STEPFUN_API_KEY"} 未配置`);
    const baseMaxTokens = options.maxTokens || 1800;
    let lastJsonError = null;

    for (let attempt = 0; attempt < 2; attempt += 1) {
      const timeout = withTimeout(options.timeoutMs || 25_000);
      try {
        const requestMessages = attempt === 0
          ? messages
          : [...messages, { role: "user", content: JSON_RETRY_PROMPT }];
        const headers = {
          "Authorization": `Bearer ${this.nextKey()}`,
          "Content-Type": "application/json",
          "Accept": "application/json"
        };
        if (this.provider === "openrouter") {
          if (this.siteUrl) headers["HTTP-Referer"] = this.siteUrl;
          if (this.appName) headers["X-Title"] = this.appName;
        }
        const response = await this.fetch(`${this.baseUrl}/chat/completions`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: this.textModel,
            messages: requestMessages,
            temperature: options.temperature ?? 0.65,
            max_tokens: attempt === 0 ? baseMaxTokens : Math.min(Math.max(baseMaxTokens * 2, 600), 4000),
            response_format: { type: "json_object" }
          }),
          signal: timeout.signal
        });
        if (!response.ok) {
          const detail = (await response.text()).slice(0, 400);
          throw new Error(`${this.provider} 文本请求失败 ${response.status}: ${detail}`);
        }

        const payload = await response.json();
        const choice = payload?.choices?.[0] || {};
        const message = choice.message || {};
        try {
          if (isTruncated(choice)) throw new Error(`${this.provider} JSON 因 token 上限被截断`);
          // reasoning_content 可能包含未完成的思考，不能作为最终结构化结果。
          const result = extractJson(message.content);
          await validateJsonResult(result, options);
          return result;
        } catch (error) {
          lastJsonError = error instanceof Error ? error : new Error(String(error));
          if (attempt === 0) continue;
        }
      } finally {
        timeout.clear();
      }
    }

    throw new Error(`${this.provider} 连续两次未返回可用 JSON: ${lastJsonError?.message || "未知结构错误"}`);
  }

  async tts(input, options = {}) {
    if (this.provider === "openrouter") throw new Error("OpenRouter 文本端点不提供 TTS，请使用独立语音服务");
    if (!this.configured) throw new Error("STEPFUN_API_KEY 未配置");
    const text = String(input || "").trim();
    if (!text || text.length > 500) throw new Error("TTS 文本长度必须为 1—500 字");
    const timeout = withTimeout(options.timeoutMs || 20_000);
    try {
      const response = await this.fetch(`${this.baseUrl}/audio/speech`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.nextKey()}`,
          "Content-Type": "application/json",
          "Accept": "audio/mpeg"
        },
        body: JSON.stringify({
          model: this.ttsModel,
          input: text,
          voice: options.voice || "cixingnansheng",
          response_format: "mp3",
          speed: options.speed || 1
        }),
        signal: timeout.signal
      });
      if (!response.ok) {
        const detail = (await response.text()).slice(0, 400);
        throw new Error(`StepFun TTS 请求失败 ${response.status}: ${detail}`);
      }
      return response;
    } finally {
      timeout.clear();
    }
  }
}

export const stepFunDefaults = {
  baseUrl: DEFAULT_STEPFUN_BASE_URL,
  textModel: DEFAULT_STEPFUN_MODEL,
  ttsModel: DEFAULT_TTS_MODEL
};

export const openRouterDefaults = {
  baseUrl: DEFAULT_OPENROUTER_BASE_URL,
  textModel: DEFAULT_OPENROUTER_MODEL
};
