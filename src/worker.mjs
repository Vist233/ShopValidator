import { StepFunClient } from "./stepfun-client.js";
import { DashScopeAsrClient } from "./dashscope-asr-client.js";
import { DashScopeTtsClient } from "./dashscope-tts-client.js";
import {
  createSearchState,
  finalizeAgentSearch,
  runAgentRound,
  runAgentSearch
} from "./agent-orchestrator.js";
import {
  computeServerDecision,
  evaluateInterviewCompleteness,
  sanitizeAgentNextQuestion,
  normalizeServerFacts,
  getAllowedInterviewFields,
  INTERVIEW_LIMITS
} from "./server-decision-adapter.mjs";

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff"
  }
});

const caseStore = new Map();
const runStore = new Map();
const publicCaseStore = new Map();
const rateBuckets = new Map();
const localAsrSessions = new Map();
const localTurnLocks = new Set();
const CASE_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_JSON_BYTES = 256 * 1024;
const MAX_TTS_PER_CASE = 40;
const QUEUE_CLAIM_TTL_MS = 10 * 60 * 1000;
const QUEUE_MAX_RETRIES = 3;
const MAX_ASR_SESSIONS_PER_CASE = 3;
const MAX_ASR_SESSION_MS = 20 * 60 * 1000;
const MAX_ASR_AUDIO_BYTES = 40 * 1024 * 1024;
const MAX_ASR_CLIP_BYTES = 3 * 1024 * 1024;
const ASR_SILENCE_DURATION_MS = 350;
const SEARCH_TARGET = 2;
const SEARCH_ROUNDS = 1;
const SEARCH_CONCURRENCY = 1;
const ACTION_RATE_LIMITS = {
  "create-case": { limit: 12, windowMs: 10 * 60 * 1000 },
  map: { limit: 60, windowMs: 60 * 1000 },
  tts: { limit: 60, windowMs: 60 * 1000 },
  turn: { limit: 40, windowMs: 60 * 1000 },
  analyze: { limit: 5, windowMs: 10 * 60 * 1000 },
  "asr-transcribe": { limit: 40, windowMs: 60 * 60 * 1000 }
};

export class AgentGate {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.active = 0;
    this.pending = [];
    this.rateFallback = new Map();
    this.rateTail = Promise.resolve();
  }

  async fetch(request) {
    if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
    const payload = await request.json();
    const path = new URL(request.url).pathname;
    if (path === "/rate") {
      const result = await this.checkRate(payload);
      return Response.json(result);
    }
    if (path === "/asr/acquire") {
      return Response.json(await this.acquireAsr(payload));
    }
    if (path === "/asr/release") {
      return Response.json(await this.releaseAsr(payload));
    }
    return new Promise((resolve) => {
      this.pending.push({ payload, resolve });
      this.drain();
    });
  }

  serialize(operation) {
    const result = this.rateTail.then(operation, operation);
    this.rateTail = result.then(() => undefined, () => undefined);
    return result;
  }

  async checkRate(payload) {
    const action = String(payload?.action || "unknown").replace(/[^a-z0-9_-]/gi, "").slice(0, 40);
    const key = String(payload?.key || "unknown").replace(/[^a-f0-9]/gi, "").slice(0, 80);
    const limit = Math.max(1, Math.min(Number(payload?.limit) || 1, 10_000));
    const windowMs = Math.max(1000, Math.min(Number(payload?.windowMs) || 60_000, 24 * 60 * 60 * 1000));
    const storageKey = `rate:${action}:${key}`;
    const execute = async () => {
      const now = Date.now();
      const storage = this.state?.storage;
      const current = storage?.get
        ? await storage.get(storageKey)
        : this.rateFallback.get(storageKey);
      const bucket = !current || Number(current.resetAt) <= now
        ? { count: 0, resetAt: now + windowMs }
        : current;
      bucket.count += 1;
      if (storage?.put) await storage.put(storageKey, bucket);
      else this.rateFallback.set(storageKey, bucket);
      return {
        allowed: bucket.count <= limit,
        remaining: Math.max(0, limit - bucket.count),
        retryAfterMs: Math.max(0, bucket.resetAt - now)
      };
    };
    // DO storage is persistent; this promise chain also makes the read/modify/
    // write operation atomic within a live AgentGate instance.
    return this.serialize(execute);
  }

  async acquireAsr(payload) {
    const key = String(payload?.key || "").replace(/[^a-f0-9]/gi, "").slice(0, 80);
    const sessionId = cleanId(payload?.sessionId, 100);
    if (!key || !sessionId) return { allowed: false, reason: "invalid" };
    return this.serialize(async () => {
      const now = Date.now();
      const storageKey = `asr:${key}`;
      const storage = this.state?.storage;
      const existing = storage?.get
        ? await storage.get(storageKey)
        : this.rateFallback.get(storageKey);
      const state = !existing || Number(existing.resetAt) <= now
        ? {
            resetAt: now + CASE_TTL_MS,
            sessions: 0,
            reservedBytes: 0,
            reservedDurationMs: 0,
            activeSessionId: "",
            activeExpiresAt: 0
          }
        : existing;
      if (state.activeSessionId && Number(state.activeExpiresAt) > now) {
        return {
          allowed: false,
          reason: "active",
          retryAfterMs: Number(state.activeExpiresAt) - now
        };
      }
      if (Number(state.sessions) >= MAX_ASR_SESSIONS_PER_CASE) {
        return { allowed: false, reason: "session-quota", retryAfterMs: state.resetAt - now };
      }
      state.sessions = Number(state.sessions) + 1;
      // Reserve the full bounded session before opening the paid upstream. A
      // crashed relay therefore cannot evade the persistent byte/time quota.
      state.reservedBytes = Number(state.reservedBytes) + MAX_ASR_AUDIO_BYTES;
      state.reservedDurationMs = Number(state.reservedDurationMs) + MAX_ASR_SESSION_MS;
      state.activeSessionId = sessionId;
      state.activeExpiresAt = now + MAX_ASR_SESSION_MS;
      if (storage?.put) await storage.put(storageKey, state);
      else this.rateFallback.set(storageKey, state);
      return {
        allowed: true,
        sessionId,
        sessionsRemaining: MAX_ASR_SESSIONS_PER_CASE - state.sessions,
        maxBytes: MAX_ASR_AUDIO_BYTES,
        maxDurationMs: MAX_ASR_SESSION_MS
      };
    });
  }

  async releaseAsr(payload) {
    const key = String(payload?.key || "").replace(/[^a-f0-9]/gi, "").slice(0, 80);
    const sessionId = cleanId(payload?.sessionId, 100);
    if (!key || !sessionId) return { released: false };
    return this.serialize(async () => {
      const storageKey = `asr:${key}`;
      const storage = this.state?.storage;
      const state = storage?.get
        ? await storage.get(storageKey)
        : this.rateFallback.get(storageKey);
      if (!state || state.activeSessionId !== sessionId) return { released: false };
      state.activeSessionId = "";
      state.activeExpiresAt = 0;
      if (storage?.put) await storage.put(storageKey, state);
      else this.rateFallback.set(storageKey, state);
      return { released: true };
    });
  }

  drain() {
    while (this.active < 5 && this.pending.length) {
      const job = this.pending.shift();
      this.active += 1;
      const client = new StepFunClient(this.env);
      client.chatJson(job.payload.messages, job.payload.options || {})
        .then((result) => job.resolve(Response.json({ result })))
        .catch((error) => job.resolve(Response.json({
          error: error instanceof Error ? error.message : "StepFun调用失败"
        }, { status: 502 })))
        .finally(() => {
          this.active -= 1;
          this.drain();
        });
    }
  }
}

function createTextLlm(env) {
  const client = new StepFunClient(env);
  if (!client.configured) return null;
  if (!env.AGENT_GATE) return client.chatJson.bind(client);
  const id = env.AGENT_GATE.idFromName("global-stepfun-gate");
  const gate = env.AGENT_GATE.get(id);
  return async (messages, options = {}) => {
    const response = await gate.fetch("https://agent-gate.internal/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, options })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "AgentGate调用失败");
    return payload.result;
  };
}

const cleanText = (value, maxLength = 40) => String(value || "")
  .replace(/[^\p{L}\p{N}\s·\-]/gu, "")
  .trim()
  .slice(0, maxLength);

const trimText = (value, maxLength = 120) => String(value || "")
  .trim()
  .slice(0, maxLength);

const cleanId = (value, maxLength = 100) => String(value || "")
  .replace(/[^a-zA-Z0-9_-]/g, "")
  .slice(0, maxLength);

function validCoordinate(lat, lng) {
  return Number.isFinite(lat) && Number.isFinite(lng) &&
    lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180;
}

function administrativeAreaOnly(address) {
  const compact = address.replace(/\s+/g, "");
  return /^(?:[\p{Script=Han}]{2,12}(?:省|自治区|特别行政区))?(?:[\p{Script=Han}]{2,12}市)?(?:[\p{Script=Han}]{1,12}(?:区|县|旗))?$/u.test(compact);
}

function tencentMapKeys(env) {
  // Keep the primary key first, then retry the secondary key only when Tencent
  // rejects or cannot serve a request.  The keys remain Worker secrets: no map
  // request ever exposes either value to the browser.  A map key is an opaque
  // credential, not user-facing text: trim it, but never run it through the
  // Chinese-text sanitizer (which could silently remove credential symbols).
  return [...new Set([
    trimText(env?.TENCENT_MAP_KEY, 160),
    trimText(env?.TENCENT_MAP_KEY_SECONDARY, 160)
  ].filter(Boolean))];
}

function configured(env) {
  return tencentMapKeys(env).length > 0;
}

function mapNotConfigured() {
  return json({
    code: "MAP_NOT_CONFIGURED",
    message: "腾讯地图密钥尚未配置"
  }, 503);
}

async function tencentRequest(path, params, keys) {
  const candidates = Array.isArray(keys) ? keys : [keys];
  let lastError = null;
  for (const key of candidates.filter(Boolean)) {
    try {
      const url = new URL(path, "https://apis.map.qq.com");
      Object.entries({ ...params, key }).forEach(([name, value]) => {
        url.searchParams.set(name, String(value));
      });
      const response = await fetch(url, {
        headers: {
          "Accept": "application/json",
          "Referer": "https://shopvalidator.zhangyvjing.com/"
        },
        signal: AbortSignal.timeout(7000)
      });
      if (!response.ok) {
        const error = new Error(`腾讯地图请求失败：${response.status}`);
        error.tencentStatus = response.status;
        throw error;
      }
      const data = await response.json();
      if (data.status !== 0) {
        const error = new Error(data.message || "腾讯地图返回异常");
        error.tencentStatus = Number(data.status);
        throw error;
      }
      return data;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("腾讯地图密钥尚未配置");
}

function isTencentQuotaError(error) {
  return Number(error?.tencentStatus) === 121
    || /每日调用量已达到上限/u.test(String(error?.message || ""));
}

function quotaFallbackContext({ lat, lng, mode, category, fallbackAddress, coordinateSystem, reason }) {
  const hasCoordinates = validCoordinate(lat, lng);
  const location = {
    address: trimText(fallbackAddress, 100)
      || (hasCoordinates ? `定位点（${Number(lat).toFixed(6)}，${Number(lng).toFixed(6)}）` : "地址已记录，等待地图恢复"),
    addressResolution: "approximate"
  };
  if (hasCoordinates) {
    location.latitude = Number(Number(lat).toFixed(6));
    location.longitude = Number(Number(lng).toFixed(6));
  }
  return {
    context: {
      source: "腾讯位置服务",
      mode: mode || "gps",
      coordinateSystem: coordinateSystem || "GCJ-02",
      location,
      nearby: {
        keyword: category || "餐饮",
        radiusMeters: 800,
        count: null,
        places: []
      },
      landmarks: [],
      environment: [],
      dataQuality: {
        status: "degraded",
        reason: reason || "腾讯地图部分服务暂时达到调用上限",
        warnings: [reason || "腾讯地图部分服务暂时达到调用上限"],
        requiresOnSiteVerification: true
      }
    }
  };
}

function staticMapQuotaFallback(lat, lng, zoom) {
  const label = `${Number(lat).toFixed(5)}，${Number(lng).toFixed(5)} · 缩放 ${zoom}`;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360" role="img" aria-label="地图额度暂时不可用">
  <rect width="640" height="360" fill="#f3f1ea"/>
  <path d="M0 72H640M0 144H640M0 216H640M0 288H640M128 0V360M256 0V360M384 0V360M512 0V360" stroke="#d4d0c5" stroke-width="1"/>
  <path d="M0 300C90 260 150 290 230 230S380 155 470 185s110 5 170-45" fill="none" stroke="#9a9589" stroke-width="3" stroke-dasharray="8 8"/>
  <circle cx="320" cy="180" r="10" fill="#1c1c1a"/><circle cx="320" cy="180" r="24" fill="none" stroke="#1c1c1a" stroke-width="2"/>
  <rect x="24" y="24" width="300" height="58" fill="#fffdf8" stroke="#1c1c1a"/>
  <text x="40" y="50" font-family="Arial, sans-serif" font-size="16" fill="#1c1c1a">腾讯地图额度暂时不可用</text>
  <text x="40" y="70" font-family="Arial, sans-serif" font-size="12" fill="#5f5b52">已保留选点：${label}</text>
</svg>`;
  return new Response(svg, {
    status: 200,
    headers: {
      "Content-Type": "image/svg+xml; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff"
    }
  });
}

// Environment signal groups scanned in site-report (rich) mode: who lives,
// works and passes through the block. Kept small to bound Tencent request fan-out.
const ENVIRONMENT_GROUPS = [
  { key: "school", label: "学校/大学", keyword: "学校" },
  { key: "office", label: "写字楼/公司", keyword: "写字楼" },
  { key: "residential", label: "住宅小区", keyword: "小区" },
  { key: "transport", label: "地铁/公交", keyword: "地铁站" },
  { key: "retail", label: "商场/超市", keyword: "商场" }
];

async function searchNearbyPlaces(center, category, keys) {
  const primaryKeyword = category && category !== "我不知道" ? category : "餐饮";
  const keywords = [...new Set([primaryKeyword, "餐饮"].filter(Boolean))];
  let lastData = null;
  let lastKeyword = primaryKeyword;

  for (const keyword of keywords) {
    const data = await tencentRequest("/ws/place/v1/search", {
      keyword,
      boundary: `nearby(${center},800,0)`,
      page_size: 20,
      page_index: 1,
      orderby: "_distance",
      output: "json"
    }, keys);
    lastData = data;
    lastKeyword = keyword;
    const places = Array.isArray(data?.data) ? data.data : [];
    if (Number(data?.count) > 0 || places.length > 0 || keyword === keywords.at(-1)) {
      return {
        data: { ...data, data: places },
        keyword,
        fallbackUsed: keyword !== primaryKeyword
      };
    }
  }

  return {
    data: { ...(lastData || {}), data: Array.isArray(lastData?.data) ? lastData.data : [] },
    keyword: lastKeyword,
    fallbackUsed: lastKeyword !== primaryKeyword
  };
}

async function buildContextFromGcj02(lat, lng, category, keys, options = {}) {
  if (!validCoordinate(lat, lng)) throw new Error("腾讯地图坐标无效");
  const center = `${lat},${lng}`;
  const baseKeyword = category && category !== "我不知道" ? category : "餐饮";
  // Reverse geocoding has its own Tencent quota. It must not take down the
  // whole location report when nearby search is still usable.
  const [reverseAttempt, nearbyAttempt] = await Promise.all([
    tencentRequest("/ws/geocoder/v1/", {
      location: center,
      get_poi: 1,
      poi_options: "address_format=short;radius=1000;policy=1",
      output: "json"
    }, keys)
      .then((data) => ({ data, error: null }))
      .catch((error) => ({ data: null, error })),
    searchNearbyPlaces(center, baseKeyword, keys)
      .then((result) => ({ ...result, error: null }))
      .catch((error) => ({ data: null, error }))
  ]);

  const reverse = reverseAttempt.data;
  if (nearbyAttempt.error && !isTencentQuotaError(nearbyAttempt.error)) {
    throw nearbyAttempt.error;
  }
  const nearby = nearbyAttempt.data || { count: null, data: [] };
  const reverseResult = reverse?.result || {};
  const addressComponent = reverseResult.address_component || {};
  const landmarks = (reverseResult.pois || []).slice(0, 10).map((poi) => ({
    title: cleanText(poi.title, 36),
    category: cleanText(poi.category, 50),
    distance: Number(poi._distance) || 0,
    direction: cleanText(poi._dir_desc, 8)
  }));
  const competitors = (nearby.data || []).slice(0, 20).map((poi) => ({
    title: cleanText(poi.title, 36),
    category: cleanText(poi.category, 50),
    distance: Number(poi._distance) || 0
  }));
  const nearestAddress = trimText(nearby.data?.[0]?.address, 100);
  const approximateAddress = trimText(options.fallbackAddress, 100)
    || (nearestAddress ? `附近：${nearestAddress}` : `定位点（${lat.toFixed(6)}，${lng.toFixed(6)}）`);
  const addressResolution = reverseResult.address ? "exact" : "approximate";
  const dataWarnings = [...(options.dataQualityWarnings || [])];
  if (reverseAttempt.error) dataWarnings.push("逆地址解析暂时不可用");
  if (nearbyAttempt.error) dataWarnings.push("周边地点搜索暂时不可用");
  if (nearbyAttempt.fallbackUsed) dataWarnings.push(`未找到“${baseKeyword}”，已用“餐饮”补充搜索`);

  // Site-report mode needs a richer read of the surroundings than a single
  // competitor keyword: who lives / works / passes through here. Each group is
  // an independent nearby search; failures degrade to an empty group.
  let environment = [];
  if (options.rich) {
    const scans = await Promise.all(ENVIRONMENT_GROUPS.map((group) =>
      tencentRequest("/ws/place/v1/search", {
        keyword: group.keyword,
        boundary: `nearby(${center},1000,0)`,
        page_size: 10,
        page_index: 1,
        orderby: "_distance",
        output: "json"
      }, keys).then((res) => ({ group, res })).catch(() => ({ group, res: null }))
    ));
    environment = scans.map(({ group, res }) => {
      const places = (res?.data || []).slice(0, 10);
      return {
        key: group.key,
        label: group.label,
        count: Number(res?.count) || places.length,
        nearestMeters: places.length ? Number(places[0]._distance) || 0 : null,
        samples: places.slice(0, 3).map((poi) => cleanText(poi.title, 36)).filter(Boolean)
      };
    });
  }

  return {
    context: {
      source: "腾讯位置服务",
      mode: options.mode || "gps",
      coordinateSystem: options.coordinateSystem || "GCJ-02",
      location: {
        // The browser uses these only for the temporary map picker. The
        // frontend strips them before it persists the private case snapshot.
        latitude: Number(lat.toFixed(6)),
        longitude: Number(lng.toFixed(6)),
        address: trimText(reverseResult.address, 100) || approximateAddress,
        addressResolution,
        province: cleanText(addressComponent.province, 20),
        city: cleanText(addressComponent.city, 20),
        district: cleanText(addressComponent.district, 20),
        adcode: cleanText(reverseResult.ad_info?.adcode, 12)
      },
      nearby: {
        keyword: nearbyAttempt.keyword || baseKeyword,
        requestedKeyword: baseKeyword,
        fallbackUsed: Boolean(nearbyAttempt.fallbackUsed),
        radiusMeters: 800,
        count: nearbyAttempt.error ? null : (Number(nearby.count) || competitors.length),
        places: competitors
      },
      landmarks,
      environment,
      ...(dataWarnings.length ? {
        dataQuality: {
          status: "degraded",
          reason: dataWarnings.join("；"),
          warnings: dataWarnings,
          requiresOnSiteVerification: true
        }
      } : {})
    }
  };
}

export async function getMapContext(url, env) {
  if (!configured(env)) return mapNotConfigured();
  const keys = tencentMapKeys(env);

  const lat = Number(url.searchParams.get("lat"));
  const lng = Number(url.searchParams.get("lng"));
  if (!validCoordinate(lat, lng)) {
    return json({ code: "INVALID_LOCATION", message: "当前位置坐标无效" }, 400);
  }
  const category = cleanText(url.searchParams.get("category"), 24) || "餐饮";

  try {
    // Browser geolocation is WGS84. Tencent WebService uses GCJ-02, so convert
    // before reverse geocoding and nearby search.
    let mapLat = lat;
    let mapLng = lng;
    let coordinateSystem = "GCJ-02";
    const dataQualityWarnings = [];
    try {
      const translated = await tencentRequest("/ws/coord/v1/translate", {
        locations: `${lat},${lng}`,
        type: 1,
        output: "json"
      }, keys);
      const mapLocation = translated.locations?.[0];
      mapLat = Number(mapLocation?.lat);
      mapLng = Number(mapLocation?.lng);
      if (!validCoordinate(mapLat, mapLng)) {
        throw new Error("腾讯地图坐标转换失败");
      }
    } catch (error) {
      if (!isTencentQuotaError(error)) throw error;
      // Coordinate conversion and POI search have independent Tencent quotas.
      // If only conversion is exhausted, still query the POI service with the
      // browser's WGS84 point. The result is deliberately marked approximate;
      // silently returning no shops would hide usable evidence.
      coordinateSystem = "WGS84-approximate";
      dataQualityWarnings.push("坐标转换服务暂时不可用，周边结果按近似位置读取");
    }

    return json(await buildContextFromGcj02(mapLat, mapLng, category, keys, {
      mode: "gps",
      rich: url.searchParams.get("rich") === "1",
      coordinateSystem,
      dataQualityWarnings
    }));
  } catch (error) {
    if (isTencentQuotaError(error)) {
      return json(quotaFallbackContext({
        lat,
        lng,
        mode: "gps",
        category,
        coordinateSystem: "WGS84",
        reason: "腾讯地图坐标转换服务暂时达到调用上限"
      }));
    }
    return json({
      code: "MAP_UPSTREAM_ERROR",
      message: error instanceof Error ? error.message : "腾讯地图暂时不可用"
    }, 502);
  }
}

export async function getAddressContext(url, env) {
  if (!configured(env)) return mapNotConfigured();
  const keys = tencentMapKeys(env);

  const address = trimText(url.searchParams.get("address"));
  if (address.length < 3) {
    return json({ code: "INVALID_ADDRESS", message: "请填写城市、商圈或详细地址" }, 400);
  }
  if (administrativeAreaOnly(address)) {
    return json({
      code: "ADDRESS_TOO_BROAD",
      message: "这个位置范围太大，请补充商圈、路名或门牌号"
    }, 422);
  }
  const category = cleanText(url.searchParams.get("category"), 24) || "餐饮";

  try {
    const geocoded = await tencentRequest("/ws/geocoder/v1/", {
      address,
      output: "json"
    }, keys);
    const location = geocoded.result?.location;
    const lat = Number(location?.lat);
    const lng = Number(location?.lng);
    if (!validCoordinate(lat, lng)) {
      return json({
        code: "ADDRESS_NOT_FOUND",
        message: "没有找到这个地址，请补充城市、商圈或门牌号"
      }, 422);
    }
    const level = cleanText(geocoded.result?.level, 16);
    if (["国家", "省", "城市", "区县", "行政区"].includes(level)) {
      return json({
        code: "ADDRESS_TOO_BROAD",
        message: "这个位置范围太大，请补充商圈、路名或门牌号"
      }, 422);
    }

    return json(await buildContextFromGcj02(lat, lng, category, keys, {
      mode: "address",
      fallbackAddress: address,
      rich: url.searchParams.get("rich") === "1"
    }));
  } catch (error) {
    if (isTencentQuotaError(error)) {
      return json(quotaFallbackContext({
        mode: "address",
        category,
        fallbackAddress: address,
        coordinateSystem: "unknown",
        reason: "腾讯地图地址解析服务暂时达到调用上限"
      }));
    }
    return json({
      code: "ADDRESS_LOOKUP_ERROR",
      message: error instanceof Error ? error.message : "地址解析暂时不可用"
    }, 502);
  }
}

// A point selected on the Tencent static map is already GCJ-02. Keeping this
// separate from browser GPS avoids applying the WGS84→GCJ02 conversion twice.
export async function getPickedMapContext(url, env) {
  if (!configured(env)) return mapNotConfigured();
  const keys = tencentMapKeys(env);
  const lat = Number(url.searchParams.get("lat"));
  const lng = Number(url.searchParams.get("lng"));
  if (!validCoordinate(lat, lng)) {
    return json({ code: "INVALID_LOCATION", message: "地图选点坐标无效" }, 400);
  }
  const category = cleanText(url.searchParams.get("category"), 24) || "餐饮";
  try {
    return json(await buildContextFromGcj02(lat, lng, category, keys, {
      mode: "map-picker",
      rich: url.searchParams.get("rich") === "1"
    }));
  } catch (error) {
    if (isTencentQuotaError(error)) {
      return json(quotaFallbackContext({
        lat,
        lng,
        mode: "map-picker",
        category,
        coordinateSystem: "GCJ-02",
        reason: "腾讯地图周边服务暂时达到调用上限"
      }));
    }
    return json({
      code: "MAP_UPSTREAM_ERROR",
      message: error instanceof Error ? error.message : "腾讯地图暂时不可用"
    }, 502);
  }
}

// Static map image is proxied through the Worker: the server-only WebService
// key stays private, while the browser can still see and click a real map.
export async function getStaticMap(url, env) {
  if (!configured(env)) return mapNotConfigured();
  const keys = tencentMapKeys(env);
  const lat = Number(url.searchParams.get("lat"));
  const lng = Number(url.searchParams.get("lng"));
  const zoom = Math.max(14, Math.min(18, Math.round(Number(url.searchParams.get("zoom")) || 16)));
  if (!validCoordinate(lat, lng)) {
    return json({ code: "INVALID_LOCATION", message: "地图坐标无效" }, 400);
  }
  let lastError = null;
  try {
    for (const key of keys) {
      try {
        const target = new URL("/ws/staticmap/v2/", "https://apis.map.qq.com");
        target.searchParams.set("center", `${lat.toFixed(6)},${lng.toFixed(6)}`);
        target.searchParams.set("zoom", String(zoom));
        target.searchParams.set("size", "640*360");
        target.searchParams.set("maptype", "roadmap");
        target.searchParams.set("key", key);
        const response = await fetch(target, {
          headers: { "Accept": "image/avif,image/webp,image/*,*/*;q=0.8", "Referer": "https://shopvalidator.zhangyvjing.com/" },
          signal: AbortSignal.timeout(7000)
        });
        if (!response.ok) {
          const error = new Error(`腾讯静态地图请求失败：${response.status}`);
          error.tencentStatus = response.status;
          throw error;
        }
        const type = response.headers.get("Content-Type") || "image/png";
        if (!type.startsWith("image/")) {
          const payload = await response.clone().json().catch(() => null);
          const error = new Error(payload?.message || "腾讯静态地图返回异常");
          error.tencentStatus = Number(payload?.status);
          throw error;
        }
        return new Response(response.body, {
          headers: {
            "Content-Type": type,
            "Cache-Control": "private, max-age=120",
            "X-Content-Type-Options": "nosniff"
          }
        });
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError || new Error("腾讯地图密钥尚未配置");
  } catch (error) {
    if (isTencentQuotaError(lastError || error)) {
      return staticMapQuotaFallback(lat, lng, zoom);
    }
    return json({
      code: "STATIC_MAP_ERROR",
      message: error instanceof Error ? error.message : "地图底图暂时不可用"
    }, 502);
  }
}

function clientIp(request) {
  const forwarded = request.headers.get("CF-Connecting-IP") ||
    request.headers.get("X-Forwarded-For")?.split(",")[0];
  const value = trimText(forwarded, 64);
  return /^[0-9a-f:.]+$/i.test(value) ? value : "";
}

export async function getApproximateLocation(request, env) {
  if (!configured(env)) return mapNotConfigured();
  const keys = tencentMapKeys(env);
  const ip = clientIp(request);
  if (!ip) {
    return json({
      code: "IP_UNAVAILABLE",
      message: "无法识别大致城市，请手动输入店铺地址"
    }, 422);
  }

  try {
    const located = await tencentRequest("/ws/location/v1/ip", {
      ip,
      output: "json"
    }, keys);
    const adInfo = located.result?.ad_info || {};
    const province = cleanText(adInfo.province, 20);
    const city = cleanText(adInfo.city, 20);
    const district = cleanText(adInfo.district, 20);
    const label = [province, city, district]
      .filter((part, index, all) => part && all.indexOf(part) === index)
      .join("");
    if (!label) throw new Error("无法识别大致城市");

    // Do not expose or use the IP result's city-government coordinate as the
    // store location. The client must still obtain GPS or resolve an address.
    return json({
      approximate: {
        source: "腾讯位置服务·网络定位",
        precision: district ? "district" : "city",
        province,
        city,
        district,
        adcode: cleanText(adInfo.adcode, 12),
        label
      }
    });
  } catch (error) {
    return json({
      code: "IP_LOCATION_ERROR",
      message: error instanceof Error ? error.message : "无法识别大致城市"
    }, 502);
  }
}

function secureId(prefix) {
  return `${prefix}_${crypto.randomUUID().replace(/-/g, "")}`;
}

async function tokenHash(token) {
  const bytes = new TextEncoder().encode(String(token || ""));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function caseSnapshot(record) {
  const { token: _token, ...snapshot } = record;
  return snapshot;
}

function cloneCaseRecord(record) {
  // Case records are JSON-compatible by construction. Work on a detached
  // value so a failed compare-and-swap cannot leak partial facts/turns into
  // this isolate's authoritative cache.
  return typeof structuredClone === "function"
    ? structuredClone(record)
    : JSON.parse(JSON.stringify(record));
}

async function persistCase(env, record) {
  if (!env.DB) return;
  await env.DB.prepare(`
    INSERT INTO cases (id, token_hash, stage, location_json, case_json, version, selected_plan_id, tts_count, created_at, updated_at, expires_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      stage=excluded.stage, location_json=excluded.location_json, case_json=excluded.case_json,
      version=excluded.version, selected_plan_id=excluded.selected_plan_id,
      updated_at=excluded.updated_at, expires_at=excluded.expires_at
  `).bind(
    record.id,
    record.tokenHash || await tokenHash(record.token),
    record.stage || "",
    JSON.stringify(record.location),
    JSON.stringify(caseSnapshot(record)),
    record.version,
    record.selectedPlanId,
    Number(record.ttsCount) || 0,
    record.createdAt,
    record.updatedAt,
    new Date(Date.now() + CASE_TTL_MS).toISOString()
  ).run();
}

async function persistCaseIfVersion(env, record, expectedVersion) {
  if (!env.DB) {
    const current = caseStore.get(record.id);
    if (!current || Number(current.version) !== Number(expectedVersion)) return false;
    caseStore.set(record.id, record);
    return true;
  }
  const result = await env.DB.prepare(`
    UPDATE cases
    SET stage = ?, location_json = ?, case_json = ?, version = ?,
        selected_plan_id = ?, updated_at = ?, expires_at = ?
    WHERE id = ? AND version = ?
  `).bind(
    record.stage || "",
    JSON.stringify(record.location),
    JSON.stringify(caseSnapshot(record)),
    record.version,
    record.selectedPlanId || null,
    record.updatedAt,
    new Date(Date.now() + CASE_TTL_MS).toISOString(),
    record.id,
    expectedVersion
  ).run();
  const committed = Number(result?.meta?.changes ?? 0) === 1;
  if (committed) caseStore.set(record.id, record);
  return committed;
}

async function persistInterviewTurnAudit(env, record, turn) {
  if (!env.DB) return;
  try {
    await env.DB.prepare(`
      INSERT OR IGNORE INTO interview_turns
        (id, case_id, field_name, question, transcript, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(
      turn.id,
      record.id,
      turn.field || "",
      turn.question || "",
      turn.transcript,
      turn.createdAt
    ).run();
  } catch {
    // The authoritative turn is already in cases.case_json. The normalized
    // audit table is useful for inspection but must not make a committed turn
    // appear failed to the caller.
  }
}

async function loadCase(env, caseId) {
  const cached = caseStore.get(caseId);
  if (!env.DB) return cached || null;
  // In production D1 is authoritative. A Queue consumer may have updated this
  // case in another isolate since the local cache entry was created.
  const row = await env.DB.prepare(
    "SELECT case_json, token_hash, tts_count FROM cases WHERE id = ? AND expires_at > ?"
  ).bind(caseId, nowIso()).first();
  if (!row?.case_json) return null;
  const record = JSON.parse(row.case_json);
  record.tokenHash = row.token_hash;
  record.ttsCount = Number(row.tts_count) || 0;
  caseStore.set(caseId, record);
  return record;
}

async function persistRun(env, run) {
  if (!env.DB) return;
  await env.DB.prepare(`
    INSERT INTO analysis_runs (id, case_id, case_version, status, progress_json, result_json, state_json, warning, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      status=excluded.status, progress_json=excluded.progress_json,
      result_json=excluded.result_json, state_json=excluded.state_json,
      warning=excluded.warning, updated_at=excluded.updated_at
  `).bind(
    run.id, run.caseId, run.caseVersion, run.status,
    JSON.stringify(run.progress), JSON.stringify(run.result), JSON.stringify({
      context: run.context,
      searchState: run.searchState
    }),
    run.warning || "", run.createdAt, run.updatedAt
  ).run();
}

async function loadRun(env, runId) {
  const cached = runStore.get(runId);
  if (!env.DB) return cached || null;
  // Always refresh persisted runs so HTTP polling cannot remain stuck on a
  // stale in-isolate "queued" snapshot after another isolate completes it.
  const row = await env.DB.prepare("SELECT * FROM analysis_runs WHERE id = ?").bind(runId).first();
  if (!row) return null;
  const persistedState = JSON.parse(row.state_json || "null") || {};
  const run = {
    id: row.id,
    caseId: row.case_id,
    caseVersion: row.case_version,
    status: row.status,
    progress: JSON.parse(row.progress_json || "null"),
    result: JSON.parse(row.result_json || "null"),
    context: persistedState.context,
    searchState: persistedState.searchState,
    claimToken: row.claim_token || null,
    claimedRound: Number(row.claimed_round) || null,
    claimExpiresAt: row.claim_expires_at || null,
    warning: row.warning || "",
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
  runStore.set(runId, run);
  return run;
}

async function insertRunIfAbsent(env, run) {
  if (!env.DB) return true;
  const result = await env.DB.prepare(`
    INSERT INTO analysis_runs (id, case_id, case_version, status, progress_json, result_json, state_json, warning, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(case_id, case_version) DO NOTHING
  `).bind(
    run.id, run.caseId, run.caseVersion, run.status,
    JSON.stringify(run.progress), JSON.stringify(run.result), JSON.stringify({
      context: run.context,
      searchState: run.searchState
    }),
    run.warning || "", run.createdAt, run.updatedAt
  ).run();
  return Number(result?.meta?.changes ?? 0) > 0;
}

async function findRunForCaseVersion(env, caseId, caseVersion) {
  if (env.DB) {
    const row = await env.DB.prepare(
      "SELECT id FROM analysis_runs WHERE case_id = ? AND case_version = ? LIMIT 1"
    ).bind(caseId, caseVersion).first();
    return row?.id ? loadRun(env, row.id) : null;
  }
  for (const run of runStore.values()) {
    if (run.caseId === caseId && run.caseVersion === caseVersion) return run;
  }
  return null;
}

export async function claimAnalysisRound(env, run, requestedRound, claimToken = secureId("claim")) {
  const expiresAt = new Date(Date.now() + QUEUE_CLAIM_TTL_MS).toISOString();
  const now = nowIso();
  if (!env.DB) {
    const existing = run.queueClaim;
    if (existing && existing.expiresAt > now) return null;
    run.queueClaim = { token: claimToken, round: requestedRound, expiresAt };
    return claimToken;
  }
  const result = await env.DB.prepare(`
    UPDATE analysis_runs
    SET claim_token = ?, claimed_round = ?, claim_expires_at = ?, updated_at = ?
    WHERE id = ?
      AND status NOT IN ('completed', 'failed')
      AND (claim_token IS NULL OR claim_expires_at IS NULL OR claim_expires_at <= ?)
  `).bind(
    claimToken, requestedRound, expiresAt, now, run.id, now
  ).run();
  return Number(result?.meta?.changes ?? 0) === 1 ? claimToken : null;
}

export async function releaseAnalysisRoundClaim(env, run, requestedRound, claimToken) {
  if (!claimToken) return;
  if (!env.DB) {
    if (run.queueClaim?.token === claimToken && run.queueClaim?.round === requestedRound) {
      run.queueClaim = null;
    }
    return;
  }
  await env.DB.prepare(`
    UPDATE analysis_runs
    SET claim_token = NULL, claimed_round = NULL, claim_expires_at = NULL, updated_at = ?
    WHERE id = ? AND claim_token = ? AND claimed_round = ?
  `).bind(nowIso(), run.id, claimToken, requestedRound).run();
}

function nowIso() {
  return new Date().toISOString();
}

function cleanupMemoryStores() {
  const cutoff = Date.now() - CASE_TTL_MS;
  for (const [id, value] of caseStore) {
    if (new Date(value.updatedAt).getTime() < cutoff) caseStore.delete(id);
  }
  for (const [id, value] of runStore) {
    if (new Date(value.updatedAt).getTime() < cutoff) runStore.delete(id);
  }
}

function corsHeaders(request, env) {
  const requestOrigin = request.headers.get("Origin");
  const allowed = allowedOrigins(env, new URL(request.url).origin);
  const headers = {
    "Vary": "Origin",
    "Access-Control-Allow-Headers": "Content-Type, X-Case-Token",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS"
  };
  if (requestOrigin && allowed.has(requestOrigin)) {
    headers["Access-Control-Allow-Origin"] = requestOrigin;
  }
  return headers;
}

function allowedOrigins(env, requestOrigin) {
  return new Set([env.APP_ORIGIN, env.DEMO_ORIGIN, requestOrigin].filter(Boolean));
}

function apiJson(request, env, body, status = 200) {
  const response = json(body, status);
  const headers = new Headers(response.headers);
  Object.entries(corsHeaders(request, env)).forEach(([key, value]) => headers.set(key, value));
  headers.set("Referrer-Policy", "same-origin");
  headers.set("Permissions-Policy", "camera=(), geolocation=(self), microphone=(self)");
  return new Response(response.body, { status: response.status, headers });
}

function originAllowed(request, env) {
  const origin = request.headers.get("Origin");
  if (!origin) return true;
  const requestUrl = new URL(request.url);
  if (allowedOrigins(env, requestUrl.origin).has(origin)) return true;
  try {
    const originUrl = new URL(origin);
    const loopback = (hostname) => ["localhost", "127.0.0.1", "::1"].includes(hostname);
    return loopback(requestUrl.hostname) && loopback(originUrl.hostname);
  } catch {
    return false;
  }
}

function rateAllowed(request, limit = 90, windowMs = 60_000, namespace = "global") {
  const key = `${namespace}:${clientIp(request) || "unknown"}`;
  return localRateAllowed(key, limit, windowMs);
}

function localRateAllowed(key, limit, windowMs) {
  const now = Date.now();
  const bucket = rateBuckets.get(key);
  if (!bucket || bucket.resetAt <= now) {
    rateBuckets.set(key, { count: 1, resetAt: now + windowMs });
    return true;
  }
  bucket.count += 1;
  return bucket.count <= limit;
}

function rateAction(request, url) {
  if (request.method === "POST" && url.pathname === "/api/cases") return "create-case";
  if (request.method === "GET" && url.pathname.startsWith("/api/map/")) return "map";
  if (request.method === "POST" && url.pathname === "/api/tts") return "tts";
  if (request.method === "POST" && /^\/api\/cases\/[^/]+\/turns$/.test(url.pathname)) return "turn";
  if (request.method === "POST" && /^\/api\/cases\/[^/]+\/analyze$/.test(url.pathname)) return "analyze";
  if (request.method === "POST" && /^\/api\/cases\/[^/]+\/asr$/.test(url.pathname)) return "asr-transcribe";
  return "";
}

async function persistentRateCheck(env, { action, key, limit, windowMs }) {
  if (!env.AGENT_GATE) {
    return {
      allowed: localRateAllowed(`${action}:${key}`, limit, windowMs),
      retryAfterMs: windowMs
    };
  }
  try {
    const id = env.AGENT_GATE.idFromName("global-stepfun-gate");
    const gate = env.AGENT_GATE.get(id);
    const response = await gate.fetch("https://agent-gate.internal/rate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action,
        key,
        limit,
        windowMs
      })
    });
    if (!response.ok) throw new Error("rate gate unavailable");
    return await response.json();
  } catch {
    // A transient DO failure must not remove quota protection.
    return {
      allowed: localRateAllowed(`${action}:${key}`, limit, windowMs),
      retryAfterMs: windowMs
    };
  }
}

async function actionRateAllowed(request, env, action) {
  const policy = ACTION_RATE_LIMITS[action];
  if (!policy) return true;
  const ipHash = await tokenHash(clientIp(request) || "unknown");
  return Boolean((await persistentRateCheck(env, {
    action,
    key: ipHash,
    limit: policy.limit,
    windowMs: policy.windowMs
  })).allowed);
}

async function readJson(request) {
  const length = Number(request.headers.get("Content-Length") || 0);
  if (length > MAX_JSON_BYTES) throw Object.assign(new Error("请求内容过大"), { status: 413 });
  const text = await request.text();
  if (new TextEncoder().encode(text).byteLength > MAX_JSON_BYTES) {
    throw Object.assign(new Error("请求内容过大"), { status: 413 });
  }
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    throw Object.assign(new Error("JSON格式无效"), { status: 400 });
  }
}

async function requireCase(request, env, caseId) {
  const record = await loadCase(env, caseId);
  if (!record) throw Object.assign(new Error("案卷不存在或已过期"), { status: 404 });
  const token = request.headers.get("X-Case-Token") || new URL(request.url).searchParams.get("token");
  const matches = token && (
    (record.token && token === record.token) ||
    (record.tokenHash && await tokenHash(token) === record.tokenHash)
  );
  if (!matches) {
    throw Object.assign(new Error("案卷访问凭证无效"), { status: 403 });
  }
  return record;
}

function normalizeFact(raw, source = "voice") {
  const id = cleanText(raw?.id || raw?.field, 60);
  if (!id) return null;
  const allowedStatus = ["confirmed", "provisional", "assumption", "unknown", "conflict"];
  const allowedEvidence = ["A", "B", "C", "D", "U"];
  return {
    id,
    field: cleanText(raw?.field || id, 60),
    value: raw?.value ?? null,
    range: raw?.range && typeof raw.range === "object" ? raw.range : null,
    unit: cleanText(raw?.unit, 24),
    period: cleanText(raw?.period, 24),
    status: allowedStatus.includes(raw?.status) ? raw.status : "provisional",
    source: cleanText(raw?.source || source, 24),
    evidence: allowedEvidence.includes(raw?.evidence) ? raw.evidence : "C",
    transcript: trimText(raw?.transcript, 500),
    updatedAt: nowIso()
  };
}

function equivalentFact(left, right) {
  const comparable = (fact) => ({
    id: fact?.id || "",
    field: fact?.field || "",
    value: fact?.value ?? null,
    range: fact?.range || null,
    unit: fact?.unit || "",
    period: fact?.period || "",
    status: fact?.status || "",
    source: fact?.source || "",
    evidence: fact?.evidence || "",
    transcript: fact?.transcript || ""
  });
  return JSON.stringify(comparable(left)) === JSON.stringify(comparable(right));
}

function interviewPolicyState(record, { facts = record.facts, turns = record.turns } = {}) {
  return {
    stage: record.stage,
    facts,
    turns,
    locationConfirmed: record.location?.confirmed === true
  };
}

function nextQuestion(record) {
  return sanitizeAgentNextQuestion(null, interviewPolicyState(record));
}

function safeFactsForModel(facts) {
  return Object.fromEntries(Object.entries(facts).map(([id, fact]) => [id, {
    value: fact.value,
    range: fact.range,
    unit: fact.unit,
    period: fact.period,
    status: fact.status,
    evidence: fact.evidence
  }]));
}

function canonicalInterviewFacts(rawFacts, { stage, source = "voice", transcript = "" } = {}) {
  const allowed = new Set(getAllowedInterviewFields(stage));
  const filtered = (Array.isArray(rawFacts) ? rawFacts : [])
    .filter((fact) => allowed.has(cleanText(fact?.field || fact?.id, 60)))
    .map((fact) => ({ ...fact, source, transcript, rawTranscript: transcript }));
  const { facts } = normalizeServerFacts(filtered);
  return facts.map((fact) => ({
    id: fact.field,
    field: fact.field,
    value: fact.value,
    range: fact.range,
    unit: fact.unit || "",
    period: fact.period || "",
    status: fact.status,
    source: fact.source,
    evidence: fact.evidenceGrade,
    transcript: fact.rawTranscript || transcript,
    updatedAt: nowIso()
  }));
}

function deterministicAnswerFact(field, transcript, stage, source) {
  const text = String(transcript || "").trim();
  if (!field || !text) return [];
  if (/^(不知道|不清楚|不确定|没有数据|unknown)$/i.test(text)) {
    return canonicalInterviewFacts([{ field, status: "unknown", value: null, evidence: "U" }], { stage, source, transcript: text });
  }
  const textFields = new Set(["bottleneck", "growthBottleneck", "channelMix", "leaseRemaining", "targetCustomer"]);
  const choiceFields = new Set(["trialSale", "trafficMatch", "visibility", "retention", "reversibleInvestment"]);
  if (textFields.has(field)) return canonicalInterviewFacts([{ field, value: text.slice(0, 100), status: "provisional", evidence: "C" }], { stage, source, transcript: text });
  if (choiceFields.has(field)) {
    const value = /(没有|没|不|否)/.test(text) ? "no" : /(有|是|做过|能|可以)/.test(text) ? "yes" : "unknown";
    return canonicalInterviewFacts([{ field, value, status: value === "unknown" ? "unknown" : "provisional", evidence: value === "unknown" ? "U" : "C" }], { stage, source, transcript: text });
  }
  const matched = text.match(/(\d+(?:\.\d+)?)\s*(万|千|百)?/);
  if (!matched) return [];
  let value = Number(matched[1]);
  if (matched[2] === "万") value *= 10000;
  if (matched[2] === "千") value *= 1000;
  if (matched[2] === "百") value *= 100;
  const period = /年|年度/.test(text) ? "year" : /天|日/.test(text) ? "day" : "month";
  return canonicalInterviewFacts([{ field, value, unit: field === "variableCostRate" ? "%" : "CNY", period, status: "provisional", evidence: "C" }], { stage, source, transcript: text });
}

async function processInterviewTurn(record, transcript, llm, source = "voice") {
  const attemptedTurn = {
    field: record.currentQuestion?.field || nextQuestion(record).field
  };
  const fallbackState = interviewPolicyState(record, {
    turns: [...record.turns, attemptedTurn]
  });
  const fallback = sanitizeAgentNextQuestion(null, fallbackState);
  const currentField = attemptedTurn.field;
  if (/^(不知道|不清楚|不确定|没有数据|unknown)$/i.test(String(transcript).trim())) {
    const facts = canonicalInterviewFacts([{
      id: currentField, field: currentField, status: "unknown", value: null, evidence: "U"
    }], { stage: record.stage, source, transcript });
    const stateAfterTurn = interviewPolicyState(record, { facts: { ...record.facts, [currentField]: facts[0] }, turns: [...record.turns, attemptedTurn] });
    return { facts, nextQuestion: sanitizeAgentNextQuestion(null, stateAfterTurn), mode: "deterministic-unknown" };
  }
  if (!llm) {
    const facts = deterministicAnswerFact(currentField, transcript, record.stage, source);
    const mergedFacts = { ...record.facts };
    for (const fact of facts) mergedFacts[fact.id] = fact;
    return {
      facts,
      nextQuestion: sanitizeAgentNextQuestion(null, interviewPolicyState(record, { facts: mergedFacts, turns: [...record.turns, attemptedTurn] })),
      mode: "deterministic-fallback"
    };
  }
  const response = await llm([
    {
      role: "system",
      content: "你只负责从本轮餐饮回答中抽取事实。只返回JSON；不得提问、不得决定流程。不得把未知当0，不得把流水当利润；金额必须保存单位和周期；范围保留范围。"
    },
    {
      role: "user",
      content: JSON.stringify({
        transcript,
        current_question: record.currentQuestion,
        confirmed_location: record.location,
        known_facts: safeFactsForModel(record.facts),
        required_fields: evaluateInterviewCompleteness(interviewPolicyState(record)).requiredFields,
        output_schema: {
          facts: [{
            id: "field", field: "field", value: null, range: null,
            unit: "", period: "", status: "provisional|unknown|conflict",
            evidence: "C|D"
          }],
          contradictions: []
        }
      })
    }
  ], {
    temperature: 0.1,
    maxTokens: 700,
    // OpenRouter free routes can spend a few seconds selecting an available
    // provider before returning structured JSON. Keep a hard upper bound, but
    // do not turn normal free-route latency into a deterministic-only turn.
    timeoutMs: 15000
  });
  const extracted = canonicalInterviewFacts(response?.facts, { stage: record.stage, source, transcript });
  const facts = (extracted.length ? extracted : deterministicAnswerFact(currentField, transcript, record.stage, source)).slice(0, 8);
  const mergedFacts = { ...record.facts };
  for (const fact of facts) mergedFacts[fact.id] = fact;
  const stateAfterTurn = interviewPolicyState(record, {
    facts: mergedFacts,
    turns: [...record.turns, attemptedTurn]
  });
  const next = sanitizeAgentNextQuestion(null, stateAfterTurn);
  return {
    facts,
    nextQuestion: next,
    contradictions: Array.isArray(response?.contradictions) ? response.contradictions.slice(0, 8) : [],
    mode: "stepfun"
  };
}

async function createCase(request, env) {
  const body = await readJson(request);
  const id = secureId("case");
  const token = secureId("token");
  const timestamp = nowIso();
  const record = {
    id,
    token,
    stage: cleanText(body.stage, 30),
    category: cleanText(body.category, 30),
    location: null,
    facts: {},
    turns: [],
    reviews: [],
    currentQuestion: null,
    version: 1,
    latestRunId: null,
    selectedPlanId: null,
    ttsCount: 0,
    createdAt: timestamp,
    updatedAt: timestamp
  };
  caseStore.set(id, record);
  await persistCase(env, record);
  return apiJson(request, env, {
    case: { id, version: 1, createdAt: timestamp },
    caseToken: token
  }, 201);
}

async function saveLocation(request, env, caseId) {
  const record = await requireCase(request, env, caseId);
  const body = await readJson(request);
  if (!body.confirmed || !body.context?.location?.address) {
    return apiJson(request, env, {
      code: "LOCATION_NOT_CONFIRMED",
      message: "必须先从候选位置中确认店铺地址"
    }, 422);
  }
  record.location = {
    confirmed: true,
    context: body.context,
    confirmedAt: nowIso()
  };
  record.facts.location = normalizeFact({
    id: "location",
    value: body.context.location.address,
    status: "confirmed",
    source: "map",
    evidence: "B"
  }, "map");
  record.version += 1;
  record.updatedAt = nowIso();
  record.currentQuestion = nextQuestion(record);
  await persistCase(env, record);
  return apiJson(request, env, {
    caseId,
    version: record.version,
    location: record.location,
    firstQuestion: record.currentQuestion,
    interview: evaluateInterviewCompleteness(interviewPolicyState(record))
  });
}

async function interviewTurn(request, env, caseId) {
  const record = await requireCase(request, env, caseId);
  if (!record.location?.confirmed) {
    return apiJson(request, env, {
      code: "LOCATION_REQUIRED",
      message: "请先确认店铺位置"
    }, 409);
  }
  if (record.turns.length >= INTERVIEW_LIMITS.maxTurns) {
    return apiJson(request, env, {
      complete: true,
      reason: "MAX_TURNS",
      facts: Object.values(record.facts)
    });
  }
  const body = await readJson(request);
  const transcript = trimText(body.answer ?? body.transcript, 4000);
  const source = ["voice", "typed", "choice"].includes(body.source) ? body.source : "voice";
  // `0` is a complete answer for numeric facts such as debt and must never be
  // confused with an empty transcript.  The field-specific FactArchive
  // normalizer below remains responsible for validating its meaning.
  if (!transcript) {
    return apiJson(request, env, {
      code: "EMPTY_TRANSCRIPT",
      message: "没有识别到有效回答，请继续说或选择不知道"
    }, 422);
  }
  const turnId = cleanText(body.turnId, 80) || secureId("turn");
  const existingTurn = record.turns.find((turn) => turn.id === turnId);
  if (existingTurn) {
    return apiJson(request, env, {
      turnId,
      duplicate: true,
      version: record.version,
      extractedFacts: [],
      contradictions: [],
      nextQuestion: record.currentQuestion || nextQuestion(record),
      complete: Boolean((record.currentQuestion || nextQuestion(record)).complete)
    });
  }
  const expectedVersion = Number(record.version);
  const suppliedVersion = body.expectedVersion ?? body.caseVersion;
  if (suppliedVersion != null && !Number.isInteger(Number(suppliedVersion))) {
    return apiJson(request, env, {
      code: "CASE_VERSION_INVALID",
      message: "案卷版本格式无效",
      version: expectedVersion
    }, 422);
  }
  if (suppliedVersion != null && Number(suppliedVersion) !== expectedVersion) {
    return apiJson(request, env, {
      code: "CASE_VERSION_CONFLICT",
      message: "案卷已经被另一轮更新，请使用最新问题重试",
      version: expectedVersion,
      nextQuestion: record.currentQuestion || nextQuestion(record)
    }, 409);
  }
  if (localTurnLocks.has(caseId)) {
    return apiJson(request, env, {
      code: "TURN_IN_PROGRESS",
      message: "上一轮仍在处理，请稍后重试同一个回答"
    }, 409);
  }
  localTurnLocks.add(caseId);
  try {
    const working = cloneCaseRecord(record);
    const questionSnapshot = working.currentQuestion || nextQuestion(working);
    const llm = createTextLlm(env);
    let processed;
    try {
      processed = await processInterviewTurn(working, transcript, llm, source);
    } catch (error) {
      processed = {
        facts: [],
        nextQuestion: nextQuestion(working),
        mode: "deterministic-fallback",
        warning: error instanceof Error ? error.message : "问诊模型暂时不可用"
      };
    }
    // A valid numeric/text answer must not disappear merely because the model
    // timed out or returned an empty facts array. This is still constrained to
    // the current deterministic field and the same FactArchive normalizer.
    if (!processed.facts?.length) {
      processed.facts = deterministicAnswerFact(questionSnapshot?.field, transcript, working.stage, source);
    }
    for (const fact of processed.facts) working.facts[fact.id] = fact;
    const committedTurn = {
      id: turnId,
      field: questionSnapshot?.field || "",
      question: questionSnapshot?.text || "",
      transcript,
      source,
      createdAt: nowIso()
    };
    working.turns.push(committedTurn);
    // Always derive the next question from the committed state (merged facts +
    // appended turn). processInterviewTurn may have produced its proposal from
    // a pre-commit snapshot when the extraction model timed out or errored,
    // which re-served the question that was just answered.
    working.currentQuestion = nextQuestion(working);
    working.version = expectedVersion + 1;
    working.updatedAt = nowIso();
    working.latestRunId = null;
    if (!await persistCaseIfVersion(env, working, expectedVersion)) {
      const latest = env.DB
        ? await loadCase(env, caseId)
        : caseStore.get(caseId);
      return apiJson(request, env, {
        code: "TURN_CONFLICT",
        message: "另一轮回答已先写入，当前回答没有覆盖它；请按最新问题重试",
        retryable: true,
        version: latest?.version,
        nextQuestion: latest?.currentQuestion || (latest ? nextQuestion(latest) : null)
      }, 409);
    }
    await persistInterviewTurnAudit(env, working, committedTurn);
    return apiJson(request, env, {
      turnId,
      version: working.version,
      extractedFacts: processed.facts,
      contradictions: processed.contradictions || [],
      nextQuestion: working.currentQuestion,
      interview: evaluateInterviewCompleteness(interviewPolicyState(working)),
      complete: working.currentQuestion.complete || working.turns.length >= INTERVIEW_LIMITS.maxTurns,
      mode: processed.mode,
      warning: processed.warning
    });
  } finally {
    localTurnLocks.delete(caseId);
  }
}

async function reviewFacts(request, env, caseId) {
  const record = await requireCase(request, env, caseId);
  if (localTurnLocks.has(caseId)) {
    return apiJson(request, env, {
      code: "TURN_IN_PROGRESS",
      message: "上一轮回答仍在处理，请等问题更新后再确认事实",
      version: record.version,
      facts: Object.values(record.facts)
    }, 409);
  }
  const body = await readJson(request);
  const expectedVersion = Number(record.version);
  const suppliedVersion = body.caseVersion;
  if (suppliedVersion != null && !Number.isInteger(Number(suppliedVersion))) {
    return apiJson(request, env, {
      code: "CASE_VERSION_INVALID",
      message: "案卷版本格式无效",
      version: expectedVersion,
      facts: Object.values(record.facts)
    }, 422);
  }
  if (suppliedVersion != null && Number(suppliedVersion) !== expectedVersion) {
    return apiJson(request, env, {
      code: "CASE_VERSION_CONFLICT",
      message: "案卷已发生变化，请刷新事实后再提交",
      version: expectedVersion,
      facts: Object.values(record.facts)
    }, 409);
  }
  const corrections = Array.isArray(body.corrections)
    ? body.corrections.slice(0, 80)
    : Array.isArray(body.facts)
      ? body.facts.slice(0, 80)
      : [];
  if (!corrections.length) {
    return apiJson(request, env, { code: "NO_CORRECTIONS", message: "没有提交任何确认选项" }, 422);
  }
  const working = cloneCaseRecord(record);
  let changed = 0;
  for (const correction of corrections) {
    const id = cleanText(correction.id, 60);
    if (!id) continue;
    const source = correction.source === "typed" ? "typed" : "choice";
    const existing = working.facts[id] || { id, field: id };
    const normalized = canonicalInterviewFacts([{
      ...existing,
      ...correction,
      id,
      field: id,
      source,
      evidence: correction.status === "unknown" ? "U" : "B"
    }], { stage: working.stage, source, transcript: correction.rawTranscript || correction.transcript || "" })[0];
    if (normalized && !equivalentFact(existing, normalized)) {
      working.facts[id] = normalized;
      changed += 1;
    }
  }
  if (!changed) {
    return apiJson(request, env, {
      caseId,
      version: expectedVersion,
      unchanged: true,
      facts: Object.values(record.facts)
    });
  }
  working.reviews = Array.isArray(working.reviews) ? working.reviews : [];
  working.reviews.push({ corrections: changed, createdAt: nowIso() });
  working.version = expectedVersion + 1;
  working.updatedAt = nowIso();
  working.latestRunId = null;
  if (!await persistCaseIfVersion(env, working, expectedVersion)) {
    const latest = env.DB
      ? await loadCase(env, caseId)
      : caseStore.get(caseId);
    return apiJson(request, env, {
      code: "CASE_VERSION_CONFLICT",
      message: "另一份事实确认已先提交；当前修改没有覆盖它",
      retryable: true,
      version: latest?.version,
      facts: Object.values(latest?.facts || {})
    }, 409);
  }
  return apiJson(request, env, {
    caseId,
    version: working.version,
    facts: Object.values(working.facts)
  });
}

function analysisContext(record, _body) {
  // Client-provided arithmetic is presentation-only and untrusted. The Worker
  // recomputes the deterministic judgment from the persisted, reviewed facts.
  const deterministic = computeServerDecision(record);
  return {
    facts: safeFactsForModel(record.facts),
    decision: cleanText(deterministic.decision, 20) || "EVIDENCE",
    title: trimText(deterministic.title, 160),
    reason: trimText(deterministic.reason, 500),
    metrics: deterministic.metrics && typeof deterministic.metrics === "object"
      ? deterministic.metrics
      : {},
    deterministic,
    flags: {
      hasStaffCapacityEvidence: Boolean(
        record.facts.staffCapacity?.status === "confirmed" &&
        record.facts.staffSchedule?.status === "confirmed"
      )
    }
  };
}

// ==== 勇哥地图选址报告（preopen 直接出报告，跳过财务问诊）====
const YONGGE_SITE_FRAMEWORK = "勇哥第一性原理选址判断：先看客群从哪来（住宅/写字楼/学校/交通/商场），再看竞争是否过度，最后看环境冲突（纯工地、拆迁、只有办公无夜间人流等要直接扣分）。地图POI只是环境证据，不等于精确客流、营业额或租金，任何结论都要标注“需现场验证”。判定口径：竞争不过度且客群清晰且无环境冲突→值得开(GO)；信号一般或混合→先低成本验证(TEST)；强竞品密集、目标客群缺失或环境明显冲突→不建议(STOP)。";

function siteGeoSummary(record) {
  const ctx = record.location?.context || {};
  const loc = ctx.location || {};
  const nearby = ctx.nearby || {};
  return {
    address: cleanText(loc.address, 100),
    city: cleanText(loc.city, 20),
    district: cleanText(loc.district, 20),
    competitorKeyword: cleanText(nearby.keyword, 24),
    competitorCount: Number(nearby.count) || (Array.isArray(nearby.places) ? nearby.places.length : 0),
    competitors: (nearby.places || []).slice(0, 12).map((poi) => ({
      title: cleanText(poi.title, 36),
      category: cleanText(poi.category, 40),
      distance: Number(poi.distance) || 0
    })),
    landmarks: (ctx.landmarks || []).slice(0, 8).map((poi) => ({
      title: cleanText(poi.title, 36),
      category: cleanText(poi.category, 40),
      distance: Number(poi.distance) || 0
    })),
    environment: (ctx.environment || []).map((group) => ({
      label: cleanText(group.label, 20),
      count: Number(group.count) || 0,
      nearestMeters: Number.isFinite(group.nearestMeters) ? group.nearestMeters : null,
      samples: Array.isArray(group.samples) ? group.samples.slice(0, 3).map((s) => cleanText(s, 36)) : []
    })),
    dataQuality: ctx.dataQuality && typeof ctx.dataQuality === "object"
      ? {
          status: cleanText(ctx.dataQuality.status, 20) || "ok",
          reason: trimText(ctx.dataQuality.reason, 240),
          requiresOnSiteVerification: Boolean(ctx.dataQuality.requiresOnSiteVerification)
        }
      : { status: "ok", reason: "", requiresOnSiteVerification: false }
  };
}

function siteMetricsFromGeo(geo) {
  const metrics = [
    {
      label: "800米同类竞品",
      value: geo.dataQuality?.status === "degraded" ? "暂未读取" : `${geo.competitorCount} 个`
    }
  ];
  const crowd = geo.environment
    .filter((group) => group.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, 2);
  crowd.forEach((group) => {
    metrics.push({ label: group.label, value: `${group.count} 处` });
  });
  while (metrics.length < 3) metrics.push({ label: "客群信号", value: "需现场核" });
  return metrics.slice(0, 3);
}

// Rule-based site judgment used when the LLM is unavailable or returns garbage.
export function fallbackSiteReport(geo, reportType, category) {
  const dataDegraded = geo.dataQuality?.status === "degraded";
  const competitors = geo.competitorCount || 0;
  const env = Object.fromEntries(geo.environment.map((group) => [group.label, group.count || 0]));
  const hasCrowd = (env["学校/大学"] || 0) + (env["写字楼/公司"] || 0) + (env["住宅小区"] || 0) > 0;
  let decision = "TEST";
  if (competitors <= 8 && hasCrowd) decision = "GO";
  else if (competitors >= 20 || !hasCrowd) decision = "STOP";
  if (dataDegraded) decision = "TEST";

  const scored = [
    {
      title: "现制茶饮 / 咖啡", why: "学校与写字楼信号同时存在时，适合用高频、小客单和复购承接碎片时间。",
      competitionReason: "先逐家看同类门店的价格带、排队和外卖评分；已有强连锁密集时，不应只靠装修硬碰。",
      operatingRequirement: "需要稳定出杯、清楚的价格锚点和工作日高峰承接能力。",
      risk: "若写字楼只在工作日短时活跃、学校并非可步行消费客群，复购假设会落空。",
      action: "连续两个工作日记录上午、午后和傍晚的经过人数、同类门店进店与外卖取餐。",
      budgetCap: 500, durationDays: 3, metric: "高峰进店率与同类价格带", successLine: "目标时段存在可观察的进店与复购需求，且能说清差异化", stopLine: "客群只路过不消费，或同类价格与产品完全无法区分"
    },
    {
      title: "快餐 / 简餐", why: "写字楼与住宅叠加时，午晚两段刚需可能支撑更高频的用餐选择。",
      competitionReason: "重点不是餐饮店数量，而是午高峰是否已被同价位、同速度的简餐完全占住。",
      operatingRequirement: "需要高峰出餐稳定、菜单聚焦，并能把堂食与外卖的动线拆开。",
      risk: "如果只有办公客群而晚间空置，或午高峰已被成熟连锁占满，翻台快并不等于能分到订单。",
      action: "分别在工作日午高峰、晚高峰记录竞品排队、出餐速度、客单提示与空桌变化。",
      budgetCap: 600, durationDays: 3, metric: "午晚高峰竞品承接与空桌变化", successLine: "两个高峰都存在未被满足的用餐需求，且可定义更快或更清楚的产品入口", stopLine: "需求只集中在单一时段，或现有竞品已无明显承接缺口"
    },
    {
      title: "小吃 / 夜宵", why: "学校与住宅信号存在时，低门槛、可外带的小吃更可能承接晚间和非正餐时段。",
      competitionReason: "要看夜间仍营业的同类是否已经形成聚集，以及用户是否为了一个单品专程停留。",
      operatingRequirement: "需要明确的单品记忆点、短制作时间和对晚间时段的稳定覆盖。",
      risk: "白天客流不能替代夜间需求；若夜间没有停留场景，低客单会放大房租和人工压力。",
      action: "在周中与周末的晚间各蹲点一次，记录停留人群、同类成交和外带比例。",
      budgetCap: 400, durationDays: 3, metric: "夜间停留与外带成交信号", successLine: "夜间存在连续停留与可观察的外带成交，且单品可与同类区分", stopLine: "夜间只有路过没有停留，或同类已经覆盖主要需求"
    },
    {
      title: "社区生鲜 / 便利", why: "住宅密集时，补货、即时消费与日常刚需比纯餐饮更不依赖一次性目的消费。",
      competitionReason: "先核对社区入口、现有便利店覆盖与即时零售配送半径，避免把住宅数量误当作购买缺口。",
      operatingRequirement: "需要靠近真实出入口、稳定高周转商品和可控损耗，而不是只摆更多 SKU。",
      risk: "若住户动线不经过门口，或周边已有成熟便利与配送覆盖，刚需不会自动转化。",
      action: "早晚各观察一次社区出入口，记录补货型消费、现有便利店排队与配送骑手密度。",
      budgetCap: 300, durationDays: 3, metric: "社区入口经过与补货型消费信号", successLine: "门口处于稳定出入口动线，且现有供给存在品类或时段缺口", stopLine: "住户动线绕开门口，或已有供给已覆盖主要补货需求"
    },
    {
      title: "正餐 / 家庭餐", why: "商场与住宅共同聚客时，周末和晚餐可能形成家庭或多人用餐场景。",
      competitionReason: "需要逐家比较家庭餐的等位、包间、客单与停车条件，不能只看商场和住宅的数量。",
      operatingRequirement: "需要较强的晚餐服务、稳定后厨和足够的停留体验，启动成本通常更高。",
      risk: "若没有周末与晚餐停留，正餐会先承担更高面积、人工和厨房投入。",
      action: "在周末午晚餐各观察一次，记录家庭客比例、等位、空桌和停车可达性。",
      budgetCap: 800, durationDays: 4, metric: "周末家庭客与晚餐停留", successLine: "周末和晚餐均出现稳定家庭停留，且竞品存在可解释的体验缺口", stopLine: "客流以单人快速消费为主，或正餐竞品已经高度饱和"
    }
  ].map((item) => ({
    ...item,
    weight: item.title === "现制茶饮 / 咖啡"
      ? (env["学校/大学"] || 0) * 2 + (env["写字楼/公司"] || 0) * 2
      : item.title === "快餐 / 简餐"
        ? (env["写字楼/公司"] || 0) * 2 + (env["住宅小区"] || 0)
        : item.title === "小吃 / 夜宵"
          ? (env["学校/大学"] || 0) + (env["住宅小区"] || 0)
          : item.title === "社区生鲜 / 便利"
            ? (env["住宅小区"] || 0) * 2
            : (env["商场/超市"] || 0) + (env["住宅小区"] || 0)
  })).sort((a, b) => b.weight - a.weight);
  const signalLabels = geo.environment.filter((group) => group.count > 0)
    .sort((a, b) => b.count - a.count).slice(0, 3).map((group) => group.label);
  const signalText = signalLabels.length ? signalLabels.join("、") : "有限的周边环境信号";

  let options;
  if (reportType === "recommend") {
    options = scored.slice(0, 3).map((item, index) => ({
      rank: ["A", "B", "C"][index],
      title: item.title,
      why: item.why,
      rankReason: index === 0
        ? `${signalText}是当前最强的环境线索；在同类餐饮竞争较高时，它对这组线索的依赖最直接，因此排第一。`
        : index === 1
          ? `它也能承接${signalText}，但对高峰时段、出餐效率或既有竞品缺口的依赖更强，所以排在首选之后。`
          : `它只在特定时段或特定停留场景成立；地图提供了可能性，但需要比前两项更多现场证据，所以排第三。`,
      competitionReason: item.competitionReason,
      operatingRequirement: item.operatingRequirement,
      risk: item.risk,
      action: item.action,
      budgetCap: item.budgetCap, durationDays: item.durationDays, metric: item.metric,
      successLine: item.successLine, stopLine: item.stopLine
    }));
  } else {
    options = [
      { title: `到现场核对${category || "该品类"}的客群与竞争`, why: "地图只能看密度，客流与消费力必须现场确认", action: "工作日与周末各选高峰时段，现场记录人流结构、竞品排队与客单。", budgetCap: 0, durationDays: 2, metric: "高峰人流与竞品排队", successLine: "目标客群与判断一致且竞品未过度饱和", stopLine: "现场人流与客群明显不足则不开" },
      { title: "低成本试卖验证真实需求", why: "把“能不能开”变成一次可撤回的小实验", action: "用最小投入摆点或快闪，测试真实下单与复购意愿。", budgetCap: 800, durationDays: 3, metric: "试卖有效订单与复购", successLine: "试卖达到预设订单线", stopLine: "试卖远低于预设线则止损" }
    ];
  }

  const headline = reportType === "recommend"
    ? "这个位置更适合开什么"
    : `这个位置能不能开${category || "这个品类"}`;
  const reason = dataDegraded
    ? "地图服务当前不可用，不能把“未读取到竞品”当成周边没有竞争；先恢复地图或到现场核对，再做开店判断。"
    : reportType === "recommend"
      ? `周边800米约${competitors}个同类，客群信号${hasCrowd ? "存在但仍需核实" : "偏弱"}；排序只说明相对匹配度，不等于可以直接签约。`
      : `周边800米约${competitors}个同类竞品，客群信号${hasCrowd ? "较清晰" : "偏弱"}；据此给出能否开的初判，结论需现场验证。`;

  const diagnosis = reportType === "recommend"
    ? dataDegraded
      ? "当前地图服务没有返回可用的周边证据，因此不生成“竞争少”或“客群清晰”的结论。先恢复地图数据，或到现场记录竞品、客群与高峰，再决定是否继续。"
      : `这不是“地图上看着热闹就能开”的结论。当前主要环境线索来自${signalText}，但800米内已有约${competitors}个同类。首选排在前面，是因为它更能承接现有环境线索、较少依赖与成熟餐饮正面硬碰；次选和第三选择分别需要验证高峰与时段场景。三项都必须先用现场观察和低成本试卖证伪。`
    : reason;
  return { decision, title: headline, reason, headline, diagnosis, rankingNarrative: diagnosis, options };
}

function normalizeSiteOptions(options, reportType) {
  const list = Array.isArray(options) ? options.slice(0, 3) : [];
  return list.map((option, index) => ({
    id: `site-${index + 1}`,
    title: cleanText(option?.title, 60) || (reportType === "recommend" ? `推荐品类 ${index + 1}` : `落地步骤 ${index + 1}`),
    score: Math.max(70, 92 - index * 5),
    bottleneck: reportType === "recommend"
      ? `推荐${cleanText(option?.rank, 2) || ["A", "B", "C"][index] || ""}`.trim()
      : "落地验证",
    action: trimText(option?.action || option?.why, 400) || "先现场核对，再低成本验证。",
    mechanism: trimText(option?.why, 300) || "基于周边客群与竞争的环境证据。",
    rankReason: trimText(option?.rankReason || option?.rank_reason, 300) || "按客群匹配度、竞争强度和经营前提的相对成立程度排序。",
    competitionReason: trimText(option?.competitionReason || option?.competition_reason, 300) || "地图只能显示周边供给，必须现场核对实际价格、排队与空桌。",
    operatingRequirement: trimText(option?.operatingRequirement || option?.operating_requirement, 300) || "先把产品、出餐和门店动线做成可被验证的最小模型。",
    risk: trimText(option?.risk, 300) || "若现场客群或竞争与地图信号不一致，应停止追加投入。",
    budgetCap: Number(option?.budgetCap) || 0,
    durationDays: Number(option?.durationDays) || 3,
    metric: cleanText(option?.metric, 60) || "现场验证指标",
    successLine: trimText(option?.successLine, 200) || "达到预设验证线",
    stopLine: trimText(option?.stopLine, 200) || "未达预设线即停止",
    detail_markdown: option?.why
      ? `**为什么是它**：${trimText(option.why, 400)}\n\n**为什么排在这个位置**：${trimText(option?.rankReason || option?.rank_reason, 400)}\n\n**竞争怎么判断**：${trimText(option?.competitionReason || option?.competition_reason, 400)}\n\n**先做什么**：${trimText(option?.action, 400) || "到现场核对客群与竞争，再低成本试卖验证。"}`
      : undefined
  }));
}

export async function runSiteReport(record, env, body) {
  const category = cleanText(body?.category, 30) || cleanText(record.category, 30) || "我不知道";
  const reportType = category === "我不知道" ? "recommend" : "feasibility";
  const geo = siteGeoSummary(record);
  const llm = createTextLlm(env);

  let core = fallbackSiteReport(geo, reportType, category);
  // The model may improve the wording, but it must never erase the concrete
  // per-category evidence / validation differences supplied by the rule-based
  // report.  Otherwise three different headings collapse into three copies of
  // the same generic "go and observe" card.
  const mergeOptionWithFallback = (fallbackOption, modelOption) => {
    const supplied = Object.fromEntries(Object.entries(modelOption || {}).filter(([, value]) => {
      if (typeof value === "string") return value.trim().length > 0;
      return value !== null && value !== undefined;
    }));
    return { ...fallbackOption, ...supplied };
  };
  if (llm) {
    try {
      // Keep the on-site experience snappy: if the model is slow, fall back to
      // the deterministic geo report instead of hanging past the client timeout.
      const llmTimeout = new Promise((_, reject) => setTimeout(() => reject(new Error("site-report-llm-timeout")), 22000));
      const response = await Promise.race([llm([
        {
          role: "system",
          content: `你是勇哥选址判断器。${YONGGE_SITE_FRAMEWORK} 只依据给定地理信息推理，禁止臆造具体人流、营业额或租金数字。reportType=recommend 时给出3个推荐品类并用 rank A>B>C 排序，每个附客群与竞争理由；reportType=feasibility 时判断给定品类能不能开，options 给出2-3条现场核对与低成本验证步骤。全中文，只返回JSON。`
        },
        {
          role: "user",
          content: JSON.stringify({
            reportType,
            category,
            geo,
            output_schema: {
              decision: "GO|TEST|STOP",
              title: "一句话结论",
              reason: "为什么这样判断（基于地理客群与竞争）",
              headline: "结果页大标题",
              diagnosis: "120-260字总体排序解读，说明客群来源、竞争密度与环境冲突，并提示需现场验证",
              rankingNarrative: "120-260字，直接解释A为什么排B前、B为什么排C前；不能把POI数量说成人流",
              options: [{
                rank: "recommend用A/B/C；feasibility可省略",
                title: "string",
                why: "这个品类为什么适合（客群+消费场景）",
                rankReason: "为什么排在这个位置，必须和其他两个比较",
                competitionReason: "要避开或面对什么竞争，不能把POI数量当客流",
                operatingRequirement: "这个品类成立前，门店/产品/时段必须满足什么",
                risk: "最大反例或失败条件",
                action: "现场核对或低成本验证动作",
                budgetCap: "number 元",
                durationDays: "number 天",
                metric: "观测指标",
                successLine: "成功线",
                stopLine: "停止线"
              }]
            }
          })
        }
      ], { temperature: 0.3, maxTokens: 1600 }), llmTimeout]);
      const decision = cleanText(response?.decision, 10).toUpperCase();
      const minimumOptions = reportType === "recommend" ? 3 : 1;
      if (["GO", "TEST", "STOP"].includes(decision) && Array.isArray(response?.options) && response.options.length >= minimumOptions) {
        core = {
          decision,
          title: cleanText(response.title, 160) || core.title,
          reason: trimText(response.reason, 500) || core.reason,
          headline: cleanText(response.headline, 160) || core.headline,
          diagnosis: trimText(response.diagnosis, 600) || core.diagnosis,
          rankingNarrative: trimText(response.rankingNarrative, 700) || trimText(response.diagnosis, 600) || core.rankingNarrative,
          options: core.options.map((fallbackOption, index) => mergeOptionWithFallback(fallbackOption, response.options[index]))
        };
      }
    } catch (_) {
      // Keep the deterministic fallback report on any LLM failure.
    }
  }

  const topPlans = normalizeSiteOptions(core.options, reportType);
  const siteMetrics = siteMetricsFromGeo(geo);
  const decisionLabel = { GO: "值得开", TEST: "先小成本验证", STOP: "不建议开" }[core.decision] || "先小成本验证";
  return {
    reportMode: "site-map",
    reportType,
    category,
    deterministic: {
      decision: core.decision,
      title: core.title,
      reason: core.reason,
      metrics: {}
    },
    siteMetrics,
    geo: { address: geo.address, city: geo.city, district: geo.district, competitorCount: geo.competitorCount, competitorKeyword: geo.competitorKeyword, environment: geo.environment, dataQuality: geo.dataQuality },
    narrative: { title: core.headline, body: core.diagnosis },
    rankingNarrative: core.rankingNarrative || core.diagnosis,
    explanation: {
      headline: core.headline,
      diagnosis: core.diagnosis,
      whyThesePlans: topPlans.map((plan) => `${plan.bottleneck}：${plan.mechanism}`),
      caution: `${decisionLabel}——地图只提供环境证据，正式投入前必须现场核对客群与竞争。${geo.dataQuality?.status === "degraded" ? ` 当前地图状态：${geo.dataQuality.reason || "部分服务不可用"}。` : ""}`
    },
    topPlans,
    candidateCount: topPlans.length,
    verified: topPlans.length,
    rejectedReasons: []
  };
}

function attachPlanDetails(result) {
  const details = Array.isArray(result?.explanation?.planDetails)
    ? result.explanation.planDetails
    : [];
  if (!details.length || !Array.isArray(result?.top3)) return;
  const byId = new Map(details.map((item) => [cleanText(item?.id, 80), trimText(item?.markdown, 2600)]));
  result.top3.forEach((plan) => {
    const markdown = byId.get(cleanText(plan?.id, 80));
    if (markdown) plan.detail_markdown = markdown;
  });
}

// Persist mid-round phase transitions so polling clients watch real progress
// instead of a stale "queued" snapshot until the whole round finishes.
// Throttled to phase changes (or one write per 4s) and serialized through a
// chain so a slower write can never overwrite a newer snapshot.
function createRunProgressSink(env, run) {
  let lastPhase = "";
  let lastWriteAt = 0;
  let chain = Promise.resolve();
  return (progress) => {
    run.progress = progress;
    run.updatedAt = nowIso();
    const phase = String(progress?.phase || "");
    const nowMs = Date.now();
    if (phase === lastPhase && nowMs - lastWriteAt < 4000) return;
    lastPhase = phase;
    lastWriteAt = nowMs;
    chain = chain.then(() => persistRun(env, run)).catch(() => {});
  };
}

async function runAnalysis(record, run, env, body) {
  const context = analysisContext(record, body);
  const llm = createTextLlm(env);
  try {
    const result = await runAgentSearch(context, {
      llm,
      concurrency: SEARCH_CONCURRENCY,
      target: SEARCH_TARGET,
      maxAttempts: SEARCH_TARGET,
      onProgress: createRunProgressSink(env, run)
    });
    result.deterministic = {
      decision: context.decision,
      title: context.title,
      reason: context.reason,
      metrics: context.metrics
    };
    result.explanation = {
      headline: context.title || "先看证据，再做下一步",
      diagnosis: context.reason || "系统已根据确认事实完成确定性判断。",
      whyThesePlans: result.top3.map((plan) => `${plan.bottleneck}：${plan.mechanism}`),
      caution: "所有方案都必须按预算、成功线和停止线执行。"
    };
    if (llm && result.top3.length) {
      try {
        result.explanation = await llm([
          {
            role: "system",
            content: "你是经营结果解释器。只能解释给定的确定性判断和Top方案，不得修改数字、排名、风险、成功线或停止线，不得新增方案。只返回JSON。"
          },
          {
            role: "user",
            content: JSON.stringify({
              facts: context.facts,
              deterministic_result: {
                decision: context.decision,
                title: context.title,
                reason: context.reason,
                metrics: context.metrics
              },
              top_plans: result.top3,
              output_schema: {
                headline: "string",
                diagnosis: "string",
                evidence: ["string"],
                unknowns: ["string"],
                whyThesePlans: ["string"],
                caution: "string",
                planDetails: [{
                  id: "must exactly equal one top_plans id",
                  markdown: "Markdown. Explain this existing plan only: why first, exact action, evidence boundary, budget/duration/metric/success/stop and fastest falsification. Do not alter any supplied value."
                }]
              }
            })
          }
        ], { temperature: 0.15, maxTokens: 1500 });
        attachPlanDetails(result);
      } catch (error) {
        run.warning = `方案已完成，但AI解释生成失败：${error instanceof Error ? error.message : "unknown"}`;
      }
    }
    run.status = "completed";
    run.result = result;
    run.progress = { phase: "completed", completed: result.generated, target: SEARCH_TARGET };
  } catch (error) {
    const result = await runAgentSearch(context, { llm: null, target: SEARCH_TARGET });
    result.deterministic = {
      decision: context.decision,
      title: context.title,
      reason: context.reason,
      metrics: context.metrics
    };
    run.status = "completed";
    run.result = result;
    run.warning = error instanceof Error ? error.message : "Agent分析失败，已使用确定性降级";
  }
  run.updatedAt = nowIso();
  await persistRun(env, run);
}

async function finishQueuedAnalysis(run, env, llm) {
  const context = run.context;
  const result = finalizeAgentSearch(context, run.searchState);
  result.deterministic = {
    decision: context.decision,
    title: context.title,
    reason: context.reason,
    metrics: context.metrics
  };
  result.explanation = {
    headline: context.title || "先看证据，再做下一步",
    diagnosis: context.reason || "系统已根据确认事实完成确定性判断。",
    whyThesePlans: result.top3.map((plan) => `${plan.bottleneck}：${plan.mechanism}`),
    caution: "所有方案都必须按预算、成功线和停止线执行。"
  };
  if (llm && result.top3.length) {
    try {
      result.explanation = await llm([
        {
          role: "system",
          content: "你是经营结果解释器。只能解释给定的确定性判断和Top方案，不得修改数字、排名、风险、成功线或停止线，不得新增方案。只返回JSON。"
        },
        {
          role: "user",
          content: JSON.stringify({
            facts: context.facts,
            deterministic_result: result.deterministic,
            top_plans: result.top3,
            output_schema: {
              headline: "string", diagnosis: "string", evidence: ["string"],
              unknowns: ["string"], whyThesePlans: ["string"], caution: "string",
              planDetails: [{
                id: "must exactly equal one top_plans id",
                markdown: "Markdown. Explain this existing plan only: why first, exact action, evidence boundary, budget/duration/metric/success/stop and fastest falsification. Do not alter any supplied value."
              }]
            }
          })
        }
      ], { temperature: 0.15, maxTokens: 1500 });
      attachPlanDetails(result);
    } catch (error) {
      run.warning = `方案已完成，但AI解释生成失败：${error instanceof Error ? error.message : "unknown"}`;
    }
  }
  run.status = "completed";
  run.result = result;
  run.progress = { phase: "completed", completed: result.generated, target: SEARCH_TARGET };
  run.updatedAt = nowIso();
  await persistRun(env, run);
}

export async function processAnalysisQueueMessage(message, env) {
  const runId = cleanId(message?.runId, 100);
  const requestedRound = Number(message?.round);
  const run = await loadRun(env, runId);
  if (!run || ["completed", "failed"].includes(run.status)) return { skipped: "terminal" };
  const currentRound = run.searchState?.round || 0;
  if (requestedRound <= currentRound) {
    // If the worker crashed after persisting a round but before publishing the
    // next one, a redelivery repairs the chain. Duplicate next-round messages
    // are harmless because that round is protected by the atomic D1 claim.
    if (requestedRound === currentRound && currentRound < SEARCH_ROUNDS &&
        run.searchState?.audited?.length < SEARCH_TARGET && env.ANALYSIS_QUEUE) {
      await env.ANALYSIS_QUEUE.send({ runId, round: currentRound + 1 });
    }
    return { skipped: "already-completed" };
  }
  if (requestedRound !== currentRound + 1) throw new Error("队列轮次乱序");

  const claimToken = await claimAnalysisRound(env, run, requestedRound);
  if (!claimToken) {
    const expiresAt = new Date(run.claimExpiresAt || 0).getTime();
    const retryAfterSeconds = Number.isFinite(expiresAt) && expiresAt > Date.now()
      ? Math.max(5, Math.min(600, Math.ceil((expiresAt - Date.now()) / 1000) + 1))
      : 30;
    return { skipped: "claimed", retryAfterSeconds };
  }
  let enqueueRound = 0;
  try {
    const llm = createTextLlm(env);
    if (!llm) {
      await finishQueuedAnalysis(run, env, null);
      return { completed: true, fallback: true };
    }
    run.status = "running";
    run.progress = { phase: "round-start", round: requestedRound, completed: run.searchState.audited.length, target: SEARCH_TARGET };
    run.updatedAt = nowIso();
    await persistRun(env, run);

    run.searchState = await runAgentRound(run.context, run.searchState, {
      llm,
      concurrency: SEARCH_CONCURRENCY,
      onProgress: createRunProgressSink(env, run)
    });
    run.updatedAt = nowIso();
    await persistRun(env, run);

    if (run.searchState.round >= SEARCH_ROUNDS || run.searchState.audited.length >= SEARCH_TARGET) {
      await finishQueuedAnalysis(run, env, llm);
      return { completed: true };
    }
    run.status = "queued";
    run.progress = {
      phase: "round-complete",
      round: run.searchState.round,
      completed: run.searchState.audited.length,
      target: SEARCH_TARGET
    };
    await persistRun(env, run);
    enqueueRound = run.searchState.round + 1;
  } finally {
    // Release before enqueuing the next round. Otherwise a fast consumer could
    // observe the previous round's still-live lease and incorrectly ack it.
    await releaseAnalysisRoundClaim(env, run, requestedRound, claimToken);
  }
  if (enqueueRound) {
    await env.ANALYSIS_QUEUE.send({ runId, round: enqueueRound });
  }
  return { queuedRound: enqueueRound };
}

async function clearAnalysisClaim(env, run) {
  if (env.DB) {
    await env.DB.prepare(`
      UPDATE analysis_runs
      SET claim_token = NULL, claimed_round = NULL, claim_expires_at = NULL
      WHERE id = ?
    `).bind(run.id).run();
  } else {
    run.queueClaim = null;
  }
  run.claimToken = null;
  run.claimedRound = null;
  run.claimExpiresAt = null;
}

export async function enqueueAnalysisRound(env, run, round) {
  run.status = "queued";
  run.warning = "";
  run.progress = {
    phase: "queued",
    round,
    completed: run.searchState?.audited?.length || 0,
    target: SEARCH_TARGET
  };
  run.updatedAt = nowIso();
  await persistRun(env, run);
  try {
    await env.ANALYSIS_QUEUE.send({ runId: run.id, round });
    return { queued: true };
  } catch (error) {
    run.status = "failed";
    run.warning = `分析任务未能进入队列：${error instanceof Error ? error.message : "unknown"}`;
    run.progress = { ...run.progress, phase: "enqueue-failed" };
    run.updatedAt = nowIso();
    await persistRun(env, run);
    return { queued: false, error };
  }
}

async function startAnalysis(request, env, caseId, ctx) {
  const record = await requireCase(request, env, caseId);
  const body = await readJson(request);
  // Preopen site-report mode skips the financial interview entirely and returns
  // a map-driven "can I open here" report in one synchronous LLM pass.
  if (body.mode === "site-map") {
    const result = await runSiteReport(record, env, body);
    // Persist the selection (confirmed location + chosen category / "我不知道")
    // and the generated report as a completed run, so this site case is saved to
    // D1 and can be pulled back later exactly like an interview analysis. Reuse an
    // existing run id for this case version so re-generating updates in place
    // instead of tripping the unique (case_id, case_version) constraint.
    const existingRun = await findRunForCaseVersion(env, record.id, record.version);
    const runId = existingRun?.id || secureId("run");
    const timestamp = nowIso();
    const run = {
      id: runId,
      caseId: record.id,
      caseVersion: record.version,
      status: "completed",
      progress: { phase: "complete", completed: 1, target: 1 },
      result,
      context: null,
      searchState: null,
      warning: "",
      createdAt: existingRun?.createdAt || timestamp,
      updatedAt: timestamp
    };
    runStore.set(runId, run);
    record.category = result.category || record.category;
    record.facts.category = normalizeFact({
      id: "category", value: result.category, status: "confirmed", source: "user", evidence: "A"
    }, "user");
    record.latestRunId = runId;
    record.updatedAt = timestamp;
    await persistCase(env, record);
    await persistRun(env, run);
    return apiJson(request, env, { status: "complete", runId, result });
  }
  if (body.caseVersion != null && !Number.isInteger(Number(body.caseVersion))) {
    return apiJson(request, env, {
      code: "CASE_VERSION_INVALID",
      message: "案卷版本格式无效",
      version: record.version,
      facts: Object.values(record.facts)
    }, 422);
  }
  if (body.caseVersion != null && Number(body.caseVersion) !== Number(record.version)) {
    return apiJson(request, env, {
      code: "CASE_VERSION_CONFLICT",
      message: "事实档案已更新，请按最新版本重新分析",
      version: record.version,
      facts: Object.values(record.facts)
    }, 409);
  }
  const previous = await findRunForCaseVersion(env, record.id, record.version);
  if (previous?.status === "completed") {
    return apiJson(request, env, {
      runId: previous.id,
      status: previous.status,
      reused: true
    });
  }
  if (previous?.status === "failed") {
    if (body.retryFailed !== true) {
      return apiJson(request, env, {
        code: "ANALYSIS_PREVIOUSLY_FAILED",
        runId: previous.id,
        status: previous.status,
        retryable: true,
        message: "上次分析失败；确认后可用 retryFailed=true 从已保存进度重试"
      }, 409);
    }
    await clearAnalysisClaim(env, previous);
    const currentRound = Number(previous.searchState?.round) || 0;
    if (currentRound >= SEARCH_ROUNDS || (previous.searchState?.audited?.length || 0) >= SEARCH_TARGET) {
      previous.status = "running";
      previous.updatedAt = nowIso();
      await persistRun(env, previous);
      await finishQueuedAnalysis(previous, env, createTextLlm(env));
      return apiJson(request, env, {
        runId: previous.id,
        status: previous.status,
        retried: true,
        reusedProgress: true
      }, previous.status === "completed" ? 200 : 202);
    }
    if (env.ANALYSIS_QUEUE) {
      const queued = await enqueueAnalysisRound(env, previous, currentRound + 1);
      if (!queued.queued) {
        return apiJson(request, env, {
          code: "ANALYSIS_ENQUEUE_FAILED",
          runId: previous.id,
          status: "failed",
          retryable: true,
          message: previous.warning
        }, 503);
      }
      return apiJson(request, env, {
        runId: previous.id,
        status: previous.status,
        retried: true,
        reusedProgress: true,
        queued: true
      }, 202);
    }
    await runAnalysis(record, previous, env, body);
    return apiJson(request, env, {
      runId: previous.id,
      status: previous.status,
      retried: true,
      reusedProgress: true
    });
  }
  if (previous && ["queued", "running"].includes(previous.status)) {
    return apiJson(request, env, {
      code: "ANALYSIS_ALREADY_RUNNING",
      runId: previous.id,
      status: previous.status
    }, 409);
  }
  const id = secureId("run");
  const timestamp = nowIso();
  const run = {
    id,
    caseId,
    caseVersion: record.version,
    status: "running",
    progress: { phase: "queued", completed: 0, target: SEARCH_TARGET },
    result: null,
    context: analysisContext(record, body),
    searchState: createSearchState({ target: SEARCH_TARGET, maxAttempts: SEARCH_TARGET }),
    createdAt: timestamp,
    updatedAt: timestamp
  };
  if (env.DB) {
    const inserted = await insertRunIfAbsent(env, run);
    if (!inserted) {
      const raced = await findRunForCaseVersion(env, record.id, record.version);
      return apiJson(request, env, {
        code: raced?.status === "completed" ? undefined : "ANALYSIS_ALREADY_EXISTS",
        runId: raced?.id,
        status: raced?.status,
        reused: raced?.status === "completed"
      }, raced?.status === "completed" ? 200 : 409);
    }
  }
  runStore.set(id, run);
  record.latestRunId = id;
  record.updatedAt = timestamp;
  await persistCase(env, record);
  if (!env.DB) await persistRun(env, run);

  if (env.ANALYSIS_QUEUE) {
    if (!env.DB) {
      run.status = "failed";
      run.warning = "生产队列需要D1绑定保存累计搜索状态";
      await persistRun(env, run);
      return apiJson(request, env, { code: "DB_REQUIRED", message: run.warning }, 503);
    }
    run.status = "queued";
    const queued = await enqueueAnalysisRound(env, run, 1);
    if (!queued.queued) {
      return apiJson(request, env, {
        code: "ANALYSIS_ENQUEUE_FAILED",
        runId: run.id,
        status: "failed",
        retryable: true,
        message: run.warning
      }, 503);
    }
  } else {
    // Local/test fallback only. Production config always binds ANALYSIS_QUEUE.
    await runAnalysis(record, run, env, body);
  }
  return apiJson(request, env, {
    runId: id,
    status: run.status,
    progress: run.progress,
    queued: run.status === "queued"
  }, 202);
}

async function getRun(request, env, caseId, runId, asEvents = false) {
  const record = await requireCase(request, env, caseId);
  const run = await loadRun(env, runId);
  if (!run || run.caseId !== record.id) {
    return apiJson(request, env, { code: "RUN_NOT_FOUND", message: "分析任务不存在" }, 404);
  }
  const stale = run.caseVersion !== record.version;
  const publicRun = publicRunSnapshot(run);
  const payload = { ...publicRun, stale };
  if (!asEvents) return apiJson(request, env, payload);
  const eventPayload = run.status === "completed"
    ? { type: "complete", result: run.result, progress: 100, stale }
    : { type: "progress", status: run.status, progress: run.progress, stale };
  const body = `data: ${JSON.stringify(eventPayload)}\n\n`;
  return new Response(body, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-store",
      "Connection": "keep-alive",
      ...corsHeaders(request, env)
    }
  });
}

export function publicRunSnapshot(run) {
  const {
    context: _context,
    searchState: _searchState,
    claimToken: _claimToken,
    claimedRound: _claimedRound,
    claimExpiresAt: _claimExpiresAt,
    queueClaim: _queueClaim,
    ...publicRun
  } = run || {};
  return publicRun;
}

async function startPlan(request, env, caseId, planId) {
  const record = await requireCase(request, env, caseId);
  const run = await loadRun(env, record.latestRunId);
  if (!run || run.status !== "completed" || run.caseVersion !== record.version) {
    return apiJson(request, env, {
      code: "CURRENT_ANALYSIS_REQUIRED",
      message: "请先完成当前版本的分析"
    }, 409);
  }
  const plan = run.result?.top3?.find((item) => item.id === planId);
  if (!plan) return apiJson(request, env, { code: "PLAN_NOT_FOUND", message: "方案不存在" }, 404);
  record.selectedPlanId = planId;
  record.updatedAt = nowIso();
  await persistCase(env, record);
  return apiJson(request, env, {
    selectedPlanId: planId,
    checklist: [
      `准备预算上限：${plan.budget_cap} 元`,
      `执行动作：${plan.action}`,
      `连续记录指标：${plan.metric}`,
      `达到成功线：${plan.success_line}`,
      `触发停止线：${plan.stop_line}`
    ],
    reviewAfterDays: plan.duration_days
  });
}

async function consumeTtsQuota(env, record) {
  if (!env.DB) {
    if ((Number(record.ttsCount) || 0) >= MAX_TTS_PER_CASE) return false;
    record.ttsCount = (Number(record.ttsCount) || 0) + 1;
    return true;
  }
  const result = await env.DB.prepare(`
    UPDATE cases
    SET tts_count = tts_count + 1, updated_at = ?
    WHERE id = ? AND tts_count < ?
  `).bind(nowIso(), record.id, MAX_TTS_PER_CASE).run();
  const allowed = Number(result?.meta?.changes ?? 0) === 1;
  if (allowed) record.ttsCount = (Number(record.ttsCount) || 0) + 1;
  return allowed;
}

async function ttsResponse(request, env) {
  const body = await readJson(request);
  const caseId = cleanId(body.caseId, 80);
  if (!caseId) {
    return apiJson(request, env, {
      code: "CASE_REQUIRED",
      message: "语音合成必须绑定有效案卷"
    }, 400);
  }
  const record = await requireCase(request, env, caseId);
  const client = new DashScopeTtsClient(env);
  if (!client.configured) {
    return apiJson(request, env, { code: "DASHSCOPE_TTS_NOT_CONFIGURED", message: "语音服务尚未配置" }, 503);
  }
  if (!await consumeTtsQuota(env, record)) {
    return apiJson(request, env, {
      code: "TTS_QUOTA_EXCEEDED",
      message: `每个案卷最多合成${MAX_TTS_PER_CASE}次语音`
    }, 429);
  }
  const audio = await client.synthesize(body.text, {
    voice: cleanText(body.voice, 40) || undefined
  });
  const headers = new Headers({ "Content-Type": audio.contentType });
  headers.set("Cache-Control", "private, no-store");
  Object.entries(corsHeaders(request, env)).forEach(([key, value]) => headers.set(key, value));
  return new Response(audio.body, { status: 200, headers });
}

function isWavFile(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 46) return false;
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const chunkId = (offset) => String.fromCharCode(
    bytes[offset],
    bytes[offset + 1],
    bytes[offset + 2],
    bytes[offset + 3]
  );
  if (chunkId(0) !== "RIFF" || chunkId(8) !== "WAVE") return false;
  // The RIFF length must describe the complete upload; accepting a forged
  // 12-byte prefix would let arbitrary data pass the former check.
  if (view.getUint32(4, true) + 8 !== buffer.byteLength) return false;

  let offset = 12;
  let hasPcmFormat = false;
  let hasAudioData = false;
  while (offset + 8 <= buffer.byteLength) {
    const id = chunkId(offset);
    const size = view.getUint32(offset + 4, true);
    const payloadOffset = offset + 8;
    const payloadEnd = payloadOffset + size;
    if (payloadEnd > buffer.byteLength) return false;

    if (id === "fmt ") {
      // Only canonical PCM generated by the browser is accepted. In
      // particular, reject float/extensible WAV, stereo, resampled audio, and
      // internally inconsistent byte-rate/block-alignment headers.
      if (hasPcmFormat || size !== 16) return false;
      const audioFormat = view.getUint16(payloadOffset, true);
      const channels = view.getUint16(payloadOffset + 2, true);
      const sampleRate = view.getUint32(payloadOffset + 4, true);
      const byteRate = view.getUint32(payloadOffset + 8, true);
      const blockAlign = view.getUint16(payloadOffset + 12, true);
      const bitsPerSample = view.getUint16(payloadOffset + 14, true);
      hasPcmFormat = audioFormat === 1
        && channels === 1
        && sampleRate === 16_000
        && bitsPerSample === 16
        && blockAlign === 2
        && byteRate === 32_000;
      if (!hasPcmFormat) return false;
    } else if (id === "data") {
      if (!hasPcmFormat || hasAudioData || size < 2 || size % 2 !== 0) return false;
      hasAudioData = true;
    }

    // RIFF chunks are word-aligned; a declared pad byte must also be present.
    offset = payloadEnd + (size % 2);
  }
  return offset === buffer.byteLength && hasPcmFormat && hasAudioData;
}

async function dashScopeAsrResponse(request, env, caseId) {
  const record = await requireCase(request, env, caseId);
  const contentType = String(request.headers.get("Content-Type") || "").toLowerCase();
  if (!contentType.startsWith("audio/wav") && !contentType.startsWith("audio/x-wav")) {
    return apiJson(request, env, {
      code: "ASR_AUDIO_FORMAT_INVALID",
      message: "语音识别只接受16kHz单声道WAV片段"
    }, 415);
  }
  const declaredLength = Number(request.headers.get("Content-Length") || 0);
  if (declaredLength > MAX_ASR_CLIP_BYTES) {
    return apiJson(request, env, {
      code: "ASR_AUDIO_TOO_LARGE",
      message: "单次回答最长约60秒，请缩短后再试"
    }, 413);
  }
  const audio = await request.arrayBuffer();
  if (audio.byteLength > MAX_ASR_CLIP_BYTES) {
    return apiJson(request, env, {
      code: "ASR_AUDIO_TOO_LARGE",
      message: "单次回答最长约60秒，请缩短后再试"
    }, 413);
  }
  if (!isWavFile(audio)) {
    return apiJson(request, env, {
      code: "ASR_AUDIO_INVALID",
      message: "没有收到有效WAV语音片段"
    }, 422);
  }

  const client = new DashScopeAsrClient(env);
  if (!client.configured) {
    return apiJson(request, env, {
      code: "DASHSCOPE_NOT_CONFIGURED",
      message: "阿里云语音识别尚未配置"
    }, 503);
  }
  const prior = record.turns
    .slice(-3)
    .map((turn) => trimText(turn.transcript, 80))
    .filter(Boolean)
    .join("；");
  const context = [
    "餐饮经营词表：营业额、流水、毛利率、变动成本、客单价、房租、人工、现金、债务、客流、门头、转让费、加盟费。",
    `当前问题：${trimText(record.currentQuestion?.text, 100)}`,
    prior ? `前文：${prior}` : ""
  ].filter(Boolean).join("\n").slice(0, 380);

  try {
    const result = await client.transcribe(audio, { context, timeoutMs: 25_000 });
    return apiJson(request, env, {
      text: result.text,
      requestId: result.requestId,
      model: result.model
    });
  } catch (error) {
    return apiJson(request, env, {
      code: "DASHSCOPE_ASR_FAILED",
      message: error instanceof Error ? error.message : "阿里云语音识别失败"
    }, 502);
  }
}

export function asrSessionConfig(env = {}) {
  return {
    type: "session.update",
    session: {
      audio: {
        input: {
          format: {
            type: "pcm",
            codec: "pcm_s16le",
            rate: 16000,
            bits: 16,
            channel: 1
          },
          transcription: {
            model: env.STEPFUN_ASR_MODEL || "stepaudio-2.5-asr-stream",
            language: "zh",
            prompt: "餐饮经营、营业额、毛利率、房租、人工、客流、选址、加盟、转让",
            full_rerun_on_commit: true,
            enable_itn: true
          },
          turn_detection: {
            type: "server_vad",
            threshold: 0.5,
            silence_duration_ms: ASR_SILENCE_DURATION_MS
          }
        }
      }
    }
  };
}

export function normalizeAsrClientEvent(payload) {
  if (["session.update", "session.start", "answer.text"].includes(payload?.type)) return null;
  if (payload?.type === "session.finish") {
    return { event_id: secureId("event"), type: "input_audio_buffer.commit" };
  }
  if (payload?.type === "input_audio_buffer.append") {
    if (typeof payload.audio !== "string" || !payload.audio) return null;
    return {
      event_id: cleanId(payload.event_id, 100) || secureId("event"),
      type: "input_audio_buffer.append",
      audio: payload.audio
    };
  }
  return null;
}

async function acquireAsrSession(env, caseId) {
  const sessionId = secureId("asr");
  const key = await tokenHash(caseId);
  if (!env.AGENT_GATE) {
    const now = Date.now();
    const existing = localAsrSessions.get(key);
    const state = !existing || Number(existing.resetAt) <= now
      ? { resetAt: now + CASE_TTL_MS, sessions: 0, activeSessionId: "", activeExpiresAt: 0 }
      : existing;
    if (state.activeSessionId && state.activeExpiresAt > now) {
      return { allowed: false, reason: "active", retryAfterMs: state.activeExpiresAt - now };
    }
    if (state.sessions >= MAX_ASR_SESSIONS_PER_CASE) {
      return { allowed: false, reason: "session-quota", retryAfterMs: state.resetAt - now };
    }
    state.sessions += 1;
    state.activeSessionId = sessionId;
    state.activeExpiresAt = now + MAX_ASR_SESSION_MS;
    localAsrSessions.set(key, state);
    return { allowed: true, key, sessionId };
  }
  const id = env.AGENT_GATE.idFromName("global-stepfun-gate");
  const gate = env.AGENT_GATE.get(id);
  const response = await gate.fetch("https://agent-gate.internal/asr/acquire", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key, sessionId })
  });
  if (!response.ok) throw new Error("ASR quota gate unavailable");
  return { ...(await response.json()), key, sessionId };
}

async function releaseAsrSession(env, lease) {
  if (!lease?.key || !lease?.sessionId) return;
  if (!env.AGENT_GATE) {
    const state = localAsrSessions.get(lease.key);
    if (state?.activeSessionId === lease.sessionId) {
      state.activeSessionId = "";
      state.activeExpiresAt = 0;
      localAsrSessions.set(lease.key, state);
    }
    return;
  }
  const id = env.AGENT_GATE.idFromName("global-stepfun-gate");
  const gate = env.AGENT_GATE.get(id);
  await gate.fetch("https://agent-gate.internal/asr/release", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: lease.key, sessionId: lease.sessionId })
  });
}

function decodedBase64Bytes(value) {
  if (typeof value !== "string" || !value || value.length % 4 === 1) return -1;
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(value)) return -1;
  const padding = value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0;
  return Math.floor(value.length * 3 / 4) - padding;
}

async function asrRelay(request, env, caseId, ctx) {
  await requireCase(request, env, caseId);
  if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
    return apiJson(request, env, {
      code: "WEBSOCKET_REQUIRED",
      message: "持续语音问诊需要WebSocket连接"
    }, 426);
  }
  if (!globalThis.WebSocketPair) {
    return apiJson(request, env, {
      code: "ASR_RELAY_UNAVAILABLE",
      message: "当前运行环境不支持WebSocket中继；可改用浏览器语音识别或文字模式"
    }, 501);
  }
  const apiKey = String(env.STEPFUN_API_KEYS || env.STEPFUN_API_KEY || env.step_API_KEY || "")
    .split(/[\s,;]+/)
    .find(Boolean);
  if (!apiKey) {
    return apiJson(request, env, { code: "STEPFUN_NOT_CONFIGURED", message: "语音服务尚未配置" }, 503);
  }
  let lease;
  try {
    lease = await acquireAsrSession(env, caseId);
  } catch {
    return apiJson(request, env, {
      code: "ASR_QUOTA_GATE_UNAVAILABLE",
      message: "语音配额服务暂时不可用，请稍后再试"
    }, 503);
  }
  if (!lease.allowed) {
    return apiJson(request, env, {
      code: lease.reason === "active" ? "ASR_ALREADY_ACTIVE" : "ASR_SESSION_QUOTA_EXCEEDED",
      message: lease.reason === "active"
        ? "这个案卷已有一个语音连接"
        : `每个案卷最多建立${MAX_ASR_SESSIONS_PER_CASE}次语音会话`,
      retryAfterMs: lease.retryAfterMs || 0
    }, 429);
  }

  // Cloudflare Workers supports outgoing WebSockets through an Upgrade fetch.
  // This relay is intentionally transparent: audio frames are never persisted.
  const configuredUrl = env.STEPFUN_ASR_URL || "wss://api.stepfun.com/v1/realtime/asr/stream";
  const upstreamUrl = configuredUrl.replace(/^wss:/, "https:");
  let upstreamResponse;
  try {
    upstreamResponse = await fetch(upstreamUrl, {
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Upgrade": "websocket"
      }
    });
  } catch (error) {
    await releaseAsrSession(env, lease).catch(() => {});
    throw error;
  }
  const upstream = upstreamResponse.webSocket;
  if (!upstream) {
    await releaseAsrSession(env, lease).catch(() => {});
    return apiJson(request, env, {
      code: "ASR_UPSTREAM_UNAVAILABLE",
      message: `StepFun ASR连接失败：${upstreamResponse.status}`
    }, 502);
  }
  upstream.accept();
  const pair = new WebSocketPair();
  const [client, server] = Object.values(pair);
  server.accept();
  upstream.send(JSON.stringify(asrSessionConfig(env)));

  const startedAt = Date.now();
  let audioBytes = 0;
  let leaseReleased = false;
  const releaseLease = () => {
    if (leaseReleased) return;
    leaseReleased = true;
    const promise = releaseAsrSession(env, lease).catch(() => {});
    if (ctx?.waitUntil) ctx.waitUntil(promise);
  };
  const durationTimer = setTimeout(() => {
    closeBoth(1008, "session duration exceeded");
  }, MAX_ASR_SESSION_MS);
  const closeBoth = (code = 1011, reason = "relay closed") => {
    clearTimeout(durationTimer);
    releaseLease();
    try { server.close(code, reason); } catch {}
    try { upstream.close(code, reason); } catch {}
  };
  server.addEventListener("message", (event) => {
    if (typeof event.data === "string" && event.data.length > 64 * 1024) return closeBoth(1009, "message too large");
    if (typeof event.data === "string") {
      try {
        const payload = JSON.parse(event.data);
        const normalized = normalizeAsrClientEvent(payload);
        if (!normalized) return;
        if (normalized.type === "input_audio_buffer.append") {
          const chunkBytes = decodedBase64Bytes(normalized.audio);
          if (chunkBytes < 0) return closeBoth(1003, "invalid audio");
          audioBytes += chunkBytes;
          if (audioBytes > MAX_ASR_AUDIO_BYTES) {
            return closeBoth(1008, "audio byte quota exceeded");
          }
          if (Date.now() - startedAt > MAX_ASR_SESSION_MS) {
            return closeBoth(1008, "session duration exceeded");
          }
        }
        upstream.send(JSON.stringify(normalized));
        return;
      } catch {
        return closeBoth(1003, "invalid json");
      }
    }
    if (typeof event.data !== "string") return closeBoth(1003, "binary audio not accepted");
    upstream.send(event.data);
  });
  upstream.addEventListener("message", (event) => server.send(event.data));
  server.addEventListener("close", () => {
    clearTimeout(durationTimer);
    releaseLease();
    try { upstream.close(1000, "client closed"); } catch {}
  });
  upstream.addEventListener("close", () => {
    clearTimeout(durationTimer);
    releaseLease();
    try { server.close(1000, "upstream closed"); } catch {}
  });
  server.addEventListener("error", () => closeBoth());
  upstream.addEventListener("error", () => closeBoth());

  return new Response(null, { status: 101, webSocket: client });
}

const PUBLIC_METRICS = new Set(["revenue", "orders", "gross_margin", "cost", "cash_burn"]);
const LOWER_IS_BETTER = new Set(["cost", "cash_burn"]);

function publicDecisionLabel(decision) {
  return ({ GO: "可以继续", TEST: "小步验证", STOP: "停止追加", EXIT: "准备退出", EVIDENCE: "小步验证" })[decision] || "小步验证";
}

export function publicDataScore(record) {
  const core = evaluateInterviewCompleteness(interviewPolicyState(record)).requiredFields;
  const allowed = getAllowedInterviewFields(record.stage);
  const factScore = (field) => {
    const fact = record.facts?.[field];
    if (!fact || fact.status === "unknown" || fact.status === "conflict") return 0;
    return fact.status === "confirmed" ? 1 : .65;
  };
  const coreScore = core.length ? core.reduce((sum, field) => sum + factScore(field), 0) / core.length : 0;
  const optional = allowed.filter((field) => !core.includes(field));
  const optionalScore = optional.length ? optional.reduce((sum, field) => sum + factScore(field), 0) / optional.length : 0;
  return Math.round((coreScore * .7 + optionalScore * .3) * 100);
}

const PUBLIC_FACT_LABELS = {
  variableCostRate: "变动成本率", bottleneck: "当前瓶颈", growthBottleneck: "增长瓶颈",
  trialSale: "试销情况", trafficMatch: "客流匹配", visibility: "门店可见性", retention: "复购情况"
};

function safePublicFacts(record) {
  const fields = ["monthlyRevenue", "variableCostRate", "fixedCostTotal", "cashReserve", "debt", "bottleneck", "growthBottleneck", "trialSale", "trafficMatch", "visibility", "retention"];
  return fields.flatMap((field) => {
    const fact = record.facts?.[field];
    if (!fact || fact.status === "unknown" || fact.status === "conflict") return [];
    // Do not publish raw amounts. Share only non-reversible operating signals.
    if (["monthlyRevenue", "fixedCostTotal", "cashReserve", "debt"].includes(field)) return [];
    return [{ field, value: fact.value, range: fact.range || null, status: fact.status }];
  });
}

function verifiedPlans(run) {
  return (run?.result?.top3 || []).filter((plan) => plan?.verification?.passed !== false && plan?.verified !== false).slice(0, 2);
}

async function readPublicCases(env) {
  if (!env.DB) return [...publicCaseStore.values()].filter((row) => row.isActive).sort((a, b) => b.rankScore - a.rankScore);
  const result = await env.DB.prepare("SELECT * FROM public_cases WHERE is_active = 1 ORDER BY rank_score DESC, updated_at DESC LIMIT 100").all();
  return (result.results || []).map((row) => ({
    id: row.id, snapshot: JSON.parse(row.snapshot_json || "{}"), dataScore: Number(row.data_score),
    outcomeScore: Number(row.outcome_score), rankScore: Number(row.rank_score), isActive: Boolean(row.is_active), updatedAt: row.updated_at
  }));
}

async function readPublicCase(env, publicId) {
  if (!publicId) return null;
  if (!env.DB) {
    const row = publicCaseStore.get(publicId);
    return row?.isActive ? row : null;
  }
  const row = await env.DB.prepare("SELECT * FROM public_cases WHERE id = ? AND is_active = 1")
    .bind(publicId).first();
  if (!row) return null;
  return {
    id: row.id,
    snapshot: JSON.parse(row.snapshot_json || "{}"),
    dataScore: Number(row.data_score),
    outcomeScore: Number(row.outcome_score),
    rankScore: Number(row.rank_score),
    isActive: Boolean(row.is_active),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

async function leaderboard(request, env) {
  const rows = await readPublicCases(env);
  return apiJson(request, env, {
    cases: rows.map((row, index) => ({ rank: index + 1, id: row.id, ...row.snapshot, dataScore: row.dataScore, outcomeScore: row.outcomeScore, rankScore: row.rankScore, updatedAt: row.updatedAt }))
  });
}

async function publicCaseDetail(request, env, publicId) {
  const row = await readPublicCase(env, publicId);
  if (!row) {
    return apiJson(request, env, { code: "PUBLIC_CASE_NOT_FOUND", message: "该匿名案例不存在，或已被下架" }, 404);
  }
  // This is deliberately the same anonymised snapshot used by the leaderboard.
  // Never return source_case_id, token hashes, audio, raw transcripts or raw
  // financial facts from a share URL.
  return apiJson(request, env, {
    id: row.id,
    ...row.snapshot,
    dataScore: row.dataScore,
    outcomeScore: row.outcomeScore,
    rankScore: row.rankScore,
    createdAt: row.createdAt,
    updatedAt: row.updatedAt
  });
}

async function publishCase(request, env, caseId) {
  const record = await requireCase(request, env, caseId);
  const run = await loadRun(env, record.latestRunId);
  const decision = computeServerDecision(record);
  const dataScore = publicDataScore(record);
  const plans = verifiedPlans(run);
  if (!run || run.status !== "completed" || !plans.length || decision.decision === "EVIDENCE" || dataScore < 70) {
    return apiJson(request, env, { code: "PUBLICATION_NOT_READY", message: "核心事实、确定性结论和主方案核验完成后才会匿名公开" }, 422);
  }
  const id = secureId("public");
  const manageToken = secureId("manage");
  const timestamp = nowIso();
  const explanation = run?.result?.explanation || {};
  const rejectedRaw = Array.isArray(run?.result?.rejectedReasons)
    ? run.result.rejectedReasons
    : (run?.result?.rejected || []).map((item) => item?.reasons?.[0] || item?.phase).filter(Boolean);
  const snapshot = {
    stage: record.stage,
    category: cleanText(record.facts?.category?.value, 80) || "餐饮",
    // No location for real user cases: keep anonymised, address is never published.
    decision: decision.decision,
    conclusion: publicDecisionLabel(decision.decision),
    statusLine: trimText(explanation.headline || decision.title, 120),
    decisionTitle: trimText(decision.title, 160),
    decisionReason: trimText(decision.reason, 500),
    narrative: {
      title: trimText(explanation.headline, 120) || "为什么这样判断",
      body: trimText(explanation.diagnosis || decision.reason, 500)
    },
    // Display-ready signals; still excludes raw amounts via safePublicFacts.
    signals: safePublicFacts(record).map((signal) => ({
      label: PUBLIC_FACT_LABELS[signal.field] || signal.field,
      value: signal.range ? `${signal.value}（${signal.range}）` : String(signal.value),
      status: signal.status
    })),
    plans: plans.map((plan, index) => ({
      role: index === 0 ? "主方案" : "备选方案",
      bottleneck: trimText(plan.bottleneck, 80),
      // Older Agent candidates did not always carry a display title. Never
      // publish a blank card title: fall back to its concrete action.
      title: trimText(plan.title || plan.action || plan.bottleneck || "待验证方案", 120),
      action: trimText(plan.action, 300),
      budgetCap: Number(plan.budget_cap ?? plan.budgetCap) || 0,
      durationDays: Number(plan.duration_days ?? plan.durationDays) || 0,
      metric: trimText(plan.metric, 80),
      successLine: trimText(plan.success_line || plan.successLine, 120),
      stopLine: trimText(plan.stop_line || plan.stopLine, 120)
    })),
    rejectedReasons: rejectedRaw.slice(0, 6).map((reason) => trimText(String(reason), 160)),
    evidenceScore: Number(decision.metrics?.completeness) || dataScore
  };
  const row = { id, sourceCaseId: record.id, manageTokenHash: await tokenHash(manageToken), snapshot, dataScore, outcomeScore: 0, rankScore: Math.round(dataScore * .7), isActive: true, createdAt: timestamp, updatedAt: timestamp };
  publicCaseStore.set(id, row);
  if (env.DB) await env.DB.prepare(`INSERT INTO public_cases (id, source_case_id, manage_token_hash, snapshot_json, data_score, outcome_score, rank_score, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)`)
    .bind(row.id, row.sourceCaseId, row.manageTokenHash, JSON.stringify(row.snapshot), row.dataScore, row.outcomeScore, row.rankScore, timestamp, timestamp).run();
  return apiJson(request, env, { publicId: id, manageToken, rankScore: row.rankScore, snapshot }, 201);
}

async function updatePublicOutcome(request, env, publicId) {
  const body = await readJson(request);
  const manageToken = request.headers.get("X-Public-Manage-Token") || body.manageToken;
  const metric = cleanText(body.metric, 30);
  const before = Number(body.before);
  const after = Number(body.after);
  if (!PUBLIC_METRICS.has(metric) || !Number.isFinite(before) || !Number.isFinite(after) || before < 0 || after < 0) return apiJson(request, env, { code: "OUTCOME_INVALID", message: "请提交有效的结构化前后数据" }, 422);
  let row = publicCaseStore.get(publicId);
  if (env.DB) {
    const stored = await env.DB.prepare("SELECT * FROM public_cases WHERE id = ? AND is_active = 1").bind(publicId).first();
    if (!stored) return apiJson(request, env, { code: "PUBLIC_CASE_NOT_FOUND", message: "公开案例不存在" }, 404);
    row = { id: stored.id, manageTokenHash: stored.manage_token_hash, snapshot: JSON.parse(stored.snapshot_json), dataScore: Number(stored.data_score), outcomeScore: Number(stored.outcome_score), rankScore: Number(stored.rank_score), isActive: Boolean(stored.is_active) };
  }
  if (!row || !manageToken || await tokenHash(manageToken) !== row.manageTokenHash) return apiJson(request, env, { code: "PUBLIC_CASE_FORBIDDEN", message: "案例管理凭证无效" }, 403);
  const delta = LOWER_IS_BETTER.has(metric) ? (before - after) / Math.max(before, 1) : (after - before) / Math.max(before, 1);
  const outcomeScore = Math.round(Math.max(0, Math.min(100, delta * 100)));
  const rankScore = Math.round(row.dataScore * .7 + outcomeScore * .3);
  const timestamp = nowIso();
  if (env.DB) {
    await env.DB.prepare("INSERT INTO public_case_outcomes (id, public_case_id, metric, before_value, after_value, created_at) VALUES (?, ?, ?, ?, ?, ?)").bind(secureId("outcome"), publicId, metric, before, after, timestamp).run();
    await env.DB.prepare("UPDATE public_cases SET outcome_score = ?, rank_score = ?, updated_at = ? WHERE id = ?").bind(outcomeScore, rankScore, timestamp, publicId).run();
  }
  publicCaseStore.set(publicId, { ...row, outcomeScore, rankScore, updatedAt: timestamp });
  return apiJson(request, env, { outcomeScore, rankScore });
}

async function unpublishCase(request, env, publicId) {
  const body = request.method === "DELETE" ? {} : await readJson(request);
  const manageToken = request.headers.get("X-Public-Manage-Token") || body.manageToken;
  let row = publicCaseStore.get(publicId);
  if (env.DB) {
    const stored = await env.DB.prepare("SELECT manage_token_hash FROM public_cases WHERE id = ? AND is_active = 1").bind(publicId).first();
    row = stored ? { manageTokenHash: stored.manage_token_hash } : null;
  }
  if (!row || !manageToken || await tokenHash(manageToken) !== row.manageTokenHash) return apiJson(request, env, { code: "PUBLIC_CASE_FORBIDDEN", message: "案例管理凭证无效" }, 403);
  if (env.DB) await env.DB.prepare("UPDATE public_cases SET is_active = 0, updated_at = ? WHERE id = ?").bind(nowIso(), publicId).run();
  publicCaseStore.delete(publicId);
  return apiJson(request, env, { unpublished: true });
}

async function routeCases(request, env, ctx, url) {
  const parts = url.pathname.split("/").filter(Boolean);
  if (request.method === "POST" && parts.length === 2) return createCase(request, env);
  const caseId = cleanId(parts[2], 80);
  if (!caseId) return null;
  if (request.method === "POST" && parts[3] === "location") return saveLocation(request, env, caseId);
  if (request.method === "POST" && parts[3] === "turns") return interviewTurn(request, env, caseId);
  if (request.method === "POST" && parts[3] === "asr") return dashScopeAsrResponse(request, env, caseId);
  if (request.method === "GET" && parts[3] === "interview") {
    return apiJson(request, env, {
      code: "ASR_PROTOCOL_RETIRED",
      message: "StepFun WebSocket ASR已停用，请使用阿里云短音频识别"
    }, 410);
  }
  if (request.method === "POST" && parts[3] === "review") return reviewFacts(request, env, caseId);
  if (request.method === "POST" && parts[3] === "analyze") return startAnalysis(request, env, caseId, ctx);
  if (request.method === "POST" && parts[3] === "publish") return publishCase(request, env, caseId);
  if (request.method === "GET" && parts[3] === "runs" && parts[4]) {
    return getRun(request, env, caseId, cleanId(parts[4], 80), parts[5] === "events");
  }
  if (request.method === "POST" && parts[3] === "plans" && parts[4] && parts[5] === "start") {
    return startPlan(request, env, caseId, cleanId(parts[4], 80));
  }
  if (request.method === "DELETE" && parts.length === 3) {
    const record = await requireCase(request, env, caseId);
    for (const [runId, run] of runStore.entries()) {
      if (run?.caseId === caseId) runStore.delete(runId);
    }
    caseStore.delete(caseId);
    if (env.DB) {
      // `analysis_runs` is private case data too. Delete it before the case so
      // a 24-hour case cleanup cannot leave orphaned reports in D1.
      await env.DB.prepare("DELETE FROM analysis_runs WHERE case_id = ?").bind(caseId).run();
      await env.DB.prepare("DELETE FROM cases WHERE id = ?").bind(caseId).run();
    }
    return apiJson(request, env, { deleted: true });
  }
  return null;
}

async function markQueueRunFailed(message, env, error) {
  const runId = cleanId(message?.runId, 100);
  const run = await loadRun(env, runId);
  if (!run || run.status === "completed") return;
  run.status = "failed";
  run.warning = `队列连续失败，已停止继续调用模型：${error instanceof Error ? error.message : "unknown"}`;
  run.progress = {
    ...(run.progress || {}),
    phase: "failed",
    failedRound: Number(message?.round) || null
  };
  run.updatedAt = nowIso();
  await persistRun(env, run);
  if (env.DB) {
    await env.DB.prepare(`
      UPDATE analysis_runs
      SET claim_token = NULL, claimed_round = NULL, claim_expires_at = NULL
      WHERE id = ?
    `).bind(runId).run();
  } else {
    run.queueClaim = null;
  }
}

export async function handleQueueMessageFailure(message, env, error) {
  // Cloudflare's max_retries counts retries after the initial delivery.
  // With max_retries=3 the terminal delivery is attempts=4. Marking the run
  // failed earlier would make the next delivery look terminal and get acked,
  // silently preventing the poison message from reaching the configured DLQ.
  if (Number(message.attempts) >= QUEUE_MAX_RETRIES + 1) {
    await markQueueRunFailed(message.body, env, error);
  }
  // Never ack a failed delivery. On the exhausted delivery this retry request
  // is what causes Queues to move the message to its dead-letter queue.
  message.retry({ delaySeconds: 15 });
}

export default {
  async queue(batch, env) {
    for (const message of batch.messages) {
      try {
        const outcome = await processAnalysisQueueMessage(message.body, env);
        if (outcome?.skipped === "claimed") {
          // Another delivery owns the D1 lease. Acking here can strand a run
          // if that owner crashes, so retain this delivery until the lease
          // expires and let the next attempt repair the chain.
          message.retry({ delaySeconds: outcome.retryAfterSeconds || 30 });
          continue;
        }
        message.ack();
      } catch (error) {
        console.error("analysis queue round failed", {
          runId: cleanId(message.body?.runId, 100),
          round: Number(message.body?.round),
          error: error instanceof Error ? error.message : "unknown"
        });
        await handleQueueMessageFailure(message, env, error);
      }
    }
  },

  async scheduled(_event, env, _ctx) {
    if (!env.DB) return { deleted: 0 };
    const result = await env.DB.prepare(
      "DELETE FROM cases WHERE expires_at <= ?"
    ).bind(nowIso()).run();
    return { deleted: Number(result?.meta?.changes ?? 0) };
  },

  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    cleanupMemoryStores();
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(request, env) });
    }
    if (url.pathname.startsWith("/api/") && !originAllowed(request, env)) {
      return apiJson(request, env, { code: "ORIGIN_DENIED", message: "请求来源不允许" }, 403);
    }
    if (url.pathname.startsWith("/api/") && !rateAllowed(request)) {
      return apiJson(request, env, { code: "RATE_LIMITED", message: "请求过于频繁，请稍后再试" }, 429);
    }
    const action = rateAction(request, url);
    if (action && !await actionRateAllowed(request, env, action)) {
      return apiJson(request, env, {
        code: "ACTION_RATE_LIMITED",
        message: "这项操作过于频繁，请稍后再试"
      }, 429);
    }
    if (request.method === "GET" && url.pathname === "/api/map/context") {
      return getMapContext(url, env);
    }
    if (request.method === "GET" && url.pathname === "/api/map/address-context") {
      return getAddressContext(url, env);
    }
    if (request.method === "GET" && url.pathname === "/api/map/pick-context") {
      return getPickedMapContext(url, env);
    }
    if (request.method === "GET" && url.pathname === "/api/map/static") {
      return getStaticMap(url, env);
    }
    if (request.method === "GET" && url.pathname === "/api/map/ip-location") {
      return getApproximateLocation(request, env);
    }
    if (request.method === "POST" && url.pathname === "/api/tts") {
      try {
        return await ttsResponse(request, env);
      } catch (error) {
        return apiJson(request, env, {
          code: "TTS_ERROR",
          message: error instanceof Error ? error.message : "语音合成失败"
        }, Number(error?.status) || 502);
      }
    }
    if (request.method === "GET" && url.pathname === "/api/leaderboard") {
      return leaderboard(request, env);
    }
    const publicMatch = url.pathname.match(/^\/api\/public-cases\/([^/]+)$/);
    if (publicMatch && request.method === "GET") return publicCaseDetail(request, env, cleanId(publicMatch[1], 100));
    if (publicMatch && request.method === "POST") return updatePublicOutcome(request, env, cleanId(publicMatch[1], 100));
    if (publicMatch && request.method === "DELETE") return unpublishCase(request, env, cleanId(publicMatch[1], 100));
    if (url.pathname.startsWith("/api/cases")) {
      try {
        const response = await routeCases(request, env, ctx, url);
        if (response) return response;
      } catch (error) {
        return apiJson(request, env, {
          code: "CASE_API_ERROR",
          message: error instanceof Error ? error.message : "案卷服务异常"
        }, Number(error?.status) || 500);
      }
    }
    if (url.pathname.startsWith("/api/")) {
      return apiJson(request, env, { code: "NOT_FOUND", message: "接口不存在" }, 404);
    }
    if (!env.ASSETS) return new Response("Not found", { status: 404 });
    return env.ASSETS.fetch(request);
  }
};
