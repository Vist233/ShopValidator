import assert from "node:assert/strict";
import { StepFunClient } from "../src/stepfun-client.js";

function mockClient(responses, calls = []) {
  return new StepFunClient(
    { STEPFUN_API_KEY: "test-key" },
    {
      fetch: async (_input, init) => {
        calls.push(JSON.parse(init.body));
        const response = responses.shift();
        if (response instanceof Response) return response;
        return Response.json(response);
      }
    }
  );
}

{
  const calls = [];
  const client = mockClient([
    {
      choices: [{
        finish_reason: "stop",
        message: {
          content: "",
          reasoning_content: '{"ok":false,"source":"reasoning"}'
        }
      }]
    },
    {
      choices: [{
        finish_reason: "stop",
        message: { content: '{"ok":true,"source":"content"}' }
      }]
    }
  ], calls);
  const result = await client.chatJson([{ role: "user", content: "返回 JSON" }]);
  assert.deepEqual(result, { ok: true, source: "content" });
  assert.equal(calls.length, 2);
  assert.match(calls[1].messages.at(-1).content, /只输出一个完整的 JSON 对象/);
}

{
  const calls = [];
  const client = mockClient([
    {
      choices: [{
        finish_reason: "length",
        message: { content: '{"ok":true}' }
      }]
    },
    {
      choices: [{
        finish_reason: "stop",
        message: { content: '{"ok":true,"retried":true}' }
      }]
    }
  ], calls);
  const result = await client.chatJson([{ role: "user", content: "返回 JSON" }], {
    maxTokens: 200
  });
  assert.equal(result.retried, true);
  assert.equal(calls[0].max_tokens, 200);
  assert.equal(calls[1].max_tokens, 600);
}

{
  const calls = [];
  const client = mockClient([
    {
      choices: [{
        finish_reason: "stop",
        message: { content: "```json\n{\"ok\": true" }
      }]
    },
    {
      choices: [{
        finish_reason: "stop",
        message: { content: '{"ok":true,"next_question":"一天多少单？"}' }
      }]
    }
  ], calls);
  const result = await client.chatJson([{ role: "user", content: "返回 JSON" }], {
    requiredKeys: ["ok", "next_question"]
  });
  assert.equal(result.next_question, "一天多少单？");
  assert.equal(calls.length, 2);
}

{
  let validations = 0;
  const client = mockClient([
    {
      choices: [{
        finish_reason: "stop",
        message: { content: '{"ok":true}' }
      }]
    },
    {
      choices: [{
        finish_reason: "stop",
        message: { content: '{"ok":true,"next_question":"客单价多少？"}' }
      }]
    }
  ]);
  const result = await client.chatJson([{ role: "user", content: "返回 JSON" }], {
    validateJson(value) {
      validations += 1;
      return typeof value.next_question === "string";
    }
  });
  assert.equal(result.next_question, "客单价多少？");
  assert.equal(validations, 2);
}

{
  const calls = [];
  const client = mockClient([
    {
      choices: [{
        finish_reason: "stop",
        message: { content: "" }
      }]
    },
    {
      choices: [{
        finish_reason: "stop",
        message: { content: "仍然不是 JSON" }
      }]
    }
  ], calls);
  await assert.rejects(
    client.chatJson([{ role: "user", content: "返回 JSON" }]),
    /连续两次未返回可用 JSON/
  );
  assert.equal(calls.length, 2);
}

console.log("StepFun client: 5 retry and validation scenarios passed");

{
  let requestUrl = "";
  let requestInit;
  const client = new StepFunClient(
    {
      AI_PROVIDER: "openrouter",
      OPENROUTER_API_KEY: "router-test-key",
      OPENROUTER_BASE_URL: "https://openrouter.ai/api/v1",
      OPENROUTER_MODEL: "openrouter/free",
      OPENROUTER_SITE_URL: "https://shopvalidator.zhangyvjing.com",
      OPENROUTER_APP_NAME: "ShopValidator"
    },
    {
      fetch: async (input, init) => {
        requestUrl = String(input);
        requestInit = init;
        return Response.json({ choices: [{ finish_reason: "stop", message: { content: '{"ok":true}' } }] });
      }
    }
  );
  assert.equal(client.provider, "openrouter");
  assert.equal(client.textModel, "openrouter/free");
  assert.deepEqual(await client.chatJson([{ role: "user", content: "返回 JSON" }]), { ok: true });
  assert.equal(requestUrl, "https://openrouter.ai/api/v1/chat/completions");
  assert.equal(requestInit.headers["HTTP-Referer"], "https://shopvalidator.zhangyvjing.com");
  assert.equal(requestInit.headers["X-Title"], "ShopValidator");
  assert.equal(JSON.parse(requestInit.body).model, "openrouter/free");
}

console.log("OpenRouter free client: endpoint, headers, and JSON mode passed");
