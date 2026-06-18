import { describe, expect, it } from "vitest";

import { createWebSocketUrl } from "./urls";

describe("createWebSocketUrl", () => {
  it("uses the current host and rewrites http to ws", () => {
    const result = createWebSocketUrl("/ws/simulation", "http://127.0.0.1:8020");

    expect(result).toBe("ws://127.0.0.1:8020/ws/simulation");
  });

  it("rewrites https to wss", () => {
    const result = createWebSocketUrl("/ws/simulation", "https://lab.example.local");

    expect(result).toBe("wss://lab.example.local/ws/simulation");
  });
});
