import { describe, expect, it } from "vitest";

import { navigationItems, routes } from "./routes";

describe("application routes", () => {
  it("defines one route for each side navigation item", () => {
    expect(navigationItems.map((item) => item.path)).toEqual([
      "/",
      "/tasks/dispatch",
      "/tasks/history",
      "/logs",
    ]);
    expect(routes.map((route) => route.path)).toEqual(navigationItems.map((item) => item.path));
  });

  it("keeps navigation labels stable", () => {
    expect(navigationItems.map((item) => item.label)).toEqual([
      "仿真监控",
      "任务下发",
      "任务历史",
      "系统日志",
    ]);
  });
});
