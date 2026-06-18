export function createWebSocketUrl(path: string, origin = window.location.origin): string {
  const url = new URL(path, origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

