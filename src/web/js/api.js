export class SearchApiClient {
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl;
    this.controller = null;
  }

  async search(payload, signal) {
    if (this.controller) {
      this.controller.abort();
    }
    this.controller = new AbortController();
    const effectiveSignal = signal || this.controller.signal;

    let response;
    try {
      response = await fetch(`${this.baseUrl}/api/v1/search/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: effectiveSignal,
      });
    } catch (err) {
      if (err.name === "AbortError") {
        throw err;
      }
      throw new Error(`Сетевая ошибка: ${err.message}`);
    }

    if (!response.ok) {
      let message = `Ошибка сервера (${response.status})`;
      try {
        const data = await response.json();
        if (data && data.detail) {
          message = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
        } else if (data && data.message) {
          message = data.message;
        }
      } catch {
        message = `Ошибка сервера (${response.status})`;
      }
      throw new Error(message);
    }

    return response.json();
  }
}
