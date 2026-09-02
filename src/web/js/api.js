export class SearchApiClient {
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl;
    this.controller = null;
  }

  async search(payload, signal) {
    return this._post("/api/v1/search/", payload, signal);
  }

  async analyzeBrand(payload, signal) {
    return this._post("/api/v1/search/analyze-brand", payload, signal);
  }

  async _post(path, payload, signal) {
    if (this.controller && !signal) {
      this.controller.abort();
    }
    const controller = signal ? null : new AbortController();
    if (controller) {
      this.controller = controller;
    }
    const effectiveSignal = signal || (controller ? controller.signal : undefined);

    let response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
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
