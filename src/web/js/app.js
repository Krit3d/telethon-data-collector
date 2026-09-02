import { SearchApiClient } from "./api.js";
import {
  renderAuthorCards,
  renderCRM,
  renderMetadataBar,
  renderSearchProgress,
  renderShortlist,
  showToast,
} from "./render.js";
import { AppStore } from "./store.js";

const store = new AppStore();
const api = new SearchApiClient();

const app = document.querySelector(".app");
if (!app) {
  throw new Error("App root not found");
}

let progressTimer = null;
let progressStep = 0;

const HORMONE_OPTIONS = [
  { value: "dopamine", label: "Дофамин" },
  { value: "serotonin", label: "Серотонин" },
  { value: "oxytocin", label: "Окситоцин" },
  { value: "adrenaline", label: "Адреналин" },
  { value: "cortisol", label: "Кортизол" },
  { value: "endorphin", label: "Эндорфин" },
];

function buildHeader() {
  const header = document.createElement("header");
  header.className = "header";
  header.innerHTML = `
    <div class="header-inner">
      <button class="brand-btn" data-action="go-search">
        <span class="brand-name">Collabrama</span>
      </button>
      <div class="header-actions">
        <button class="tab-btn" data-action="tab" data-tab="search" title="AI-Поиск">
          <svg viewBox="0 0 24 24" width="21" height="21" fill="none" stroke="#6366f1" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 13.8 9 20 10.8 13.8 12.6 12 19 10.2 12.6 4 10.8 10.2 9z"></path></svg>
        </button>
        <button class="tab-btn" data-action="tab" data-tab="shortlist" title="Шортлист">
          <svg viewBox="0 0 24 24" width="21" height="21" fill="none" stroke="#8b5cf6" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8.5" r="3.7" fill="rgba(139,92,246,.16)"></circle><path d="M4.8 20c.6-3.6 3.6-5.6 7.2-5.6s6.6 2 7.2 5.6" fill="rgba(139,92,246,.16)"></path></svg>
          <span class="short-badge" data-short-badge style="display:none">0</span>
        </button>
        <button class="tab-btn" data-action="tab" data-tab="crm" title="CRM">
          <svg viewBox="0 0 24 24" width="21" height="21" fill="none" stroke="#6366f1" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.5C4 5.7 4.7 5 5.5 5h13c.8 0 1.5.7 1.5 1.5v8c0 .8-.7 1.5-1.5 1.5H9l-4 3v-3H5.5C4.7 16 4 15.3 4 14.5z" fill="rgba(99,102,241,.16)"></path></svg>
        </button>
      </div>
    </div>`;
  return header;
}

function buildFilterBar() {
  const hormonePills = HORMONE_OPTIONS.map((h) => {
    const on = store.selectedHormones.includes(h.value) ? " on" : "";
    return `<button class="pill hormone-pill${on}" data-action="toggle-hormone" data-hormone="${h.value}">${h.label}</button>`;
  }).join("");
  return `
    <div class="search-filters">
      <select class="select-pill" data-country-select>
        <option value="all" selected>Все страны</option>
        <option value="kz">Казахстан</option>
        <option value="ru">Россия</option>
        <option value="by">Беларусь</option>
        <option value="uz">Узбекистан</option>
        <option value="ae">ОАЭ</option>
        <option value="us">США / Global</option>
      </select>
      <select class="select-pill" data-language-select>
        <option value="all" selected>Все языки</option>
        <option value="ru">Русский</option>
        <option value="kk">Казахский</option>
        <option value="en">Английский</option>
        <option value="uk">Украинский</option>
      </select>
      <div class="subscribers-group">
        <span class="subscribers-label">Подписчики:</span>
        <input type="number" class="range-input" data-filter="min-followers" placeholder="от">
        <span class="range-divider">—</span>
        <input type="number" class="range-input" data-filter="max-followers" placeholder="до">
      </div>
      <select class="select-pill" data-tone-select>
        <option value="all" selected>Все стили</option>
        <option value="expert">Экспертный</option>
        <option value="educational">Обучающий</option>
        <option value="entertainment">Развлекательный</option>
        <option value="provocative">Провокационный</option>
        <option value="casual">Повседневный</option>
        <option value="analytical">Аналитический</option>
      </select>
      <div class="hormone-pills">${hormonePills}</div>
      <input type="text" class="stop-topics-input" data-filter="stop-topics" placeholder="Исключить темы...">
    </div>`;
}

function buildSearchTab() {
  const container = document.createElement("div");
  const isStepOne = store.currentStep === 1;
  const analyzeLabel = store.isAnalyzingBrand ? "Анализ ЦА..." : "Определить ЦА ✨";
  const brandValue = store.brandDescription || store.searchQuery;
  const audienceValue = store.targetAudienceDescription || store.searchQuery;
  const brandRows = Math.max(2, Math.ceil((brandValue.length || 1) / 60));
  const audienceRows = Math.max(4, Math.ceil((audienceValue.length || 1) / 60));

  const cardInner = isStepOne
    ? `
      <div class="search-top">
        <div class="search-label">
          <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="#6366f1" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 13.8 9 20 10.8 13.8 12.6 12 19 10.2 12.6 4 10.8 10.2 9z"></path></svg>
          <span class="search-label-text">AI-подбор авторов</span>
        </div>
      </div>
      <textarea class="search-textarea" data-search-input rows="${brandRows}" placeholder="Например: натуральная косметика для чувствительной кожи, продаёмся на Wildberries. Нужны авторы, которые реально говорят про уход и составы.">${escapeText(brandValue)}</textarea>
      <div class="search-bottom">
        ${buildFilterBar()}
        <div class="search-controls">
          <button class="btn-secondary-ai" data-action="analyze-brand"${store.isAnalyzingBrand ? " disabled" : ""}>
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 13.8 9 20 10.8 13.8 12.6 12 19 10.2 12.6 4 10.8 10.2 9z"></path></svg>
            <span>${analyzeLabel}</span>
          </button>
        </div>
      </div>`
    : `
      <div class="search-top">
        <div class="search-label">
          <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="#6366f1" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 13.8 9 20 10.8 13.8 12.6 12 19 10.2 12.6 4 10.8 10.2 9z"></path></svg>
          <span class="search-label-text">AI-подбор авторов</span>
        </div>
      </div>
      <div class="audience-confirm-card">
        <div class="audience-confirm-head">
          <span class="audience-confirm-badge">🎯</span>
          <span class="audience-confirm-title">Корректно ли определена ваша целевая аудитория?</span>
        </div>
        <textarea class="search-textarea audience-textarea" data-audience-input rows="${audienceRows}">${escapeText(audienceValue)}</textarea>
        <div class="brand-summary-badge">
          <span class="brand-summary-label">Исходный бренд:</span>
          <span class="brand-summary-text">${escapeText(store.brandDescription)}</span>
        </div>
      </div>
      <div class="search-bottom">
        ${buildFilterBar()}
        <div class="search-controls">
          <button class="btn-secondary-ai" data-action="reset-audience">↺ Сбросить к бренду</button>
          <button class="btn-primary" data-action="run-search">🔍 Найти авторов</button>
        </div>
      </div>`;

  container.innerHTML = `
    <div class="search-card">
      ${cardInner}
    </div>
    <div data-progress></div>
    <div data-metadata></div>
    <div class="results" data-results></div>`;
  return container;
}

function buildShortlistTab() {
  const container = document.createElement("div");
  container.innerHTML = `
    <div class="shortlist-card">
      <div class="shortlist-head">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#6366f1" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="5" height="16" rx="1.5"></rect><rect x="10" y="4" width="5" height="11" rx="1.5"></rect><rect x="17" y="4" width="4" height="7" rx="1.5"></rect></svg>
        <h2 class="shortlist-title">Мой шортлист</h2>
        <span class="badge-soft" data-short-count>0 авторов</span>
      </div>
      <div data-shortlist></div>
    </div>`;
  return container;
}

function buildCrmTab() {
  const container = document.createElement("div");
  container.className = "crm-layout";
  container.innerHTML = `
    <div class="crm-sidebar">
      <div class="crm-sidebar-head">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="#6366f1" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.5C4 5.7 4.7 5 5.5 5h13c.8 0 1.5.7 1.5 1.5v8c0 .8-.7 1.5-1.5 1.5H9l-4 3v-3H5.5C4.7 16 4 15.3 4 14.5z"></path></svg>
        <div class="crm-sidebar-title">Переговоры</div>
        <span class="badge ml-auto" data-thread-count>0</span>
      </div>
      <div class="thread-list" data-thread-list></div>
    </div>
    <div class="crm-chat" data-chat></div>`;
  return container;
}

function bindFilterBar(tab) {
  const countrySelect = tab.querySelector("[data-country-select]");
  if (countrySelect) {
    countrySelect.value = store.selectedCountry || "all";
    countrySelect.addEventListener("change", () => {
      store.selectedCountry = countrySelect.value;
    });
  }
  const languageSelect = tab.querySelector("[data-language-select]");
  if (languageSelect) {
    languageSelect.value = store.selectedLanguage || "all";
    languageSelect.addEventListener("change", () => {
      store.selectedLanguage = languageSelect.value;
    });
  }
  const minFollowers = tab.querySelector("[data-filter='min-followers']");
  if (minFollowers) {
    minFollowers.value = store.minFollowers ?? "";
    minFollowers.addEventListener("input", () => {
      store.minFollowers = minFollowers.value === "" ? null : Number(minFollowers.value);
    });
  }
  const maxFollowers = tab.querySelector("[data-filter='max-followers']");
  if (maxFollowers) {
    maxFollowers.value = store.maxFollowers ?? "";
    maxFollowers.addEventListener("input", () => {
      store.maxFollowers = maxFollowers.value === "" ? null : Number(maxFollowers.value);
    });
  }
  const toneSelect = tab.querySelector("[data-tone-select]");
  if (toneSelect) {
    toneSelect.value = store.selectedTone || "all";
    toneSelect.addEventListener("change", () => {
      store.selectedTone = toneSelect.value;
    });
  }
  const stopTopicsInput = tab.querySelector("[data-filter='stop-topics']");
  if (stopTopicsInput) {
    stopTopicsInput.value = store.stopTopicsInput || "";
    stopTopicsInput.addEventListener("input", () => {
      store.stopTopicsInput = stopTopicsInput.value;
    });
  }
  const hormonePills = tab.querySelectorAll("[data-action='toggle-hormone']");
  hormonePills.forEach((pill) => {
    pill.classList.toggle("on", store.selectedHormones.includes(pill.dataset.hormone));
  });
}

function render() {
  const main = app.querySelector(".main");
  if (!main) return;

  const header = app.querySelector(".header");
  if (header) header.remove();
  app.prepend(buildHeader());

  main.innerHTML = "";
  const tabButtons = app.querySelectorAll(".tab-btn");
  tabButtons.forEach((btn) => {
    btn.classList.toggle("on", btn.dataset.tab === store.activeTab);
  });

  const shortBadge = app.querySelector("[data-short-badge]");
  if (shortBadge) {
    shortBadge.textContent = String(store.shortlist.length);
    shortBadge.style.display = store.shortlist.length > 0 ? "flex" : "none";
  }

  if (store.activeTab === "search") {
    const tab = buildSearchTab();
    main.appendChild(tab);
    const brandInput = tab.querySelector("[data-search-input]");
    if (brandInput) {
      brandInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          analyzeBrandAction();
        }
      });
      brandInput.addEventListener("input", () => {
        store.brandDescription = brandInput.value;
        if (store.precomputedPlan && brandInput.value.trim() !== store.searchQuery.trim()) {
          store.precomputedPlan = null;
        }
      });
    }
    const audienceInput = tab.querySelector("[data-audience-input]");
    if (audienceInput) {
      audienceInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          runSearch();
        }
      });
      audienceInput.addEventListener("input", () => {
        store.targetAudienceDescription = audienceInput.value;
      });
    }
    bindFilterBar(tab);
    renderResults();
  } else if (store.activeTab === "shortlist") {
    const tab = buildShortlistTab();
    main.appendChild(tab);
    const countEl = tab.querySelector("[data-short-count]");
    if (countEl) {
      const n = store.shortlist.length;
      countEl.textContent = n === 1 ? "1 автор" : `${n} авторов`;
    }
    renderShortlist(store.shortlist, tab.querySelector("[data-shortlist]"), store);
  } else if (store.activeTab === "crm") {
    const tab = buildCrmTab();
    main.appendChild(tab);
    const countEl = tab.querySelector("[data-thread-count]");
    if (countEl) countEl.textContent = String(store.threads.length);
    renderCRM(
      store.threads,
      store.activeThreadId,
      tab.querySelector("[data-thread-list]"),
      tab.querySelector("[data-chat]"),
      store
    );
    const composer = tab.querySelector("[data-composer]");
    if (composer) {
      composer.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });
    }
  }
}

function renderResults() {
  const main = app.querySelector(".main");
  if (!main) return;
  const resultsEl = main.querySelector("[data-results]");
  const metadataEl = main.querySelector("[data-metadata]");
  if (!resultsEl) return;

  resultsEl.innerHTML = "";

  renderMetadataBar(store.queryMetadata, metadataEl);

  let filtered = store.searchResults.slice();
  if (store.platformFilter !== "all") {
    filtered = filtered.filter((a) => (a.platform || "").toLowerCase() === store.platformFilter);
  }
  if (store.matchTypeFilter !== "all") {
    filtered = filtered.filter((a) => a.match_type === store.matchTypeFilter);
  }
  if (store.reachFilter !== "all") {
    filtered = filtered.filter((a) => {
      const subs = a.subscribers_count || 0;
      switch (store.reachFilter) {
        case "1k-10k":
          return subs >= 1000 && subs < 10000;
        case "10k-50k":
          return subs >= 10000 && subs < 50000;
        case "50k-100k":
          return subs >= 50000 && subs < 100000;
        case "100k+":
          return subs >= 100000;
        default:
          return true;
      }
    });
  }
  filtered.sort((a, b) => (b.final_score || 0) - (a.final_score || 0));

  if (store.searchResults.length === 0) {
    resultsEl.innerHTML = `<div class="empty-state">
      <div class="empty-title">Запустите поиск</div>
      <div class="empty-text">Опишите задачу и нажмите «Найти авторов»</div>
    </div>`;
    return;
  }

  const filters = document.createElement("div");
  filters.className = "filters";
  filters.innerHTML = `
    <svg class="filter-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="#b3b8c8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 5h18"></path><path d="M6 12h12"></path><path d="M10 19h4"></path></svg>
    <select class="filter-select" data-filter="platform">
      <option value="all">Все платформы</option>
      <option value="instagram">Instagram</option>
      <option value="telegram">Telegram</option>
    </select>
    <select class="filter-select" data-filter="reach">
      <option value="all">Любой охват</option>
      <option value="1k-10k">1K – 10K</option>
      <option value="10k-50k">10K – 50K</option>
      <option value="50k-100k">50K – 100K</option>
      <option value="100k+">100K+</option>
    </select>`;
  resultsEl.appendChild(filters);

  const platformSel = filters.querySelector("[data-filter='platform']");
  platformSel.value = store.platformFilter;
  platformSel.addEventListener("change", () => {
    store.platformFilter = platformSel.value;
    renderResults();
  });
  const reachSel = filters.querySelector("[data-filter='reach']");
  reachSel.value = store.reachFilter;
  reachSel.addEventListener("change", () => {
    store.reachFilter = reachSel.value;
    renderResults();
  });

  const pills = document.createElement("div");
  pills.className = "match-type-pills";
  const totalCount = store.searchResults.length;
  const directCount = store.searchResults.filter((a) => a.match_type === "direct").length;
  const affinityCount = store.searchResults.filter((a) => a.match_type === "affinity").length;
  const pillOptions = [
    { value: "all", label: `🔀 Все авторы (${totalCount})` },
    { value: "affinity", label: `💡 Целевая аудитория (${affinityCount})` },
    { value: "direct", label: `🎯 Прямой поиск (${directCount})` },
  ];
  pills.innerHTML = pillOptions.map((opt) => `
    <button class="match-type-pill${store.matchTypeFilter === opt.value ? " on" : ""}" data-action="match-type" data-match-type="${opt.value}">
      <span>${opt.label}</span>
    </button>`).join("");
  resultsEl.appendChild(pills);

  const grid = document.createElement("div");
  grid.dataset.grid = "";
  resultsEl.appendChild(grid);
  renderAuthorCards(filtered, grid, store);
}

const ESC_AMP = String.fromCharCode(38);
const ESC_LT = String.fromCharCode(60);
const ESC_GT = String.fromCharCode(62);
const ESC_QUOT = String.fromCharCode(34);
const ESC_APOS = String.fromCharCode(39);

function escapeText(value) {
  return String(value ?? "")
    .replace(/&/g, ESC_AMP + "amp;")
    .replace(/</g, ESC_LT + "lt;")
    .replace(/>/g, ESC_GT + "gt;")
    .replace(/"/g, ESC_QUOT + "quot;")
    .replace(/'/g, ESC_APOS + "#39;");
}

function startProgress() {
  stopProgress();
  progressStep = 0;
  const main = app.querySelector(".main");
  const progressEl = main ? main.querySelector("[data-progress]") : null;
  if (progressEl) renderSearchProgress(0, progressEl);
  progressTimer = setInterval(() => {
    progressStep = (progressStep + 1) % 4;
    const el = app.querySelector("[data-progress]");
    if (el) renderSearchProgress(progressStep, el);
  }, 900);
}

function stopProgress() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

async function runSearch() {
  const main = app.querySelector(".main");
  const audienceInput = main ? main.querySelector("[data-audience-input]") : null;
  const brandInput = main ? main.querySelector("[data-search-input]") : null;
  const query = audienceInput
    ? audienceInput.value.trim()
    : (brandInput ? brandInput.value.trim() : store.searchQuery.trim());
  if (!query) {
    showToast("Введите описание задачи", "error");
    return;
  }
  store.targetAudienceDescription = query;
  store.searchQuery = query;
  startProgress();

  try {
    const data = await api.search(store.buildSearchRequest(query));
    store.searchResults = data.items || [];
    store.queryMetadata = data.query_metadata || null;
    if (data.inferred_filters) {
      store.applyInferredFilters(data.inferred_filters);
      syncInferredFilters();
    }
    stopProgress();
    const progressEl = app.querySelector("[data-progress]");
    if (progressEl) progressEl.innerHTML = "";
    store.currentStep = 3;
    renderResults();
    if (store.searchResults.length === 0) {
      showToast("Ничего не найдено, попробуйте изменить запрос", "error");
    } else {
      showToast(`Найдено авторов: ${store.searchResults.length}`);
    }
  } catch (err) {
    stopProgress();
    const progressEl = app.querySelector("[data-progress]");
    if (progressEl) progressEl.innerHTML = "";
    if (err.name === "AbortError") {
      return;
    }
    showToast(err.message || "Ошибка поиска", "error");
  }
}

function syncInferredFilters() {
  const main = app.querySelector(".main");
  if (!main) return;
  const countrySelect = main.querySelector("[data-country-select]");
  if (countrySelect) {
    countrySelect.value = store.selectedCountry || "all";
    flashInferred(countrySelect);
  }
  const languageSelect = main.querySelector("[data-language-select]");
  if (languageSelect) {
    languageSelect.value = store.selectedLanguage || "all";
    flashInferred(languageSelect);
  }
  const minFollowers = main.querySelector("[data-filter='min-followers']");
  if (minFollowers) {
    minFollowers.value = store.minFollowers ?? "";
    flashInferred(minFollowers);
  }
  const maxFollowers = main.querySelector("[data-filter='max-followers']");
  if (maxFollowers) {
    maxFollowers.value = store.maxFollowers ?? "";
    flashInferred(maxFollowers);
  }
  const toneSelect = main.querySelector("[data-tone-select]");
  if (toneSelect) {
    toneSelect.value = store.selectedTone || "all";
    flashInferred(toneSelect);
  }
  const stopTopicsInput = main.querySelector("[data-filter='stop-topics']");
  if (stopTopicsInput) {
    stopTopicsInput.value = store.stopTopicsInput || "";
    flashInferred(stopTopicsInput);
  }
  const hormonePills = main.querySelectorAll("[data-action='toggle-hormone']");
  hormonePills.forEach((pill) => {
    pill.classList.toggle("on", store.selectedHormones.includes(pill.dataset.hormone));
  });
}

function flashInferred(el) {
  el.classList.remove("ai-inferred");
  void el.offsetWidth;
  el.classList.add("ai-inferred");
  setTimeout(() => el.classList.remove("ai-inferred"), 2000);
}

function findAuthor(accountId) {
  return store.searchResults.find((a) => a.account_id === accountId) || null;
}

function sendMessage() {
  const main = app.querySelector(".main");
  const composer = main ? main.querySelector("[data-composer]") : null;
  const text = composer ? composer.value : "";
  if (!store.activeThreadId || !text.trim()) return;
  store.sendMessage(store.activeThreadId, text);
  if (composer) composer.value = "";
  render();
  const feed = app.querySelector("[data-chat-feed]");
  if (feed) feed.scrollTop = feed.scrollHeight;
}

async function analyzeBrandAction() {
  const main = app.querySelector(".main");
  const input = main ? main.querySelector("[data-search-input]") : null;
  const text = input ? input.value.trim() : "";
  if (!text) {
    showToast("Опишите бренд для анализа", "error");
    return;
  }
  store.brandDescription = text;
  store.searchResults = [];
  store.queryMetadata = null;
  store.isAnalyzingBrand = true;
  render();
  const analyzeBtn = app.querySelector("[data-action='analyze-brand']");
  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Анализ ЦА...";
  }
  try {
    const data = await api.analyzeBrand({
      brand_description: text,
      stop_topics: store.stopTopicsInput ? store.stopTopicsInput.split(",").map((s) => s.trim()).filter(Boolean) : [],
    });
    store.applyBrandAnalysis(data);
    store.isAnalyzingBrand = false;
    store.currentStep = 2;
    render();
    syncInferredFilters();
    showToast("Целевая аудитория определена");
  } catch (err) {
    store.isAnalyzingBrand = false;
    render();
    if (err.name === "AbortError") {
      return;
    }
    showToast(err.message || "Ошибка анализа ЦА", "error");
  }
}

function resetAudience() {
  store.resetAudienceState();
  store.matchTypeFilter = "all";
  render();
  syncInferredFilters();
}

app.addEventListener("click", (e) => {
  const target = e.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;

  if (action === "tab") {
    store.activeTab = target.dataset.tab;
    render();
  } else if (action === "go-search") {
    store.activeTab = "search";
    render();
  } else if (action === "platform") {
    store.platformFilter = target.dataset.platform;
    render();
  } else if (action === "run-search") {
    runSearch();
  } else if (action === "analyze-brand") {
    analyzeBrandAction();
  } else if (action === "reset-audience") {
    resetAudience();
  } else if (action === "toggle-hormone") {
    store.toggleHormone(target.dataset.hormone);
    const pills = app.querySelectorAll("[data-action='toggle-hormone']");
    pills.forEach((pill) => {
      pill.classList.toggle("on", store.selectedHormones.includes(pill.dataset.hormone));
    });
  } else if (action === "match-type") {
    store.matchTypeFilter = target.dataset.matchType || "all";
    renderResults();
  } else if (action === "toggle-shortlist") {
    const author = findAuthor(Number(target.dataset.authorId));
    if (author) {
      store.toggleShortlist(author);
      render();
      showToast(store.isInShortlist(author.account_id) ? "Добавлено в шортлист" : "Убрано из шортлиста");
    }
  } else if (action === "remove-shortlist") {
    const author = store.shortlist.find((a) => a.account_id === Number(target.dataset.authorId));
    if (author) {
      store.toggleShortlist(author);
      render();
      showToast("Убрано из шортлиста");
    }
  } else if (action === "open-chat") {
    const author = findAuthor(Number(target.dataset.authorId)) ||
      store.shortlist.find((a) => a.account_id === Number(target.dataset.authorId));
    if (author) {
      store.openChatWithAuthor(author);
      render();
    }
  } else if (action === "select-thread") {
    store.activeThreadId = target.dataset.threadId;
    render();
  } else if (action === "send-message") {
    sendMessage();
  } else if (action === "quick-reply") {
    const text = target.dataset.text || "";
    if (store.activeThreadId && text) {
      store.sendMessage(store.activeThreadId, text);
      render();
      const feed = app.querySelector("[data-chat-feed]");
      if (feed) feed.scrollTop = feed.scrollHeight;
    }
  } else if (action === "toggle-explanation") {
    const card = target.closest(".author-card");
    if (!card) return;
    const textEl = card.querySelector("[data-explanation]");
    const labelEl = card.querySelector("[data-explanation-label]");
    if (!textEl || !labelEl) return;
    const collapsed = textEl.classList.contains("explanation-clamped");
    if (collapsed) {
      textEl.classList.remove("explanation-clamped");
      labelEl.textContent = "свернуть";
    } else {
      textEl.classList.add("explanation-clamped");
      labelEl.textContent = "подробнее";
    }
  }
});

render();
