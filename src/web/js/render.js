const STAGES = [
  "Анализируем продукт",
  "Подбираем платформы",
  "Ищем релевантных авторов",
  "Собираем shortlist",
];

const AVATAR_COLORS = [
  "#6366f1",
  "#8b5cf6",
  "#ec4899",
  "#f59e0b",
  "#10b981",
  "#3b82f6",
  "#ef4444",
  "#14b8a6",
];

const AMP = String.fromCharCode(38);
const LT = String.fromCharCode(60);
const GT = String.fromCharCode(62);
const QUOT = String.fromCharCode(34);
const APOS = String.fromCharCode(39);
const CLAMP_CLASS = "explanation-clamped";

const TONE_LABELS = { expert: "Экспертный", educational: "Обучающий", entertainment: "Развлекательный", provocative: "Провокационный", casual: "Повседневный", analytical: "Аналитический" };
const HORMONE_LABELS = { dopamine: "Дофамин", serotonin: "Серотонин", oxytocin: "Окситоцин", adrenaline: "Адреналин", cortisol: "Кортизол", endorphin: "Эндорфин" };
const HORMONE_HINTS = { dopamine: "Драйв и тренды", serotonin: "Статус и порядок", oxytocin: "Семья и забота", adrenaline: "Экстрим и риск", cortisol: "Боли и проблемы", endorphin: "Юмор и позитив" };
const TONE_DESCRIPTIONS = {
  expert: "Экспертный: глубокий профессиональный анализ и авторитет",
  educational: "Обучающий: пошаговые инструкции, лайфхаки и советы",
  entertainment: "Развлекательный: шоу, эмоции и вирусный контент",
  provocative: "Провокационный: острые дискуссии, мнения и споры",
  casual: "Повседневный: лайфстайл, личные истории и искренность",
  analytical: "Аналитический: факты, цифры, сравнения и логика",
};
const HORMONE_DESCRIPTIONS = {
  dopamine: "Дофамин: новизна, вдохновение, тренды и вау-эффект",
  serotonin: "Серотонин: статус, уверенность, экспертность и контроль",
  oxytocin: "Окситоцин: забота, дети, семья, безопасность и доверие",
  adrenaline: "Адреналин: вызов, смелость, скорость и спорт",
  cortisol: "Кортизол: решение проблем, страхи, боли и защита",
  endorphin: "Эндорфин: радость, смех, легкость и хорошее настроение",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, AMP + "amp;")
    .replace(/</g, AMP + "lt;")
    .replace(/>/g, AMP + "gt;")
    .replace(/"/g, QUOT + "quot;")
    .replace(/'/g, APOS + "#39;");
}

function initials(name) {
  const cleaned = String(name || "")
    .replace(/[.,/()\[\]{}"'-]/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (cleaned.length === 0) return "?";
  if (cleaned.length === 1) return cleaned[0].slice(0, 2).toLocaleUpperCase("ru-RU");
  return (cleaned[0][0] + cleaned[1][0]).toLocaleUpperCase("ru-RU");
}

function colorFor(value) {
  let hash = 0;
  const str = String(value ?? "");
  for (let i = 0; i < str.length; i += 1) {
    hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
  }
  return AVATAR_COLORS[hash % AVATAR_COLORS.length];
}

function formatCount(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return "—";
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(1).replace(/\.0$/, "")}M`;
  }
  if (num >= 1000) {
    return `${(num / 1000).toFixed(1).replace(/\.0$/, "")}K`;
  }
  return String(num);
}

function formatTime(ts) {
  if (!ts) return "";
  const date = new Date(ts);
  return date.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
}

export function renderSearchProgress(stepIndex, container) {
  if (!container) return;
  const rows = STAGES.map((label, i) => {
    let cls = "stage-row";
    let dotCls = "stage-dot";
    let textCls = "stage-text";
    let mark = String(i + 1);
    if (i < stepIndex) {
      cls += " done";
      dotCls += " done";
      mark = `<svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"></path></svg>`;
    } else if (i === stepIndex) {
      cls += " active";
      dotCls += " active";
    } else {
      textCls += " pending";
    }
    return `<div class="${cls}">
      <span class="${dotCls}">${mark}</span>
      <span class="${textCls}">${escapeHtml(label)}</span>
    </div>`;
  }).join("");
  container.innerHTML = `<div class="stages">${rows}</div>`;
}

export function renderAudienceValidationCard(store) {
  if (!store || store.currentStep < 2 || !store.targetAudienceDescription) {
    return "";
  }
  return `<div class="audience-validation-card">
    <div class="audience-validation-head">
      <div class="audience-validation-title-group">
        <span class="audience-validation-title">Корректно ли определена ваша целевая аудитория?</span>
        <span class="audience-validation-badge">🎯 ЦА</span>
      </div>
      <button class="btn-banner-reset" data-action="reset-audience">↺ Сбросить к бренду</button>
    </div>
    <textarea class="audience-editable-textarea" data-audience-input rows="3">${escapeHtml(store.targetAudienceDescription)}</textarea>
  </div>`;
}

export function renderHormoneChips(selectedHormones = []) {
  const selected = Array.isArray(selectedHormones) ? selectedHormones : [];
  const chips = Object.entries(HORMONE_LABELS).map(([key, label]) => {
    const on = selected.includes(key) ? " on" : "";
    const desc = HORMONE_DESCRIPTIONS[key] || "";
    const hint = HORMONE_HINTS[key] || "";
    return `<button class="hormone-chip${on}" data-action="toggle-hormone" data-hormone="${key}" title="${escapeHtml(desc)}"><span class="hormone-label">${escapeHtml(label)}</span><span class="hormone-hint">· ${escapeHtml(hint)}</span></button>`;
  }).join("");
  return `<div class="hormone-chips">${chips}</div>`;
}

export function renderAuthorCards(authors, container, store) {
  if (!container) return;
  if (!authors || authors.length === 0) {
    container.innerHTML = `<div class="empty-state">
      <div class="empty-title">Ничего не найдено</div>
      <div class="empty-text">Попробуйте изменить запрос или фильтры</div>
    </div>`;
    return;
  }

  const cards = authors.map((item) => {
    const id = item.account_id;
    const name = item.title || item.username || "Автор";
    const handle = item.username ? `@${item.username}` : "";
    const url = item.url || "#";
    const platformRaw = item.platform || "";
    const platform = platformRaw ? platformRaw.charAt(0).toUpperCase() + platformRaw.slice(1).toLowerCase() : "";
    const relPct = Math.min(100, Math.max(0, (item.final_score || 0) * 100));
    const rel = `${Math.round(relPct)}%`;
    const er = item.static_avg_er != null ? `${Number(item.static_avg_er).toFixed(1)}%` : "—";
    const subs = formatCount(item.subscribers_count);
    const category = item.category_path || "";
    const inShort = store.isInShortlist(id);
    const shortCls = inShort ? "btn-short in" : "btn-short";
    const shortLabel = inShort ? "В шортлисте" : "В шортлист";
    const color = colorFor(name);
    const explanation = item.explanation || "";
    const matchBadge = item.match_type === "affinity"
      ? `<span class="badge-affinity"><svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="9" cy="12" r="5"></circle><circle cx="15" cy="12" r="5"></circle></svg><span>${item.affinity_reason ? `Смежная ЦА · ${escapeHtml(item.affinity_reason)}` : "Смежная ЦА"}</span></span>`
      : item.match_type === "direct"
        ? `<span class="badge-direct"><svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="7"></circle><circle cx="12" cy="12" r="2.5" fill="currentColor" stroke="none"></circle><path d="M12 2v3"></path><path d="M12 19v3"></path><path d="M2 12h3"></path><path d="M19 12h3"></path></svg><span>Прямой поиск</span></span>`
        : "";
    const psychoBadges = [];
    if (item.primary_tone) {
      const toneDesc = TONE_DESCRIPTIONS[item.primary_tone] || "";
      psychoBadges.push(`<span class="badge-psycho" title="${escapeHtml(toneDesc)}">${escapeHtml(TONE_LABELS[item.primary_tone] || item.primary_tone)}</span>`);
    }
    if (item.primary_hormone) {
      const hormoneDesc = HORMONE_DESCRIPTIONS[item.primary_hormone] || "";
      psychoBadges.push(`<span class="badge-psycho" title="${escapeHtml(hormoneDesc)}">${escapeHtml(HORMONE_LABELS[item.primary_hormone] || item.primary_hormone)}</span>`);
    }
    const langBadge = item.primary_language && String(item.primary_language).trim() !== ""
      ? `<span class="badge-lang" title="Язык контента: ${escapeHtml(String(item.primary_language).toUpperCase())}">${escapeHtml(String(item.primary_language).toUpperCase())}</span>`
      : "";
    const psychoRow = psychoBadges.length > 0 ? `<div class="badges-psycho">${psychoBadges.join("")}</div>` : "";
    const badgesRow = (matchBadge || langBadge || psychoRow) ? `<div class="author-badges">${matchBadge}${langBadge}${psychoRow}</div>` : "";
    let thirdStat = "";
    if (item.location) {
      thirdStat = item.location.length > 28 ? item.location.slice(0, 26) + "…" : item.location;
    } else if (item.primary_language) {
      thirdStat = `Язык: ${item.primary_language.toUpperCase()}`;
    } else if (category) {
      const lastLevel = category.split(">").pop().trim();
      thirdStat = lastLevel.length > 25 ? lastLevel.slice(0, 25) : lastLevel;
    }

    const handleHtml = handle
      ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(handle)}</a>`
      : "";
    const platformHtml = platform
      ? `${handle ? " · " : ""}${escapeHtml(platform)}`
      : "";

    return `<div class="author-card" data-author-id="${id}">
      <div class="author-top">
        <span class="avatar" style="width:44px;height:44px;background:${color};color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px;">${escapeHtml(initials(name))}</span>
        <div class="author-info">
          <div class="author-name-row">
            <span class="author-name">${escapeHtml(name)}</span>
            <svg class="verified-badge" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"></path></svg>
          </div>
          <div class="author-handle">
            ${handleHtml}
            ${platformHtml}
          </div>
        </div>
        <div class="rel-block">
          <div class="rel-value">${rel}</div>
          <div class="rel-track"><div class="rel-bar" style="width:${relPct}%"></div></div>
          <div class="rel-caption">релевантность</div>
        </div>
      </div>

      ${badgesRow}

      ${explanation ? `<div class="author-bio">
        <div class="explanation-text ${CLAMP_CLASS}" data-explanation>${escapeHtml(explanation)}</div>
        <button class="btn-link" data-action="toggle-explanation" data-author-id="${id}">
          <span data-explanation-label>подробнее</span>
          <svg class="chevron" viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"></path></svg>
        </button>
      </div>` : ""}

      <div class="author-stats">
        <div class="stat">
          <span class="stat-value">${subs}</span>
          <span class="stat-label">подписчиков</span>
        </div>
        <span class="stat-divider"></span>
        <div class="stat pad-left">
          <span class="stat-label">ER</span>
          <span class="stat-value">${er}</span>
        </div>
        ${thirdStat ? `<span class="stat-divider"></span><div class="stat-geo">${escapeHtml(thirdStat)}</div>` : ""}
      </div>

      <div class="author-actions">
        <button class="${shortCls}" data-action="toggle-shortlist" data-author-id="${id}">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M6 4h9l3 3v13l-6-3-6 3z" fill="currentColor" fill-opacity=".14"></path></svg>
          <span>${shortLabel}</span>
        </button>
        <button class="btn-chat" data-action="open-chat" data-author-id="${id}" title="Написать автору">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.5C4 5.7 4.7 5 5.5 5h13c.8 0 1.5.7 1.5 1.5v8c0 .8-.7 1.5-1.5 1.5H9l-4 3v-3H5.5C4.7 16 4 15.3 4 14.5z" fill="currentColor" fill-opacity=".14"></path></svg>
          <span>Диалог</span>
        </button>
      </div>
    </div>`;
  }).join("");

  container.innerHTML = `<div class="grid">${cards}</div>`;
}

export function renderMetadataBar(metadata, container) {
  if (!container) return;
  if (!metadata) {
    container.innerHTML = "";
    return;
  }
  const time = metadata.execution_time_ms != null ? `${Number(metadata.execution_time_ms).toFixed(0)} мс` : "—";
  const qdrant = metadata.qdrant_candidates_count != null ? metadata.qdrant_candidates_count : "—";
  const graph = metadata.graph_evidences_count != null ? metadata.graph_evidences_count : "—";
  const total = metadata.total_candidates_count != null ? metadata.total_candidates_count : "—";
  const timings = metadata.timings || {};
  const timingRows = Object.entries(timings)
    .map(([key, val]) => `<div class="analytics-row">
      <span class="analytics-key">${escapeHtml(key)}</span>
      <span class="analytics-value">${Number(val).toFixed(0)} мс</span>
    </div>`)
    .join("");

  container.innerHTML = `<details class="analytics-accordion">
    <summary class="analytics-summary">
      <span class="analytics-total">${time}</span>
      <span class="analytics-badge">Аналитика пайплайна</span>
      <svg class="analytics-chevron" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"></path></svg>
    </summary>
    <div class="analytics-body">
      <div class="analytics-grid">
        <div class="analytics-metric">
          <span class="analytics-metric-label">Кандидаты Qdrant</span>
          <span class="analytics-metric-value">${qdrant}</span>
        </div>
        <div class="analytics-metric">
          <span class="analytics-metric-label">Кандидаты Graph</span>
          <span class="analytics-metric-value">${graph}</span>
        </div>
        <div class="analytics-metric">
          <span class="analytics-metric-label">Всего кандидатов</span>
          <span class="analytics-metric-value">${total}</span>
        </div>
      </div>
      <div class="analytics-section">
        <div class="analytics-section-title">Тайминги</div>
        ${timingRows || `<div class="analytics-row"><span class="analytics-key">нет данных</span></div>`}
      </div>
    </div>
  </details>`;
}

export function renderShortlist(shortlist, container, store) {
  if (!container) return;
  if (!shortlist || shortlist.length === 0) {
    container.innerHTML = `<div class="empty-state">
      <div class="empty-title">Шортлист пуст</div>
      <div class="empty-text">Добавляйте авторов из результатов подбора — кнопкой «В шортлист»</div>
    </div>`;
    return;
  }

  const items = shortlist.map((item) => {
    const name = item.title || item.username || "Автор";
    const handle = item.username ? `@${item.username}` : "";
    const platform = item.platform || "";
    const subs = formatCount(item.subscribers_count);
    const er = item.static_avg_er != null ? `${Number(item.static_avg_er).toFixed(1)}%` : "—";
    const rel = `${Math.round((item.final_score || 0) * 100)}%`;
    const color = colorFor(name);

    return `<div class="shortlist-item" data-author-id="${item.account_id}">
      <span class="avatar" style="width:40px;height:40px;background:${color};color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;">${escapeHtml(initials(name))}</span>
      <div class="shortlist-item-info">
        <div class="shortlist-item-name">${escapeHtml(name)}</div>
        <div class="shortlist-item-sub">${handle ? escapeHtml(handle) : ""}${platform ? ` · ${escapeHtml(platform)}` : ""}${subs !== "—" ? ` · ${subs}` : ""}${er !== "—" ? ` · ER ${er}` : ""}</div>
      </div>
      <span class="badge-rel">${rel}</span>
      <button class="btn-solid" data-action="open-chat" data-author-id="${item.account_id}">Написать</button>
      <button class="btn-soft" data-action="remove-shortlist" data-author-id="${item.account_id}">Убрать</button>
    </div>`;
  }).join("");

  container.innerHTML = `<div class="shortlist-list">${items}</div>`;
}

export function renderCRM(threads, activeThreadId, sidebarContainer, chatContainer, store) {
  if (sidebarContainer) {
    if (!threads || threads.length === 0) {
      sidebarContainer.innerHTML = `<div class="empty-state">
        <div class="empty-title">Нет диалогов</div>
        <div class="empty-text">Откройте диалог с автором из результатов поиска</div>
      </div>`;
    } else {
      const items = threads.map((t) => {
        const last = t.messages && t.messages.length > 0 ? t.messages[t.messages.length - 1].text : "Нет сообщений";
        const time = t.messages && t.messages.length > 0 ? formatTime(t.messages[t.messages.length - 1].time) : "";
        const cls = t.id === activeThreadId ? "thread-item on" : "thread-item";
        const color = colorFor(t.title);
        return `<div class="${cls}" data-thread-id="${t.id}" data-action="select-thread">
          <span class="avatar" style="width:38px;height:38px;background:${color};color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;">${escapeHtml(initials(t.title))}</span>
          <div class="thread-info">
            <div class="thread-name-row">
              <div class="thread-name">${escapeHtml(t.title)}</div>
              <span class="thread-time">${time}</span>
            </div>
            <div class="thread-last">${escapeHtml(last)}</div>
          </div>
        </div>`;
      }).join("");
      sidebarContainer.innerHTML = `<div class="thread-list">${items}</div>`;
    }
  }

  if (chatContainer) {
    const active = threads.find((t) => t.id === activeThreadId);
    if (!active) {
      chatContainer.innerHTML = `<div class="empty-state">
        <div class="empty-title">Выберите диалог</div>
        <div class="empty-text">Или начните новый из карточки автора</div>
      </div>`;
      return;
    }

    const messages = (active.messages || []).map((m) => {
      const us = m.from === "us";
      const rowCls = us ? "msg-row us" : "msg-row";
      const bubbleCls = us ? "msg-bubble us" : "msg-bubble";
      const timeCls = us ? "msg-time us" : "msg-time";
      return `<div class="${rowCls}">
        <div class="${bubbleCls}">
          <div>${escapeHtml(m.text)}</div>
          <div class="${timeCls}">${formatTime(m.time)}</div>
        </div>
      </div>`;
    }).join("");

    const quickReplies = [
      "Здравствуйте! Мы ищем авторов для коллаборации",
      "Расскажите о вашей аудитории",
      "Какие форматы вам интересны?",
      "Обсудим условия сотрудничества?",
    ].map((q) => `<button class="quick-btn" data-action="quick-reply" data-text="${escapeHtml(q)}">
      <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="#6366f1" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 13.8 9 20 10.8 13.8 12.6 12 19 10.2 12.6 4 10.8 10.2 9z"></path></svg>
      <span>${escapeHtml(q)}</span>
    </button>`).join("");

    chatContainer.innerHTML = `
      <div class="crm-chat-head">
        <span class="avatar" style="width:40px;height:40px;background:${colorFor(active.title)};color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;">${escapeHtml(initials(active.title))}</span>
        <div>
          <div class="crm-chat-name">${escapeHtml(active.title)}</div>
          <div class="crm-chat-sub">${active.username ? escapeHtml(`@${active.username}`) : ""}${active.platform ? ` · ${escapeHtml(active.platform)}` : ""}</div>
        </div>
        <span class="badge-followup">авто follow-up</span>
      </div>
      <div class="chat-feed" data-chat-feed>${messages}</div>
      <div class="chat-composer">
        <div class="quick-replies">${quickReplies}</div>
        <div class="composer-row">
          <textarea class="composer-textarea" data-composer rows="2" placeholder="Напишите сообщение…"></textarea>
          <button class="btn-primary" data-action="send-message">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="m4 12 16-8-6 16-2-7z"></path></svg>
            <span>Отправить</span>
          </button>
        </div>
      </div>`;
  }
}

export function showToast(message, type = "success") {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();
  const toast = document.createElement("div");
  toast.className = "toast";
  const color = type === "error" ? "#ef4444" : "#10b981";
  toast.innerHTML = `<span class="toast-dot" style="background:${color}"></span><span class="toast-text">${escapeHtml(message)}</span>`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}
