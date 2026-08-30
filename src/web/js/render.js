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

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, AMP + "amp;")
    .replace(/</g, AMP + "lt;")
    .replace(/>/g, AMP + "gt;")
    .replace(/"/g, QUOT + "quot;")
    .replace(/'/g, APOS + "#39;");
}

function initials(name) {
  const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[1][0]).toUpperCase();
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
      mark = "✓";
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
    const rel = `${Math.round((item.final_score || 0) * 100)}%`;
    const er = item.static_avg_er != null ? `${Number(item.static_avg_er).toFixed(1)}%` : "—";
    const subs = formatCount(item.subscribers_count);
    const category = item.category_path || "";
    const inShort = store.isInShortlist(id);
    const shortCls = inShort ? "btn-short in" : "btn-short";
    const shortLabel = inShort ? "В шортлисте" : "В шортлист";
    const color = colorFor(name);
    const explanation = item.explanation || "";
    const relPct = Math.min(100, Math.max(0, (item.final_score || 0) * 100));
    let thirdStat = "";
    if (item.location) {
      thirdStat = item.location.length > 28 ? item.location.slice(0, 26) + "…" : item.location;
    } else if (item.primary_language) {
      thirdStat = `Язык: ${item.primary_language.toUpperCase()}`;
    } else if (category) {
      const lastLevel = category.split(">").pop().trim();
      thirdStat = lastLevel.length > 25 ? lastLevel.slice(0, 25) : lastLevel;
    }

    return `<div class="author-card" data-author-id="${id}">
      <div class="author-top">
        <span class="avatar" style="width:44px;height:44px;background:${color};color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px;">${escapeHtml(initials(name))}</span>
        <div class="author-info">
          <div class="author-name-row">
            <span class="author-name">${escapeHtml(name)}</span>
            <svg class="verified-badge" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"></path></svg>
          </div>
          <div class="author-handle">
            ${handle ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(handle)}</a>` : ""}
            ${platform ? ` · ${escapeHtml(platform)}` : ""}
          </div>
        </div>
        <div class="rel-block">
          <div class="rel-value">${rel}</div>
          <div class="rel-track"><div class="rel-bar" style="width:${relPct}%"></div></div>
          <div class="rel-caption">релевантность</div>
        </div>
      </div>

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
  const candidates = metadata.total_candidates_count != null ? metadata.total_candidates_count : "—";
  const timings = metadata.timings || {};
  const timingParts = Object.entries(timings)
    .map(([key, val]) => `<span class="meta-chip">${escapeHtml(key)}: ${Number(val).toFixed(0)} мс</span>`)
    .join("");

  container.innerHTML = `<div class="metadata-bar">
    <span class="meta-chip">Время: ${time}</span>
    <span class="meta-chip">Кандидатов: ${candidates}</span>
    ${timingParts}
  </div>`;
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
