const SHORTLIST_KEY = "collabrama_shortlist";
const THREADS_KEY = "collabrama_crm_threads";

function readStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    return;
  }
}

export class AppStore {
  constructor() {
    this.activeTab = "search";
    this.searchQuery = "";
    this.selectedCountry = "all";
    this.selectedLanguage = "all";
    this.minFollowers = null;
    this.maxFollowers = null;
    this.platformFilter = "all";
    this.sortFilter = "relevance";
    this.reachFilter = "all";
    this.authorType = "expert";
    this.searchResults = [];
    this.queryMetadata = null;
    this.shortlist = readStorage(SHORTLIST_KEY, []);
    this.threads = readStorage(THREADS_KEY, []);
    this.activeThreadId = null;
  }

  buildSearchRequest(query) {
    return {
      query,
      limit: 20,
      author_type: this.authorType || "expert",
      min_followers: (this.minFollowers && Number(this.minFollowers) > 0) ? Number(this.minFollowers) : null,
      max_followers: (this.maxFollowers && Number(this.maxFollowers) > 0) ? Number(this.maxFollowers) : null,
      location: this.selectedCountry !== "all" ? this.selectedCountry : null,
      languages: this.selectedLanguage !== "all" ? [this.selectedLanguage] : null,
      include_contacts: false,
      include_analytics: true,
    };
  }

  toggleShortlist(author) {
    const index = this.shortlist.findIndex((a) => a.account_id === author.account_id);
    if (index >= 0) {
      this.shortlist.splice(index, 1);
    } else {
      this.shortlist.push(author);
    }
    writeStorage(SHORTLIST_KEY, this.shortlist);
  }

  isInShortlist(accountId) {
    return this.shortlist.some((a) => a.account_id === accountId);
  }

  openChatWithAuthor(author) {
    let thread = this.threads.find((t) => t.authorId === author.account_id);
    if (!thread) {
      thread = {
        id: `thread_${author.account_id}_${Date.now()}`,
        authorId: author.account_id,
        title: author.title || author.username || "Автор",
        username: author.username || "",
        platform: author.platform || "",
        url: author.url || "",
        messages: [],
        createdAt: Date.now(),
      };
      this.threads.push(thread);
      writeStorage(THREADS_KEY, this.threads);
    }
    this.activeThreadId = thread.id;
    this.activeTab = "crm";
  }

  sendMessage(threadId, text) {
    const trimmed = (text || "").trim();
    if (!trimmed) return;
    const thread = this.threads.find((t) => t.id === threadId);
    if (!thread) return;
    thread.messages.push({
      id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      text: trimmed,
      from: "us",
      time: Date.now(),
    });
    writeStorage(THREADS_KEY, this.threads);
  }
}
