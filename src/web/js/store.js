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

const COUNTRY_ALIASES = {
  "казахстан": "kz",
  "kazakhstan": "kz",
  "kz": "kz",
  "россия": "ru",
  "russia": "ru",
  "рф": "ru",
  "ru": "ru",
  "беларусь": "by",
  "belarus": "by",
  "by": "by",
  "узбекистан": "uz",
  "uzbekistan": "uz",
  "uz": "uz",
  "оаэ": "ae",
  "uae": "ae",
  "эмираты": "ae",
  "ae": "ae",
  "сша": "us",
  "usa": "us",
  "америка": "us",
  "us": "us",
};

function mapCountry(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) return "all";
  return COUNTRY_ALIASES[normalized] || "all";
}

export class AppStore {
  constructor() {
    this.activeTab = "search";
    this.searchQuery = "";
    this.brandDescription = "";
    this.targetAudienceDescription = "";
    this.directCluster = null;
    this.audienceClusters = [];
    this.isAudienceConfirmed = false;
    this.isAnalyzingBrand = false;
    this.currentStep = 1;
    this.selectedCountry = "all";
    this.selectedLanguage = "all";
    this.minFollowers = null;
    this.maxFollowers = null;
    this.platformFilter = "all";
    this.sortFilter = "relevance";
    this.reachFilter = "all";
    this.authorType = "expert";
    this.matchTypeFilter = "all";
    this.selectedTone = "all";
    this.selectedHormones = [];
    this.stopTopicsInput = "";
    this.precomputedPlan = null;
    this.inferredFilters = null;
    this.searchResults = [];
    this.queryMetadata = null;
    this.shortlist = readStorage(SHORTLIST_KEY, []);
    this.threads = readStorage(THREADS_KEY, []);
    this.activeThreadId = null;
  }

  applyBrandAnalysis(data) {
    if (!data) return;
    this.targetAudienceDescription = data.target_audience_description || "";
    this.searchQuery = data.target_audience_description || "";
    this.directCluster = data.direct_cluster || null;
    this.audienceClusters = Array.isArray(data.audience_clusters) ? data.audience_clusters : [];
    this.isAudienceConfirmed = true;
    this.currentStep = 2;
    this.applyInferredFilters(data.inferred_filters);
  }

  applyInferredFilters(filters) {
    if (!filters) return;
    if (filters.target_tone) {
      this.selectedTone = filters.target_tone;
    }
    if (Array.isArray(filters.target_hormones)) {
      this.selectedHormones = filters.target_hormones.slice(0, 2);
    }
    if (Array.isArray(filters.stop_topics)) {
      this.stopTopicsInput = filters.stop_topics.join(", ");
    }
    if (filters.country && String(filters.country).trim() !== "") {
      this.selectedCountry = mapCountry(String(filters.country));
    }
    if (Array.isArray(filters.languages) && filters.languages.length > 0) {
      this.selectedLanguage = filters.languages[0];
    }
    if (filters.min_followers != null) {
      this.minFollowers = filters.min_followers;
    }
    if (filters.max_followers != null) {
      this.maxFollowers = filters.max_followers;
    }
    this.inferredFilters = filters;
  }

  buildSearchRequest(query) {
    return {
      query: (this.targetAudienceDescription || query || this.brandDescription || "").trim(),
      limit: 40,
      author_type: this.authorType || "expert",
      platform: this.platformFilter || "all",
      min_followers: (this.minFollowers && Number(this.minFollowers) > 0) ? Number(this.minFollowers) : null,
      max_followers: (this.maxFollowers && Number(this.maxFollowers) > 0) ? Number(this.maxFollowers) : null,
      location: this.selectedCountry !== "all" ? this.selectedCountry : null,
      languages: this.selectedLanguage !== "all" ? [this.selectedLanguage] : null,
      target_tone: this.selectedTone !== "all" ? this.selectedTone : null,
      target_hormones: this.selectedHormones,
      stop_topics: this.stopTopicsInput ? this.stopTopicsInput.split(",").map((s) => s.trim()).filter(Boolean) : [],
      direct_cluster: this.directCluster || null,
      audience_clusters: (Array.isArray(this.audienceClusters) && this.audienceClusters.length > 0) ? this.audienceClusters : [],
      precomputed_plan: this.precomputedPlan || null,
      include_contacts: false,
      include_analytics: true,
    };
  }

  resetAudienceState() {
    this.currentStep = 1;
    this.isAudienceConfirmed = false;
    this.targetAudienceDescription = "";
    this.directCluster = null;
    this.audienceClusters = [];
    this.precomputedPlan = null;
    this.searchResults = [];
    this.queryMetadata = null;
    this.selectedCountry = "all";
    this.selectedLanguage = "all";
    this.selectedTone = "all";
    this.selectedHormones = [];
    this.minFollowers = null;
    this.maxFollowers = null;
    this.stopTopicsInput = "";
    this.searchQuery = this.brandDescription || "";
  }

  toggleHormone(hormone) {
    const index = this.selectedHormones.indexOf(hormone);
    if (index >= 0) {
      this.selectedHormones.splice(index, 1);
      return;
    }
    if (this.selectedHormones.length < 2) {
      this.selectedHormones.push(hormone);
      return;
    }
    this.selectedHormones.shift();
    this.selectedHormones.push(hormone);
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
