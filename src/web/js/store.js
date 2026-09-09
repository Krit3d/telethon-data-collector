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

export const DEFAULT_COUNTRIES = {
  RU: "Россия",
  CN: "Китай",
  US: "США",
  AE: "ОАЭ",
  BY: "Беларусь",
  KZ: "Казахстан",
  UZ: "Узбекистан",
  KG: "Кыргызстан",
  TJ: "Таджикистан",
  AM: "Армения",
  AZ: "Азербайджан",
  GE: "Грузия",
  TR: "Турция",
  DE: "Германия",
  FR: "Франция",
  IT: "Италия",
  ES: "Испания",
  GB: "Великобритания",
};

function normalizeCountryCode(value) {
  return String(value || "").trim().toUpperCase();
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
    this.countries = { ...DEFAULT_COUNTRIES };
    this.countryAliases = {};
    this.selectedCountries = [];
    this.selectedGender = "both";
    this.isGenderManuallySet = false;
    this.isCountryManuallySet = false;
    this.selectedLanguages = [];
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

  get selectedLanguage() {
    return this.selectedLanguages.length > 0 ? this.selectedLanguages[0] : "all";
  }

  toggleLanguage(code) {
    const normalized = String(code || "").trim().toLowerCase();
    if (!normalized || normalized === "all") {
      this.selectedLanguages = [];
      return;
    }
    const index = this.selectedLanguages.indexOf(normalized);
    if (index >= 0) {
      this.selectedLanguages.splice(index, 1);
      return;
    }
    this.selectedLanguages.push(normalized);
  }

  setLanguages(codes) {
    if (!Array.isArray(codes)) return;
    const unique = [];
    for (const raw of codes) {
      const normalized = String(raw || "").trim().toLowerCase();
      if (!normalized || normalized === "all") continue;
      if (!unique.includes(normalized)) {
        unique.push(normalized);
      }
    }
    this.selectedLanguages = unique;
  }

  initCountries(data) {
    if (data && Array.isArray(data.countries)) {
      const map = {};
      for (const item of data.countries) {
        const code = normalizeCountryCode(item.code);
        if (!code) continue;
        map[code] = item.name_ru || item.name_en || code;
      }
      this.countries = { ...DEFAULT_COUNTRIES, ...map };
    }
    if (data && data.aliases) {
      this.countryAliases = data.aliases;
    }
  }

  toggleCountry(code) {
    const normalized = normalizeCountryCode(code);
    if (!normalized || normalized === "ALL") {
      this.selectedCountries = [];
      this.isCountryManuallySet = true;
      return;
    }
    const index = this.selectedCountries.indexOf(normalized);
    if (index >= 0) {
      this.selectedCountries.splice(index, 1);
    } else {
      this.selectedCountries.push(normalized);
    }
    this.isCountryManuallySet = true;
  }

  setCountries(codes) {
    if (!Array.isArray(codes)) return;
    const unique = [];
    for (const raw of codes) {
      const normalized = normalizeCountryCode(raw);
      if (!normalized || normalized === "ALL") continue;
      if (!this.countries[normalized] && !DEFAULT_COUNTRIES[normalized] && normalized.length !== 2) continue;
      if (!unique.includes(normalized)) {
        unique.push(normalized);
      }
    }
    this.selectedCountries = unique;
  }

  setGender(gender) {
    const normalized = String(gender || "").trim().toLowerCase();
    this.selectedGender = ["male", "female", "both"].includes(normalized) ? normalized : "both";
    this.isGenderManuallySet = true;
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
    if (!this.isCountryManuallySet && (Array.isArray(filters.countries) || Array.isArray(filters.target_countries))) {
      this.setCountries(filters.countries || filters.target_countries);
    }
    if (!this.isGenderManuallySet && filters.target_gender) {
      const normalized = String(filters.target_gender).trim().toLowerCase();
      this.selectedGender = ["male", "female", "both"].includes(normalized) ? normalized : "both";
    }
    if (Array.isArray(filters.languages)) {
      this.setLanguages(filters.languages);
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
      countries: this.selectedCountries.length > 0 ? this.selectedCountries : null,
      gender: this.isGenderManuallySet || this.selectedGender !== "both" ? this.selectedGender : null,
      languages: this.selectedLanguages.length > 0 ? this.selectedLanguages : null,
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
    this.selectedCountries = [];
    this.selectedGender = "both";
    this.isGenderManuallySet = false;
    this.isCountryManuallySet = false;
    this.selectedLanguages = [];
    this.selectedTone = "all";
    this.selectedHormones = [];
    this.minFollowers = null;
    this.maxFollowers = null;
    this.stopTopicsInput = "";
    this.sortFilter = "relevance";
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
