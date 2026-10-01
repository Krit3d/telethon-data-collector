import type { Deal, Social } from '../data';

export interface CreatorRecord {
  id: string;
  accountid?: string | null;
  accountId?: string | null;
  handle?: string | null;
  username?: string | null;
  name?: string | null;
  Name?: string | null;
  platform?: string | null;
  Platform?: string | null;
  followers?: number | null;
  subscribers_count?: number | null;
  avgreach?: number | null;
  avgReach?: number | null;
  avg_reach?: number | null;
  er?: number | null;
  static_avg_er?: number | null;
  cpm?: number | null;
  niche?: string | null;
  category_path?: string | null;
  status?: string | null;
  hue?: number | null;
  dealscount?: number | null;
  dealsCount?: number | null;
  title?: string | null;
  source?: 'search' | 'manual' | string | null;
  account_status?: string | null;
  accountStatus?: string | null;
}

export interface ManualCreatorPayload {
  platform: 'Instagram' | 'Telegram' | string;
  username: string;
  title?: string;
  subscribers_count?: number;
  telegram_commercial?: string;
  telegram_personal?: string;
  email?: string;
  phone?: string;
}

export interface CreatorPostItem {
  id: number;
  platform_content_id: string;
  text: string | null;
  published_at: string;
  views: number;
  likes: number;
  comments: number;
  shares: number;
  er: number;
  post_type: string;
  url: string | null;
}

export interface CreatorProfileDetail {
  id: string;
  platform: string;
  username: string | null;
  title: string;
  description: string | null;
  subscribers_count: number;
  static_avg_er: number;
  category_path: string | null;
  country: string | null;
  city: string | null;
  gender: string | null;
  status: string;
  profile_url: string;
  cpm: number;
  avg_reach: number;
  deals_count: number;
  posts: CreatorPostItem[];
  source?: 'search' | 'manual' | string | null;
  account_status?: string | null;
}

export interface LoginResponse {
  token: string;
  user: Record<string, unknown>;
}

export interface CreatorsResponse {
  data: CreatorRecord[];
  total: number;
}

export interface DealAuthorSummary {
  id: string;
  platform: string;
  username: string | null;
  title: string;
  subscribers_count: number | null;
  static_avg_er: number | null;
  category_path: string | null;
}

export interface FileUploadResponse {
  media_url: string;
  media_name: string;
  media_type: string;
  file_size: number;
}

export interface DealMessageItem {
  id: number;
  deal_id?: number | null;
  account_id?: string;
  sender_type: 'user' | 'creator' | 'system' | string;
  text?: string | null;
  is_read: boolean;
  created_at: string;
  channel_type?: string | null;
  channel_target?: string | null;
  external_message_id?: string | null;
  media_url?: string | null;
  media_name?: string | null;
  media_type?: string | null;
}

export interface DealItem {
  id: number;
  user_id: number;
  account_id: string;
  title: string;
  stage: number;
  budget: number;
  type: string;
  brand_name: string | null;
  pub_date: string | null;
  terms: string | null;
  created_at: string;
  updated_at: string;
  author: DealAuthorSummary | null;
  last_message: DealMessageItem | null;
  unread_count: number;
}

export interface DealCreatePayload {
  account_id: string;
  title: string;
  budget?: number;
  type?: string;
  brand_name?: string | null;
  pub_date?: string | null;
  terms?: string | null;
  initial_message?: string | null;
}

export interface DealUpdatePayload {
  title?: string;
  stage?: number;
  budget?: number;
  type?: string;
  brand_name?: string | null;
  pub_date?: string | null;
  terms?: string | null;
}

export interface CommunicationChannelItem {
  deal_id: number | null;
  author_id: string;
  author_name: string;
  author_handle: string;
  platform: string;
  deal_title: string | null;
  stage: number | null;
  last_message: string;
  last_message_time: string;
  unread_count: number;
  is_archived: boolean;
  channel_type?: 'telegram' | 'email' | 'whatsapp' | 'internal' | string | null;
}

const TOKEN_KEY = 'creatorflow_token';
const USER_KEY = 'creatorflow_user';

const ROOT_API_URL = (import.meta.env.VITE_API_URL || '/api/v1').replace(/\/+$/, '');
const API_BASE = `${ROOT_API_URL.replace(/\/crm\/?$/, '')}/crm`;
export const MEDIA_SERVER_BASE = ROOT_API_URL.replace(/\/api(\/v\d+)?.*$/, '');

export function resolveMediaUrl(path?: string | null): string {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  return `${MEDIA_SERVER_BASE}/${path.replace(/^\/+/, '')}`;
}

class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...(init.headers as Record<string, string> | undefined) };
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (response.status === 401) {
    api.logout();
    window.dispatchEvent(new CustomEvent('creatorflow:auth_expired'));
    throw new ApiError('Сессия истекла. Пожалуйста, войдите снова.', 401);
  }
  if (!response.ok) {
    let detail = `Ошибка запроса (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* ignore parse failures */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

function getStoredUser(): Record<string, unknown> | null {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export const api = {
  async login(email: string, password: string): Promise<LoginResponse> {
    const data = await request<LoginResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    localStorage.setItem(TOKEN_KEY, data.token);
    localStorage.setItem(USER_KEY, JSON.stringify(data.user));
    return data;
  },

  async register(email: string, password: string, name?: string): Promise<LoginResponse> {
    const data = await request<LoginResponse>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, name }),
    });
    localStorage.setItem(TOKEN_KEY, data.token);
    localStorage.setItem(USER_KEY, JSON.stringify(data.user));
    return data;
  },

  async getCreators(): Promise<CreatorRecord[]> {
    const data = await request<CreatorsResponse>('/creators');
    return data.data;
  },

  async getCreatorProfile(creatorId: string): Promise<CreatorProfileDetail> {
    return request<CreatorProfileDetail>(`/creators/${encodeURIComponent(creatorId)}`);
  },

  async addCreatorManual(payload: ManualCreatorPayload): Promise<CreatorRecord> {
    return request<CreatorRecord>('/creators/manual', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async exportToCrmShortlist(accountIds: string[], userEmail: string): Promise<{ added_count: number; redirect_url: string }> {
    return request<{ added_count: number; redirect_url: string }>('/shortlist', {
      method: 'POST',
      body: JSON.stringify({ account_ids: accountIds, user_email: userEmail }),
    });
  },

  async exportShortlist(accountIds: string[]): Promise<{ added_count: number; redirect_url: string }> {
    return request<{ added_count: number; redirect_url: string }>('/shortlist', {
      method: 'POST',
      body: JSON.stringify({ account_ids: accountIds }),
    });
  },

  async updateCreatorStatus(creatorId: string, status: string, archiveActiveDeals?: boolean): Promise<CreatorRecord> {
    return request<CreatorRecord>(`/creators/${encodeURIComponent(creatorId)}`, {
      method: 'PATCH',
      body: JSON.stringify({ status, archive_active_deals: archiveActiveDeals ?? false }),
    });
  },

  async deleteCreator(creatorId: string): Promise<{ status: string; creator_id: string }> {
    return request<{ status: string; creator_id: string }>(`/creators/${encodeURIComponent(creatorId)}`, {
      method: 'DELETE',
    });
  },

  async getDeals(params?: { stage?: number; search?: string }): Promise<DealItem[]> {
    const query = new URLSearchParams();
    if (params?.stage !== undefined) query.set('stage', String(params.stage));
    if (params?.search) query.set('search', params.search);
    const suffix = query.toString() ? `?${query.toString()}` : '';
    return request<DealItem[]>(`/deals${suffix}`);
  },

  async createDeal(payload: DealCreatePayload): Promise<DealItem> {
    return request<DealItem>('/deals', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async getDeal(dealId: number): Promise<DealItem> {
    return request<DealItem>(`/deals/${dealId}`);
  },

  async updateDeal(dealId: number, payload: DealUpdatePayload): Promise<DealItem> {
    return request<DealItem>(`/deals/${dealId}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    });
  },

  async deleteDeal(dealId: number): Promise<{ status: string; deal_id: number }> {
    return request<{ status: string; deal_id: number }>(`/deals/${dealId}`, {
      method: 'DELETE',
    });
  },

  async getDealMessages(dealId: number): Promise<DealMessageItem[]> {
    return request<DealMessageItem[]>(`/deals/${dealId}/messages`);
  },

  async sendDealMessage(dealId: number, text: string, senderType: string = 'user'): Promise<DealMessageItem> {
    return request<DealMessageItem>(`/deals/${dealId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ text, sender_type: senderType }),
    });
  },

  async getCommunications(): Promise<CommunicationChannelItem[]> {
    return request<CommunicationChannelItem[]>('/communications');
  },

  async initCommunication(creatorId: string): Promise<CommunicationChannelItem> {
    return request<CommunicationChannelItem>(`/communications/init?creator_id=${encodeURIComponent(creatorId)}`, {
      method: 'POST',
    });
  },

  async getCreatorMessages(creatorId: string): Promise<DealMessageItem[]> {
    return request<DealMessageItem[]>(`/creators/${encodeURIComponent(creatorId)}/messages`);
  },

  async uploadChatFile(file: File): Promise<FileUploadResponse> {
    const form = new FormData();
    form.append('file', file);
    const headers: Record<string, string> = {};
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const response = await fetch(`${API_BASE}/upload`, { method: 'POST', headers, body: form });
    if (response.status === 401) {
      api.logout();
      window.dispatchEvent(new CustomEvent('creatorflow:auth_expired'));
      throw new ApiError('Сессия истекла. Пожалуйста, войдите снова.', 401);
    }
    if (!response.ok) {
      let detail = `Ошибка запроса (${response.status})`;
      try {
        const body = (await response.json()) as { detail?: unknown };
        if (typeof body.detail === 'string') detail = body.detail;
      } catch {
        /* ignore parse failures */
      }
      throw new ApiError(detail, response.status);
    }
    return (await response.json()) as FileUploadResponse;
  },

  async sendCreatorMessage(creatorId: string, text?: string | null, senderType: string = 'user', media?: { media_url: string; media_name: string; media_type: string }): Promise<DealMessageItem> {
    return request<DealMessageItem>(`/creators/${encodeURIComponent(creatorId)}/messages`, {
      method: 'POST',
      body: JSON.stringify({
        text: text?.trim() || null,
        sender_type: senderType,
        media_url: media?.media_url || null,
        media_name: media?.media_name || null,
        media_type: media?.media_type || null,
      }),
    });
  },

  logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    window.dispatchEvent(new CustomEvent('creatorflow:logout'));
  },

  isAuthenticated(): boolean {
    return Boolean(localStorage.getItem(TOKEN_KEY));
  },

  getCurrentUser(): Record<string, unknown> | null {
    return getStoredUser();
  },

  getToken(): string {
    return localStorage.getItem(TOKEN_KEY) ?? '';
  },
};

export function mapDealItemToDeal(item: DealItem): Deal {
  const authorTitle = item.author ? (item.author.username || item.author.title) : '';
  return {
    id: String(item.id),
    title: item.title || authorTitle,
    brandId: item.brand_name ? String(item.brand_name) : 'b1',
    authorId: String(item.account_id),
    authorSummary: item.author,
    budget: item.budget,
    stage: item.stage,
    date: new Date(item.created_at).toLocaleDateString('ru-RU'),
    pubDate: item.pub_date || new Date(item.created_at).toLocaleDateString('ru-RU'),
    type: item.type as Deal['type'],
    msgs: item.last_message ? [{ id: String(item.last_message.id), from: item.last_message.sender_type === 'creator' ? 'author' : 'user', text: item.last_message.text || '', time: new Date(item.last_message.created_at).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) }] : [],
    desc: item.terms || '',
    terms: item.terms || `Фиксированная оплата ${item.budget} ₽`,
    exclusive: false,
    edits: 2,
  };
}

export function userName(user: Record<string, unknown> | null): string {
  if (!user) return '';
  const email = typeof user.email === 'string' ? user.email : '';
  const name = typeof user.name === 'string' && user.name ? user.name : '';
  return name || email;
}

export function userHandle(user: Record<string, unknown> | null): string {
  const email = typeof user?.email === 'string' && user.email ? user.email : '';
  return email ? '@' + email.split('@')[0] : '';
}

export const STATUS_OPTIONS: string[] = ['Свободен', 'В сделке', 'В архиве'];

export function normalizeSocial(raw: string | null | undefined): Social {
  const value = (raw ?? '').trim();
  const lower = value.toLowerCase();
  if (lower === 'instagram') return 'Instagram';
  if (lower === 'youtube') return 'YouTube';
  if (lower === 'telegram') return 'Telegram';
  if (lower === 'tiktok') return 'TikTok';
  if (lower === 'vk' || lower === 'вконтакте') return 'VK';
  return 'Дзен';
}
