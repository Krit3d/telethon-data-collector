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
  id: number;
  platform: string;
  username: string | null;
  title: string;
  subscribers_count: number | null;
  static_avg_er: number | null;
  category_path: string | null;
}

export interface DealMessageItem {
  id: number;
  deal_id: number;
  sender_type: 'user' | 'creator' | 'system' | string;
  text: string;
  is_read: boolean;
  created_at: string;
}

export interface DealItem {
  id: number;
  user_id: number;
  account_id: number;
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
  account_id: string | number;
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
  deal_id: number;
  author_id: number;
  author_name: string;
  author_handle: string;
  platform: string;
  deal_title: string;
  stage: number;
  last_message: string;
  last_message_time: string;
  unread_count: number;
}

const TOKEN_KEY = 'creatorflow_token';
const USER_KEY = 'creatorflow_user';

const API_BASE = typeof window !== 'undefined'
  ? `${window.location.protocol}//${window.location.hostname}:8000/api/v1/crm`
  : 'http://localhost:8000/api/v1/crm';

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

  async exportToCrmShortlist(accountIds: string[], userEmail: string): Promise<{ added_count: number; redirect_url: string }> {
    return request<{ added_count: number; redirect_url: string }>('/shortlist', {
      method: 'POST',
      body: JSON.stringify({ account_ids: accountIds, user_email: userEmail }),
    });
  },

  async exportShortlist(accountIds: string[], userEmail?: string): Promise<{ added_count: number; redirect_url: string }> {
    const currentUser = getStoredUser();
    const email = userEmail || (typeof currentUser?.email === 'string' ? currentUser.email : '');
    return request<{ added_count: number; redirect_url: string }>('/shortlist', {
      method: 'POST',
      body: JSON.stringify({ account_ids: accountIds, user_email: email }),
    });
  },

  async updateCreatorStatus(creatorId: string, status: string): Promise<CreatorRecord> {
    return request<CreatorRecord>(`/creators/${encodeURIComponent(creatorId)}`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
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

  logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
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
    msgs: item.last_message ? [{ id: String(item.last_message.id), from: item.last_message.sender_type === 'creator' ? 'author' : 'user', text: item.last_message.text, time: new Date(item.last_message.created_at).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) }] : [],
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

export const STATUS_OPTIONS: string[] = ['Свободен', 'В сделке', 'На паузе', 'В архиве'];

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
