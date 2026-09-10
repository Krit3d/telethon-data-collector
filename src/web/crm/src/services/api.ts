import type { Social } from '../data';

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

const TOKEN_KEY = 'creatorflow_token';
const USER_KEY = 'creatorflow_user';

const envUrl = import.meta.env.VITE_API_URL;
const BASE_URL: string =
  envUrl && String(envUrl).trim() !== ''
    ? String(envUrl)
    : typeof window !== 'undefined'
      ? `${window.location.protocol}//${window.location.hostname}:8000/api/v1/crm`
      : 'http://localhost:8000/api/v1/crm';

const API_BASE = BASE_URL.replace(/\/+$/, '');

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

  async getCreators(): Promise<CreatorRecord[]> {
    const data = await request<CreatorsResponse>('/creators');
    return data.data;
  },

  async updateCreatorStatus(creatorId: string, status: string): Promise<CreatorRecord> {
    return request<CreatorRecord>(`/creators/${encodeURIComponent(creatorId)}`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    });
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
};

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

export const STATUS_OPTIONS: string[] = ['Свободен', 'В сделке', 'На паузе'];

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
