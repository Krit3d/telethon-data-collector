import { Archive, ArchiveRestore, ArrowDown, ArrowUp, ArrowUpDown, Briefcase, ChevronLeft, ChevronRight, MessageSquare, MoreVertical, RefreshCw, Search, Sparkles, Trash2, Users, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { SocialIcon } from '../components/icons';
import { Avatar, Badge, Btn, Card, inputCls, Modal, Toggle, useToast } from '../components/ui';
import { fmtMoney, type Author, type Deal, type Social } from '../data';
import { api, normalizeSocial, STATUS_OPTIONS, type CreatorRecord } from '../services/api';

type SortKey = 'nick' | 'social' | 'followers' | 'niche' | 'reach' | 'er' | 'cpm' | 'status';
type SortOrder = 'asc' | 'desc';

type ActiveStatus = Author['status'];
const statusTone: Record<ActiveStatus, string> = { 'Свободен': 'green', 'В сделке': 'blue', 'В архиве': 'gray' };
const SOCIALS: Social[] = ['Instagram', 'VK', 'Telegram', 'TikTok', 'YouTube', 'Дзен'];
const PAGE = 15;
const DEFAULT_SEARCH_PORT = 8000;

function getSearchBaseUrl(): string {
  const host = window.location.hostname;
  if (host) return `${window.location.protocol}//${host}:${DEFAULT_SEARCH_PORT}`;
  return `http://localhost:${DEFAULT_SEARCH_PORT}`;
}

const fmtER = (n: number) => `${n.toFixed(1).replace('.', ',')}%`;
const fmtBig = (n: number) => n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1).replace('.0', '').replace('.', ',')}M` : n.toLocaleString('ru-RU');

function computedStatus(a: Author, dealsCount: number): ActiveStatus {
  if (a.status === 'В архиве') return 'В архиве';
  if (dealsCount > 0) return 'В сделке';
  return 'Свободен';
}

function formatNiche(niche: string): string {
  const parts = niche.split('>');
  if (parts.length <= 1) return niche;
  return parts.slice(-2).map((p: string) => p.trim()).join(' › ');
}

function hashHue(seed: string): number {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) & 0x7fffffff;
  return h % 360;
}

function toAuthor(rec: CreatorRecord): Author {
  const social = normalizeSocial(rec.platform || rec.Platform || 'Instagram');
  const followers = Number(rec.followers ?? rec.subscribers_count ?? 0);
  const er = Number(rec.er ?? rec.static_avg_er ?? 0);
  const status = STATUS_OPTIONS.includes(rec.status ?? '') ? (rec.status as Author['status']) : 'Свободен';
  const accountId = String(rec.accountid || rec.accountId || rec.id || '');
  return {
    id: accountId,
    twentyId: rec.id,
    nick: rec.name || rec.Name || rec.handle || rec.username || 'Без имени',
    social,
    followers,
    niche: rec.niche || rec.category_path || 'Общее',
    reach: Number(rec.avgreach ?? rec.avgReach ?? 0),
    cpm: Number(rec.cpm ?? 0),
    er,
    roas: 0,
    status,
    hue: Number(rec.hue ?? hashHue(rec.id)),
    deals: Number(rec.dealscount ?? rec.dealsCount ?? 0),
  };
}

function getPendingImportIds(): string[] {
  const hash = window.location.hash;
  const search = window.location.search;
  const fromSearch = new URLSearchParams(search).get('import_ids');
  const raw = fromSearch ?? (hash.includes('?') ? new URLSearchParams(hash.substring(hash.indexOf('?'))).get('import_ids') : null);
  if (!raw) return [];
  const decoded = decodeURIComponent(raw);
  const ids = decoded.split(',').map((item: string) => decodeURIComponent(item).trim().replace(/^@/, '')).filter(Boolean);
  return Array.from(new Set(ids));
}

export default function Authors({ deals, onOpenProfile, onNewDeal, onOpenComms }: { deals: Deal[]; onOpenProfile: (id: string) => void; onNewDeal: (authorId: string, meta?: { name?: string; handle?: string }) => void; onOpenComms: (authorId: string) => void }) {
  const toast = useToast();
  const [authors, setAuthors] = useState<Author[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [q, setQ] = useState('');
  const [fSocial, setFSocial] = useState('all');
  const [fMin, setFMin] = useState('');
  const [fMax, setFMax] = useState('');
  const [onlyFree, setOnlyFree] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const [page, setPage] = useState(0);
  const [sortKey, setSortKey] = useState<SortKey | null>(null);
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [menu, setMenu] = useState<{ id: string; top: number; left: number } | null>(null);
  const [confirmDeleteAuthor, setConfirmDeleteAuthor] = useState<Author | null>(null);
  const [authorToArchive, setAuthorToArchive] = useState<Author | null>(null);
  const [closeDeals, setCloseDeals] = useState(true);

  const load = async () => {
    setLoading(true);
    setLoadError('');
    try {
      const records = await api.getCreators();
      setAuthors(records.map(toAuthor));
    } catch (err) {
      setLoadError(err instanceof Error && err.message ? err.message : 'Не удалось загрузить авторов');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void load(); }, []);

  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    window.addEventListener('scroll', close, true);
    window.addEventListener('resize', close);
    return () => {
      window.removeEventListener('scroll', close, true);
      window.removeEventListener('resize', close);
    };
  }, [menu]);

  const openSearch = () => {
    const baseUrl = getSearchBaseUrl();
    const token = window.localStorage.getItem('creatorflow_token');
    if (!token) {
      window.open(baseUrl, '_blank');
      return;
    }
    const params = new URLSearchParams();
    params.set('session_token', token);
    let user: { id?: unknown; email?: unknown } | null = null;
    try {
      const raw = window.localStorage.getItem('creatorflow_user');
      user = raw ? JSON.parse(raw) as { id?: unknown; email?: unknown } : null;
    } catch {
      user = null;
    }
    if (user) {
      if (user.id != null) params.set('user_id', String(user.id));
      if (user.email != null) params.set('user_email', String(user.email));
    }
    window.open(`${baseUrl}?${params.toString()}`, '_blank');
  };

  useEffect(() => {
    const ids = getPendingImportIds();
    if (ids.length === 0) return;
    const run = async () => {
      try {
        const res = await api.exportShortlist(ids);
        await load();
        toast('ok', `Успешно импортировано авторов: ${res?.added_count ?? ids.length}`);
      } catch (err) {
        toast('err', err instanceof Error ? err.message : 'Ошибка импорта');
      } finally {
        window.history.replaceState(null, '', `${window.location.pathname}#/authors`);
      }
    };
    void run();
  }, []);

  const dealsCountFor = (a: Author): number => deals.filter(d => String(d.authorId) === String(a.id) || (a.twentyId && String(d.authorId) === String(a.twentyId))).length;

  const activeDealsCountFor = (a: Author): number => deals.filter(d => (String(d.authorId) === String(a.id) || (a.twentyId && String(d.authorId) === String(a.twentyId))) && d.stage >= 1 && d.stage <= 6).length;

  const changeStatus = async (a: Author, status: Author['status']) => {
    try {
      await api.updateCreatorStatus(a.twentyId || a.id, status);
      setAuthors(prev => prev.map(x => x.id === a.id ? { ...x, status } : x));
      toast('ok', `Статус ${a.nick} обновлён: ${status}`);
    } catch (err) {
      toast('err', err instanceof Error && err.message ? err.message : 'Не удалось обновить статус');
    }
  };

  const handleDeleteAuthor = async (a: Author) => {
    setMenu(null);
    setConfirmDeleteAuthor(null);
    if (dealsCountFor(a) > 0) {
      toast('err', 'Нельзя удалить автора со сделками или перепиской. Отправьте его в архив.');
      return;
    }
    try {
      await api.deleteCreator(a.twentyId || a.id);
      setAuthors(prev => prev.filter(x => x.id !== a.id));
      toast('ok', `${a.nick} удалён из шортлиста`);
    } catch (err) {
      toast('err', err instanceof Error && err.message ? err.message : 'Не удалось удалить автора');
    }
  };

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(o => o === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortOrder(['followers', 'reach', 'er', 'cpm'].includes(key) ? 'desc' : 'asc');
    }
    setPage(0);
  };

  const list = useMemo(() => {
    const filtered = authors.filter(a =>
      (showArchived ? a.status === 'В архиве' : a.status !== 'В архиве') &&
      (!q || a.nick.toLowerCase().includes(q.toLowerCase()) || a.niche.toLowerCase().includes(q.toLowerCase())) &&
      (fSocial === 'all' || a.social === fSocial) &&
      (!fMin || a.followers >= Number(fMin) * 1000) &&
      (!fMax || a.followers <= Number(fMax) * 1000) &&
      (!onlyFree || computedStatus(a, activeDealsCountFor(a)) === 'Свободен')
    );
    if (!sortKey) return filtered;
    const numeric = ['followers', 'reach', 'er', 'cpm'].includes(sortKey);
    return [...filtered].sort((x, y) => {
      if (numeric) {
        const vA = x[sortKey] as number;
        const vB = y[sortKey] as number;
        return sortOrder === 'asc' ? vA - vB : vB - vA;
      }
      const sA = sortKey === 'status' ? computedStatus(x, activeDealsCountFor(x)) : String(x[sortKey]);
      const sB = sortKey === 'status' ? computedStatus(y, activeDealsCountFor(y)) : String(y[sortKey]);
      return sortOrder === 'asc' ? sA.localeCompare(sB, 'ru') : sB.localeCompare(sA, 'ru');
    });
  }, [authors, q, fSocial, fMin, fMax, onlyFree, showArchived, sortKey, sortOrder]);

  const pages = Math.max(1, Math.ceil(list.length / PAGE));
  const pageItems = list.slice(page * PAGE, (page + 1) * PAGE);
  const freeCount = authors.filter(a => computedStatus(a, activeDealsCountFor(a)) === 'Свободен').length;

  return (
    <div className="h-full overflow-y-auto scroll-thin" onClick={() => setMenu(null)}>
      <div className="px-6 pt-5 pb-4 bg-gradient-to-b from-white to-transparent">
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-[22px] font-display font-semibold text-gray-900">Авторы</h1>
            <p className="text-[12.5px] font-medium text-gray-400 mt-1">База блогеров и площадок · {authors.length} в базе, {freeCount} свободны</p>
          </div>
          <div className="flex items-center gap-2">
            <Btn variant="secondary" onClick={() => void load()}><RefreshCw size={14} />Обновить</Btn>
            <Btn onClick={openSearch}><Sparkles size={14} />AI-подбор авторов</Btn>
          </div>
        </div>

        <div className="flex items-center gap-2 mt-4 flex-wrap">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300" />
            <input className={inputCls + ' !w-56 !pl-9'} placeholder="Поиск по автору или нише…" value={q} onChange={e => { setQ(e.target.value); setPage(0); }} />
          </div>
          <select className={selCls} value={fSocial} onChange={e => { setFSocial(e.target.value); setPage(0); }}>
            <option value="all">Все соцсети</option>
            {SOCIALS.map(s => <option key={s}>{s}</option>)}
          </select>
          <div className="flex items-center gap-1.5 text-[12px] font-semibold text-gray-400">
            <input className={inputCls + ' !w-24'} placeholder="от, K" type="number" value={fMin} onChange={e => { setFMin(e.target.value); setPage(0); }} />
            —
            <input className={inputCls + ' !w-24'} placeholder="до, K" type="number" value={fMax} onChange={e => { setFMax(e.target.value); setPage(0); }} />
          </div>
          <label className="flex items-center gap-2 text-[12px] font-bold text-gray-500 cursor-pointer pl-1">
            <Toggle on={onlyFree} onChange={v => { setOnlyFree(v); setPage(0); }} />Только свободные
          </label>
          <label className="flex items-center gap-2 text-[12px] font-bold text-gray-500 cursor-pointer pl-1">
            <Toggle on={showArchived} onChange={v => { setShowArchived(v); setPage(0); }} />Архив
          </label>
          {(q || fSocial !== 'all' || fMin || fMax || onlyFree || showArchived) && (
            <button className="text-[12px] font-bold text-indigo-600 hover:text-indigo-700 inline-flex items-center gap-1"
              onClick={() => { setQ(''); setFSocial('all'); setFMin(''); setFMax(''); setOnlyFree(false); setShowArchived(false); setPage(0); }}>
              <X size={12} />Сбросить
            </button>
          )}
        </div>
      </div>

      <div className="px-6 pb-6">
        {loading ? (
          <Card className="overflow-hidden">
            <div className="flex flex-col gap-3 px-4 py-6">
              {Array.from({ length: 6 }, (_, i) => (
                <div key={i} className="h-10 rounded-lg bg-slate-100 animate-pulse" />
              ))}
            </div>
          </Card>
        ) : loadError ? (
          <Card className="p-10 text-center">
            <div className="w-12 h-12 rounded-xl bg-red-50 text-red-500 flex items-center justify-center mx-auto"><Users size={22} /></div>
            <div className="text-[14px] font-bold text-gray-800 mt-3">Не удалось загрузить авторов</div>
            <div className="text-[12.5px] font-medium text-gray-400 mt-1">{loadError}</div>
            <Btn className="mt-4" onClick={() => void load()}><RefreshCw size={14} />Повторить</Btn>
          </Card>
        ) : authors.length === 0 ? (
          <Card className="p-12 text-center">
            <div className="w-14 h-14 rounded-2xl bg-indigo-50 text-indigo-500 flex items-center justify-center mx-auto"><Users size={26} /></div>
            <div className="text-[15px] font-bold text-gray-800 mt-4">Шортлист пуст</div>
            <div className="text-[13px] font-medium text-gray-400 mt-1">Найдите авторов через сервис поиска и добавьте их в шортлист</div>
          </Card>
        ) : (
          <Card className="overflow-visible">
            <div className="overflow-x-auto scroll-thin">
              <table className="w-full text-[12.5px] min-w-[960px]">
                <thead>
                  <tr className="text-left text-[11px] font-bold text-gray-400 uppercase tracking-wide border-b border-gray-100 bg-slate-50/60">
                    {([
                      { key: 'nick', label: 'Автор' },
                      { key: 'social', label: 'Соцсеть' },
                      { key: 'followers', label: 'Подписчики' },
                      { key: 'niche', label: 'Ниша' },
                      { key: 'reach', label: 'Ср. охват' },
                      { key: 'er', label: 'ER' },
                      { key: 'cpm', label: 'CPM' },
                      { key: 'status', label: 'Статус' },
                    ] as { key: SortKey; label: string }[]).map(col => {
                      const active = sortKey === col.key;
                      return (
                        <th key={col.key} className="px-4 py-2.5 whitespace-nowrap">
                          <button onClick={() => toggleSort(col.key)}
                            className="inline-flex items-center gap-1 cursor-pointer select-none group/th hover:text-gray-700">
                            {col.label}
                            {active ? (
                              sortOrder === 'asc'
                                ? <ArrowUp size={12} className="text-indigo-600" />
                                : <ArrowDown size={12} className="text-indigo-600" />
                            ) : (
                              <ArrowUpDown size={12} className="text-gray-300 opacity-0 group-hover/th:opacity-100 transition-opacity" />
                            )}
                          </button>
                        </th>
                      );
                    })}
                    <th className="px-4 py-2.5 whitespace-nowrap" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50">
                  {pageItems.map(a => {
                    const dealsCount = dealsCountFor(a);
                    return (
                      <tr key={a.id} className="group hover:bg-indigo-50/40 transition-colors">
                        <td className="px-4 py-2.5">
                          <button onClick={e => { e.stopPropagation(); onOpenProfile(a.id); }} className="flex items-center gap-2.5 hover:text-indigo-600 transition-colors">
                            <Avatar nick={a.nick} hue={a.hue} size={30} />
                            <span className="text-left">
                              <span className="block font-bold text-gray-800 group-hover:text-indigo-600">{a.nick}</span>
                              <span className="block text-[10.5px] font-semibold text-gray-400">{dealsCount} сделок · ER {fmtER(a.er)}</span>
                            </span>
                          </button>
                        </td>
                        <td className="px-4 py-2.5"><span className="inline-flex items-center gap-1.5 font-semibold text-gray-600"><SocialIcon social={a.social} size={14} />{a.social}</span></td>
                        <td className="px-4 py-2.5 font-bold text-gray-800 tabular-nums">{fmtBig(a.followers)}</td>
                        <td className="px-4 py-2.5"><span title={a.niche}><Badge tone="violet">{formatNiche(a.niche)}</Badge></span></td>
                        <td className="px-4 py-2.5 font-semibold text-gray-500 tabular-nums">{fmtBig(a.reach)}</td>
                        <td className="px-4 py-2.5 font-bold text-emerald-600 tabular-nums">{fmtER(a.er)}</td>
                        <td className="px-4 py-2.5 font-bold text-gray-800 tabular-nums">{a.cpm} ₽</td>
                        <td className="px-4 py-2.5" onClick={e => e.stopPropagation()}>
                          <Badge tone={statusTone[computedStatus(a, activeDealsCountFor(a))]}>{computedStatus(a, activeDealsCountFor(a))}</Badge>
                        </td>
                        <td className="px-4 py-2.5 text-right" onClick={e => e.stopPropagation()}>
                          <button className="w-7 h-7 rounded-lg text-gray-300 group-hover:text-gray-500 hover:bg-gray-100 inline-flex items-center justify-center transition-colors"
                            onClick={e => {
                              e.stopPropagation();
                              if (menu?.id === a.id) {
                                setMenu(null);
                                return;
                              }
                              const rect = e.currentTarget.getBoundingClientRect();
                              const dropdownHeight = 185;
                              const dropdownWidth = 192;
                              const openUp = rect.bottom + dropdownHeight > window.innerHeight;
                              const top = openUp ? Math.max(8, rect.top - dropdownHeight) : rect.bottom + 4;
                              const left = Math.max(8, rect.right - dropdownWidth);
                              setMenu({ id: a.id, top, left });
                            }}><MoreVertical size={15} /></button>
                        </td>
                      </tr>
                    );
                  })}
                  {pageItems.length === 0 && (
                    <tr><td colSpan={9} className="px-4 py-14 text-center text-[13px] font-semibold text-gray-400">Никого не нашли — попробуйте смягчить фильтры</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="flex items-center justify-between px-4 py-3 border-t border-gray-100">
              <span className="text-[12px] font-semibold text-gray-400">Показано {list.length === 0 ? 0 : page * PAGE + 1}–{Math.min((page + 1) * PAGE, list.length)} из {list.length}</span>
              <div className="flex items-center gap-1">
                <button disabled={page === 0} onClick={() => setPage(p => p - 1)} className="w-8 h-8 rounded-lg border border-gray-200 text-gray-500 disabled:opacity-30 hover:bg-gray-50 flex items-center justify-center"><ChevronLeft size={14} /></button>
                {Array.from({ length: pages }, (_, i) => (
                  <button key={i} onClick={() => setPage(i)} className={`w-8 h-8 rounded-lg text-[12px] font-bold transition-colors ${page === i ? 'bg-indigo-500 text-white' : 'text-gray-500 hover:bg-gray-50'}`}>{i + 1}</button>
                ))}
                <button disabled={page === pages - 1} onClick={() => setPage(p => p + 1)} className="w-8 h-8 rounded-lg border border-gray-200 text-gray-500 disabled:opacity-30 hover:bg-gray-50 flex items-center justify-center"><ChevronRight size={14} /></button>
              </div>
            </div>
          </Card>
        )}
      </div>

      {confirmDeleteAuthor && (
        <Modal onClose={() => setConfirmDeleteAuthor(null)} w="max-w-md">
          <div className="px-5 py-4">
            <h3 className="text-[16px] font-bold text-gray-900">Удалить автора из шортлиста?</h3>
            <p className="text-[13px] font-medium text-gray-500 mt-2">Автор @{confirmDeleteAuthor.nick} будет удален из рабочей базы. Это действие нельзя отменить.</p>
          </div>
          <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-gray-100">
            <Btn variant="secondary" onClick={() => setConfirmDeleteAuthor(null)}>Отмена</Btn>
            <Btn variant="danger" onClick={() => void handleDeleteAuthor(confirmDeleteAuthor)}>Удалить</Btn>
          </div>
        </Modal>
      )}

      {authorToArchive && (() => {
        const activeDeals = deals.filter(d => (String(d.authorId) === String(authorToArchive.id) || (authorToArchive.twentyId && String(d.authorId) === String(authorToArchive.twentyId))) && d.stage >= 1 && d.stage <= 6);
        const activeBudget = activeDeals.reduce((acc, d) => acc + d.budget, 0);
        const handleConfirm = async () => {
          try {
            await api.updateCreatorStatus(authorToArchive.twentyId || authorToArchive.id, 'В архиве', activeDeals.length > 0 ? closeDeals : false);
            setAuthors(prev => prev.map(x => x.id === authorToArchive.id ? { ...x, status: 'В архиве' } : x));
            toast('ok', `Автор ${authorToArchive.nick} архивирован`);
          } catch (err) {
            toast('err', err instanceof Error && err.message ? err.message : 'Не удалось архивировать автора');
          } finally {
            setAuthorToArchive(null);
          }
        };
        return (
          <Modal onClose={() => setAuthorToArchive(null)} w="max-w-sm">
            <div className="px-5 py-4 border-b border-gray-200">
              <h2 className="text-[16px] font-display font-semibold text-gray-900">Архивировать автора?</h2>
            </div>
            {activeDeals.length === 0 ? (
              <div className="px-5 py-4 text-[13px] font-medium text-gray-600 leading-relaxed">
                Автор @{authorToArchive.nick} будет перемещен в архив. Вы сможете восстановить его в любой момент.
              </div>
            ) : (
              <div className="px-5 py-4">
                <div className="rounded-xl border border-amber-200 bg-amber-50 px-3.5 py-2.5 text-[12.5px] font-medium text-amber-800 leading-relaxed">
                  У автора {activeDeals.length} {activeDeals.length % 10 === 1 && activeDeals.length % 100 !== 11 ? 'открытая сделка' : activeDeals.length % 10 >= 2 && activeDeals.length % 10 <= 4 && (activeDeals.length % 100 < 10 || activeDeals.length % 100 >= 20) ? 'открытые сделки' : 'открытых сделок'} на сумму {fmtMoney(activeBudget)}.
                </div>
                <label className="mt-3 flex items-center gap-2.5 cursor-pointer select-none">
                  <input type="checkbox" checked={closeDeals} onChange={e => setCloseDeals(e.target.checked)} className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-200" />
                  <span className="text-[13px] font-medium text-gray-700">Перевести активные сделки в архив (сорваны)</span>
                </label>
              </div>
            )}
            <div className="px-5 py-4 border-t border-gray-200 flex items-center justify-end gap-2">
              <Btn variant="secondary" size="sm" onClick={() => setAuthorToArchive(null)}>Отмена</Btn>
              <Btn size="sm" onClick={() => void handleConfirm()}><Archive size={13} />В архив</Btn>
            </div>
          </Modal>
        );
      })()}

      {(() => {
        const activeMenuAuthor = authors.find(x => x.id === menu?.id);
        if (!menu || !activeMenuAuthor) return null;
        return (
          <div className="fixed z-50 w-48 bg-white rounded-xl border border-gray-200 shadow-xl py-1.5 text-left"
            style={{ top: `${menu.top}px`, left: `${menu.left}px` }}
            onClick={e => e.stopPropagation()}>
            <MenuItem icon={<Briefcase size={14} />} label="Новая сделка" onClick={() => { setMenu(null); onNewDeal(activeMenuAuthor.id, { name: activeMenuAuthor.nick }); }} />
            <MenuItem icon={<MessageSquare size={14} />} label="Написать" onClick={() => { setMenu(null); onOpenComms(String(activeMenuAuthor.id)); }} />
            <div className="h-px bg-gray-100 my-1" />
            {activeMenuAuthor.status === 'В архиве' ? (
              <MenuItem icon={<ArchiveRestore size={14} />} label="Восстановить" onClick={() => { setMenu(null); void changeStatus(activeMenuAuthor, 'Свободен'); }} />
            ) : (
              <MenuItem icon={<Archive size={14} />} label="В архив" onClick={() => { setMenu(null); setCloseDeals(true); setAuthorToArchive(activeMenuAuthor); }} />
            )}
            <div className="h-px bg-gray-100 my-1" />
            <MenuItem icon={<Trash2 size={14} />} label="Удалить из шортлиста" danger blocked={dealsCountFor(activeMenuAuthor) > 0}
              onClick={() => {
                setMenu(null);
                if (dealsCountFor(activeMenuAuthor) > 0) {
                  toast('err', 'Нельзя удалить автора с историей сделок. Отправьте его в архив.');
                } else {
                  setConfirmDeleteAuthor(activeMenuAuthor);
                }
              }} />
          </div>
        );
      })()}
    </div>
  );
}

function MenuItem({ icon, label, onClick, danger, blocked }: { icon: React.ReactNode; label: string; onClick: () => void; danger?: boolean; blocked?: boolean }) {
  return (
    <button onClick={onClick} className={`w-full flex items-center gap-2.5 px-3.5 py-2 text-[12.5px] font-semibold text-left hover:bg-slate-50 transition-colors ${danger ? 'text-red-500' : 'text-gray-700'} ${blocked ? 'opacity-40 cursor-not-allowed' : ''}`}>
      {icon}{label}
    </button>
  );
}

const selCls = inputCls + ' !w-44 !h-9.5 text-[12.5px] font-semibold';

