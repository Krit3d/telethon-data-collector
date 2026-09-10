import { useEffect, useMemo, useState } from 'react';
import { Search, MoreVertical, Filter, Upload, Briefcase, MessageSquare, Archive, ChevronLeft, ChevronRight, X, Users, RefreshCw } from 'lucide-react';
import { fmtNum, fmtMoney, type Author, type Social } from '../data';
import { Badge, Avatar, Card, Btn, Modal, Field, inputCls, useToast, Toggle } from '../components/ui';
import { SocialIcon } from '../components/icons';
import { api, normalizeSocial, STATUS_OPTIONS, type CreatorRecord } from '../services/api';

const statusTone: Record<Author['status'], string> = { 'Свободен': 'green', 'В сделке': 'blue', 'На паузе': 'gray' };
const SOCIALS: Social[] = ['Instagram', 'VK', 'Telegram', 'TikTok', 'YouTube', 'Дзен'];
const NICHES = ['Красота', 'Технологии', 'Еда', 'Путешествия', 'Фитнес', 'Мода', 'Финансы', 'Семья', 'Авто', 'Игры'];
const PAGE = 15;

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
  return {
    id: rec.accountid ?? rec.accountId ?? rec.id,
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

export default function Authors({ onOpenProfile, onNewDeal }: { onOpenProfile: (id: string) => void; onNewDeal: (authorId: string) => void }) {
  const toast = useToast();
  const [authors, setAuthors] = useState<Author[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [q, setQ] = useState('');
  const [fSocial, setFSocial] = useState('all');
  const [fNiche, setFNiche] = useState<string[]>([]);
  const [fMin, setFMin] = useState('');
  const [fMax, setFMax] = useState('');
  const [onlyFree, setOnlyFree] = useState(false);
  const [page, setPage] = useState(0);
  const [archived, setArchived] = useState<string[]>([]);
  const [menu, setMenu] = useState<string | null>(null);
  const [picker, setPicker] = useState(false);
  const [updating, setUpdating] = useState<string | null>(null);

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

  const changeStatus = async (a: Author, status: Author['status']) => {
    setUpdating(a.id);
    try {
      await api.updateCreatorStatus(a.id, status);
      setAuthors(prev => prev.map(x => x.id === a.id ? { ...x, status } : x));
      toast('ok', `Статус ${a.nick} обновлён: ${status}`);
    } catch (err) {
      toast('err', err instanceof Error && err.message ? err.message : 'Не удалось обновить статус');
    } finally {
      setUpdating(null);
    }
  };

  const list = useMemo(() => authors.filter(a =>
    !archived.includes(a.id) &&
    (!q || a.nick.toLowerCase().includes(q.toLowerCase())) &&
    (fSocial === 'all' || a.social === fSocial) &&
    (fNiche.length === 0 || fNiche.includes(a.niche)) &&
    (!fMin || a.followers >= Number(fMin) * 1000) &&
    (!fMax || a.followers <= Number(fMax) * 1000) &&
    (!onlyFree || a.status === 'Свободен')
  ), [authors, q, fSocial, fNiche, fMin, fMax, onlyFree, archived]);

  const pages = Math.max(1, Math.ceil(list.length / PAGE));
  const pageItems = list.slice(page * PAGE, (page + 1) * PAGE);
  const freeCount = authors.filter(a => a.status === 'Свободен').length;

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
            <Btn variant="secondary" onClick={() => toast('ok', 'Импортировано 34 автора из csv-файла')}><Upload size={14} />Импорт из CSV</Btn>
            <Btn onClick={() => setPicker(true)}><Filter size={14} />Подбор авторов</Btn>
          </div>
        </div>

        <div className="flex items-center gap-2 mt-4 flex-wrap">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300" />
            <input className={inputCls + ' !w-56 !pl-9'} placeholder="Поиск по никнейму…" value={q} onChange={e => { setQ(e.target.value); setPage(0); }} />
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
          {(q || fSocial !== 'all' || fNiche.length > 0 || fMin || fMax || onlyFree) && (
            <button className="text-[12px] font-bold text-indigo-600 hover:text-indigo-700 inline-flex items-center gap-1"
              onClick={() => { setQ(''); setFSocial('all'); setFNiche([]); setFMin(''); setFMax(''); setOnlyFree(false); setPage(0); }}>
              <X size={12} />Сбросить
            </button>
          )}
        </div>
        <div className="flex items-center gap-1.5 mt-3 flex-wrap">
          {NICHES.map(n => {
            const on = fNiche.includes(n);
            return (
              <button key={n} onClick={() => { setFNiche(x => on ? x.filter(v => v !== n) : [...x, n]); setPage(0); }}
                className={`px-2.5 h-7.5 rounded-full border text-[11.5px] font-bold transition-all active:scale-95 ${on ? 'bg-gray-900 border-gray-900 text-white' : 'bg-white border-gray-200 text-gray-500 hover:border-gray-400'}`}>
                {n}
              </button>
            );
          })}
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
          <Card className="overflow-hidden">
            <div className="overflow-x-auto scroll-thin">
              <table className="w-full text-[12.5px] min-w-[960px]">
                <thead>
                  <tr className="text-left text-[11px] font-bold text-gray-400 uppercase tracking-wide border-b border-gray-100 bg-slate-50/60">
                    {['Автор', 'Соцсеть', 'Подписчики', 'Ниша', 'Ср. охват', 'ER', 'CPM', 'Статус', ''].map((h, i) => <th key={i} className="px-4 py-2.5 whitespace-nowrap">{h}</th>)}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50">
                  {pageItems.map(a => (
                    <tr key={a.id} className="group hover:bg-indigo-50/40 transition-colors">
                      <td className="px-4 py-2.5">
                        <button onClick={e => { e.stopPropagation(); onOpenProfile(a.id); }} className="flex items-center gap-2.5 hover:text-indigo-600 transition-colors">
                          <Avatar nick={a.nick} hue={a.hue} size={30} />
                          <span className="text-left">
                            <span className="block font-bold text-gray-800 group-hover:text-indigo-600">{a.nick}</span>
                            <span className="block text-[10.5px] font-semibold text-gray-400">{a.deals} сделок · ER {a.er}%</span>
                          </span>
                        </button>
                      </td>
                      <td className="px-4 py-2.5"><span className="inline-flex items-center gap-1.5 font-semibold text-gray-600"><SocialIcon social={a.social} size={14} />{a.social}</span></td>
                      <td className="px-4 py-2.5 font-bold text-gray-800 tabular-nums">{fmtNum(a.followers)}</td>
                      <td className="px-4 py-2.5"><Badge tone="violet">{a.niche}</Badge></td>
                      <td className="px-4 py-2.5 font-semibold text-gray-500 tabular-nums">{fmtNum(a.reach)}</td>
                      <td className="px-4 py-2.5 font-bold text-emerald-600 tabular-nums">{a.er}%</td>
                      <td className="px-4 py-2.5 font-bold text-gray-800 tabular-nums">{a.cpm} ₽</td>
                      <td className="px-4 py-2.5" onClick={e => e.stopPropagation()}>
                        <select
                          value={a.status}
                          disabled={updating === a.id}
                          onChange={e => void changeStatus(a, e.target.value as Author['status'])}
                          className="h-7.5 pl-2 pr-1 rounded-lg border border-gray-200 bg-white text-[11.5px] font-semibold text-gray-700 outline-none focus:border-indigo-400 disabled:opacity-50">
                          {STATUS_OPTIONS.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                      </td>
                      <td className="px-4 py-2.5 text-right relative" onClick={e => e.stopPropagation()}>
                        <button className="w-7 h-7 rounded-lg text-gray-300 group-hover:text-gray-500 hover:bg-gray-100 inline-flex items-center justify-center transition-colors"
                          onClick={() => setMenu(m => m === a.id ? null : a.id)}><MoreVertical size={15} /></button>
                        {menu === a.id && (
                          <div className="absolute right-4 top-9 z-30 w-48 bg-white rounded-xl border border-gray-200 shadow-xl py-1.5 pop-in text-left">
                            <MenuItem icon={<Briefcase size={14} />} label="Новая сделка" onClick={() => { setMenu(null); onNewDeal(a.id); }} />
                            <MenuItem icon={<MessageSquare size={14} />} label="Написать" onClick={() => { setMenu(null); toast('ok', `Чат с ${a.nick} открыт в Коммуникациях`); }} />
                            <MenuItem icon={<Archive size={14} />} label="Архивировать" danger onClick={() => { setMenu(null); setArchived(x => [...x, a.id]); toast('info', `${a.nick} перемещён в архив`); }} />
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
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

      {picker && <PickerModal authors={authors} onClose={() => setPicker(false)} onDeal={id => { setPicker(false); onNewDeal(id); }} />}
    </div>
  );
}

function MenuItem({ icon, label, onClick, danger }: { icon: React.ReactNode; label: string; onClick: () => void; danger?: boolean }) {
  return (
    <button onClick={onClick} className={`w-full flex items-center gap-2.5 px-3.5 py-2 text-[12.5px] font-semibold text-left hover:bg-slate-50 transition-colors ${danger ? 'text-red-500' : 'text-gray-700'}`}>
      {icon}{label}
    </button>
  );
}

function PickerModal({ authors, onClose, onDeal }: { authors: Author[]; onClose: () => void; onDeal: (id: string) => void }) {
  const toast = useToast();
  const [niche, setNiche] = useState<string[]>(['Красота', 'Технологии']);
  const [minEr, setMinEr] = useState(3);
  const [maxCpm, setMaxCpm] = useState(250);
  const [onlyFree, setOnlyFree] = useState(true);

  const results = useMemo(() => authors.filter(a =>
    (niche.length === 0 || niche.includes(a.niche)) && a.er >= minEr && a.cpm <= maxCpm && (!onlyFree || a.status === 'Свободен')
  ).sort((x, y) => y.roas - x.roas).slice(0, 8), [authors, niche, minEr, maxCpm, onlyFree]);

  return (
    <Modal onClose={onClose} w="max-w-[760px]">
      <div className="px-5 py-4 border-b border-gray-100 flex items-center">
        <div>
          <h2 className="text-[15.5px] font-bold text-gray-900">Подбор авторов</h2>
          <p className="text-[12px] font-medium text-gray-400 mt-0.5">Расширенные фильтры по базе из {authors.length} авторов</p>
        </div>
        <button onClick={onClose} className="ml-auto text-gray-300 hover:text-gray-600"><X size={18} /></button>
      </div>
      <div className="p-5 grid grid-cols-[240px_1fr] gap-5">
        <div className="flex flex-col gap-4">
          <Field label="Ниша">
            <div className="flex flex-wrap gap-1.5">
              {NICHES.slice(0, 8).map(n => {
                const on = niche.includes(n);
                return (
                  <button key={n} onClick={() => setNiche(x => on ? x.filter(v => v !== n) : [...x, n])}
                    className={`px-2 h-7 rounded-full border text-[11px] font-bold transition-all ${on ? 'bg-indigo-500 border-indigo-500 text-white' : 'bg-white border-gray-200 text-gray-500'}`}>{n}</button>
                );
              })}
            </div>
          </Field>
          <Field label={`ER не ниже: ${minEr}%`}>
            <input type="range" min={0} max={8} step={0.5} value={minEr} onChange={e => setMinEr(Number(e.target.value))} className="w-full accent-indigo-500" />
          </Field>
          <Field label={`CPM не выше: ${maxCpm} ₽`}>
            <input type="range" min={90} max={350} step={10} value={maxCpm} onChange={e => setMaxCpm(Number(e.target.value))} className="w-full accent-indigo-500" />
          </Field>
          <label className="flex items-center gap-2 text-[12px] font-bold text-gray-500 cursor-pointer"><Toggle on={onlyFree} onChange={setOnlyFree} />Только свободные</label>
        </div>
        <div className="min-h-[300px]">
          <div className="text-[11.5px] font-bold text-gray-400 uppercase tracking-wide mb-2">Результаты · {results.length}</div>
          <div className="flex flex-col gap-2 max-h-[380px] overflow-y-auto scroll-thin pr-1">
            {results.map(a => (
              <div key={a.id} className="flex items-center gap-3 rounded-xl border border-gray-200 px-3.5 py-2.5 hover:border-indigo-300 hover:shadow-sm transition-all">
                <Avatar nick={a.nick} hue={a.hue} size={34} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5"><span className="text-[13px] font-bold text-gray-900">{a.nick}</span><SocialIcon social={a.social} size={12} /></div>
                  <div className="text-[11px] font-semibold text-gray-400">{fmtNum(a.followers)} · {a.niche} · CPM {a.cpm} ₽ · ER {a.er}%</div>
                </div>
                <Badge tone="green">ER {a.er}%</Badge>
                <Btn size="xs" variant="outline" onClick={() => { toast('ok', `Сделка с ${a.nick} создаётся…`); onDeal(a.id); }}>Сделка</Btn>
              </div>
            ))}
            {results.length === 0 && <div className="py-14 text-center text-[12.5px] font-semibold text-gray-400">Нет авторов под эти критерии</div>}
          </div>
        </div>
      </div>
    </Modal>
  );
}
const selCls = inputCls + ' !w-44 !h-9.5 text-[12.5px] font-semibold';
