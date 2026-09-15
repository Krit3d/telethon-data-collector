import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, Briefcase, MessageSquare, Archive, ArchiveRestore, ArrowDownRight, ArrowUpRight, StickyNote, FileText, TrendingUp, Target, X, Sparkles, Upload, Download, Trash2 } from 'lucide-react';
import { fmtNum, fmtMoney, ARCHIVED_STAGE, getStage, STAGES, type Deal } from '../data';
import { Badge, Avatar, Card, Btn, useToast, Modal, inputCls } from '../components/ui';
import { SocialIcon, PostThumb } from '../components/icons';
import { LineChart, HBars } from '../components/charts';
import { api, normalizeSocial, type CreatorPostItem, type CreatorProfileDetail } from '../services/api';

const loadNotes = (id: string): string[] => {
  try {
    const raw = localStorage.getItem(`crm_notes_${id}`);
    if (raw) {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) return parsed.filter((x): x is string => typeof x === 'string');
    }
  } catch {
    return [];
  }
  return [];
};

interface AuthorDocItem {
  id: string;
  name: string;
  size: string;
  date: string;
  ext: 'pdf' | 'doc' | 'img' | 'other';
  source: 'deal' | 'upload';
  dealTitle?: string;
  dataUrl?: string;
}

const loadAuthorDocs = (id: string): AuthorDocItem[] => {
  try {
    const raw = localStorage.getItem(`crm_author_docs_${id}`);
    if (raw) {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) return parsed as AuthorDocItem[];
    }
  } catch {
    return [];
  }
  return [];
};

const extFromName = (name: string): AuthorDocItem['ext'] => {
  const ext = name.split('.').pop()?.toLowerCase() ?? '';
  if (ext === 'pdf') return 'pdf';
  if (ext === 'doc' || ext === 'docx') return 'doc';
  if (ext === 'png' || ext === 'jpg' || ext === 'jpeg' || ext === 'gif' || ext === 'webp' || ext === 'svg') return 'img';
  return 'other';
};

const fmtFileSize = (bytes: number): string => {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
};

const docBadgeCls = (ext: AuthorDocItem['ext']): string => {
  if (ext === 'pdf') return 'bg-red-50 text-red-600';
  if (ext === 'doc') return 'bg-blue-50 text-blue-600';
  if (ext === 'img') return 'bg-emerald-50 text-emerald-600';
  return 'bg-gray-100 text-gray-500';
};

const fmtBudget = (v: number): string => v >= 1000000 ? `${(v / 1000000).toFixed(1).replace('.0', '')}M` : v >= 1000 ? `${Math.round(v / 1000)}K` : String(v);

const getOptimalBudget = (cpm: number, avgReach: number): string => {
  if (cpm <= 0) return 'Бюджет рассчитывается индивидуально';
  const reach = avgReach > 0 ? avgReach : 100000;
  const min = Math.round(cpm * reach / 1000 * 0.8);
  const max = Math.round(cpm * reach / 1000 * 1.2);
  return `Диапазон: ${fmtBudget(min)}–${fmtBudget(max)} ₽ (на основе CPM и среднего охвата)`;
};

const getBestFormat = (posts: CreatorPostItem[], platform: string): string => {
  if (posts.length === 0) {
    const s = normalizeSocial(platform.toLowerCase());
    return s === 'YouTube' ? 'Видео' : s === 'TikTok' ? 'Короткие видео' : 'Reels';
  }
  const sums = new Map<string, { sum: number; count: number }>();
  for (const p of posts) {
    const cur = sums.get(p.post_type) ?? { sum: 0, count: 0 };
    cur.sum += p.er;
    cur.count += 1;
    sums.set(p.post_type, cur);
  }
  let best = '';
  let bestAvg = -1;
  for (const [type, { sum, count }] of sums) {
    const avg = sum / count;
    if (avg > bestAvg) {
      bestAvg = avg;
      best = type;
    }
  }
  return best;
};

const DAYS = ['Вс', 'Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб'];

const getBestPublishTime = (posts: CreatorPostItem[]): string => {
  if (posts.length === 0) return 'Вторник / Четверг 18:00–20:00 (МСК)';
  let best = posts[0];
  for (const p of posts) if (p.er > best.er) best = p;
  const msk = new Date(new Date(best.published_at).getTime() + 3 * 3600 * 1000);
  return `${DAYS[msk.getUTCDay()]} ${String(msk.getUTCHours()).padStart(2, '0')}:00 (МСК)`;
};

const ratingFromEr = (er: number): string => {
  if (er >= 5) return 'A';
  if (er >= 3.5) return 'B';
  if (er >= 2) return 'C';
  return 'D';
};

const fmtDate = (iso: string): string => {
  const d = new Date(iso);
  return `${String(d.getDate()).padStart(2, '0')}.${String(d.getMonth() + 1).padStart(2, '0')}`;
};

const postTone = (type: string): string => {
  if (type === 'Stories') return 'violet';
  if (type === 'Reels') return 'sky';
  if (type === 'Видео') return 'red';
  return 'indigo';
};

const postHue = (id: number): number => Math.abs(id * 137) % 360;

const fmtEr = (v: number | null | undefined): string => typeof v === 'number' && !Number.isNaN(v) ? `${v.toFixed(1)}%` : '0.0%';

const cleanText = (text: string | null | undefined): string | null => {
  if (text === null || text === undefined || text.trim() === '') return null;
  let hasContent = false;
  for (const ch of text) {
    if (ch >= '0' && ch <= '9') {
      hasContent = true;
      break;
    }
    if (ch.toLowerCase() !== ch.toUpperCase()) {
      hasContent = true;
      break;
    }
  }
  return hasContent ? text.trim() : null;
};

export default function AuthorProfile({ authorId, deals, onBack, onNewDeal, onOpenDeal, onOpenComms }: {
  authorId: string; deals: Deal[]; onBack: () => void; onNewDeal: (authorId: string, meta?: { name?: string; handle?: string }) => void; onOpenDeal: (id: string, tab?: string) => void; onOpenComms: () => void;
}) {
  const toast = useToast();
  const [tab, setTab] = useState('analytics');
  const [profile, setProfile] = useState<CreatorProfileDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [managerNotes, setManagerNotes] = useState<string[]>(() => loadNotes(authorId));
  const [noteOpen, setNoteOpen] = useState(false);
  const [noteText, setNoteText] = useState('');
  const [customDocs, setCustomDocs] = useState<AuthorDocItem[]>(() => loadAuthorDocs(authorId));
  const [confirmArchiveOpen, setConfirmArchiveOpen] = useState(false);
  const [closeDeals, setCloseDeals] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const myDeals = deals.filter(d => String(d.authorId) === String(authorId) || (profile && String(d.authorId) === String(profile.id)));

  const dealDocs: AuthorDocItem[] = myDeals.flatMap(d => {
    const brandName = (d as Deal & { brandName?: string }).brandName || 'Интеграция';
    const base: AuthorDocItem = {
      id: `deal-contract-${d.id}`,
      name: `Договор_№${d.id}_${brandName.replace(/\s+/g, '_')}.pdf`,
      size: '180 KB',
      date: d.date || fmtDate(new Date().toISOString()),
      ext: 'pdf',
      source: 'deal',
      dealTitle: d.title,
    };
    const items: AuthorDocItem[] = [base];
    if (d.file) {
      items.push({
        id: `deal-file-${d.id}`,
        name: d.file,
        size: '180 KB',
        date: d.date || fmtDate(new Date().toISOString()),
        ext: extFromName(d.file),
        source: 'deal',
        dealTitle: d.title,
      });
    }
    return items;
  });

  const allDocs: AuthorDocItem[] = [...dealDocs, ...customDocs];

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setProfile(null);
    setManagerNotes(loadNotes(authorId));
    api.getCreatorProfile(authorId)
      .then(p => {
        if (cancelled) return;
        setProfile(p);
      })
      .catch(e => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : 'Не удалось загрузить профиль');
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [authorId]);

  const addNote = () => {
    const t = noteText.trim();
    if (!t) return;
    const next = [...managerNotes, t];
    setManagerNotes(next);
    localStorage.setItem(`crm_notes_${authorId}`, JSON.stringify(next));
    setNoteText('');
    setNoteOpen(false);
    toast('ok', 'Заметка добавлена');
  };

  const deleteNote = (index: number) => {
    const next = managerNotes.filter((_, i) => i !== index);
    setManagerNotes(next);
    localStorage.setItem(`crm_notes_${authorId}`, JSON.stringify(next));
    toast('info', 'Заметка удалена');
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = typeof reader.result === 'string' ? reader.result : undefined;
      const item: AuthorDocItem = {
        id: `upload-${Date.now()}`,
        name: file.name,
        size: fmtFileSize(file.size),
        date: fmtDate(new Date().toISOString()),
        ext: extFromName(file.name),
        source: 'upload',
        dataUrl,
      };
      const next = [...customDocs, item];
      setCustomDocs(next);
      localStorage.setItem(`crm_author_docs_${authorId}`, JSON.stringify(next));
      toast('ok', `Файл ${file.name} загружен`);
    };
    reader.readAsDataURL(file);
  };

  const downloadDoc = (doc: AuthorDocItem) => {
    if (doc.dataUrl) {
      const a = document.createElement('a');
      a.href = doc.dataUrl;
      a.download = doc.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } else if (doc.source === 'deal') {
      const text = `Договор №${doc.id}\n\nСделка: ${doc.dealTitle ?? ''}\n\nНастоящий договор заключен между сторонами на условиях, согласованных в рамках сделки.`;
      const blob = new Blob([text], { type: 'application/pdf' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = doc.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
    toast('ok', `Файл ${doc.name} скачан`);
  };

  const deleteDoc = (doc: AuthorDocItem) => {
    const next = customDocs.filter(x => x.id !== doc.id);
    setCustomDocs(next);
    localStorage.setItem(`crm_author_docs_${authorId}`, JSON.stringify(next));
    toast('info', 'Документ удален');
  };

  const handleRestore = async () => {
    if (!profile) return;
    try {
      await api.updateCreatorStatus(String(profile.id), 'Свободен');
      setProfile({ ...profile, status: 'Свободен' });
      toast('ok', 'Автор восстановлен');
    } catch {
      toast('err', 'Не удалось восстановить автора');
    }
  };

  const handleArchiveConfirm = async () => {
    if (!profile) return;
    const activeDeals = myDeals.filter(d => d.stage >= 1 && d.stage <= 6);
    try {
      await api.updateCreatorStatus(String(profile.id), 'В архиве', activeDeals.length > 0 ? closeDeals : false);
      setProfile({ ...profile, status: 'В архиве' });
      setConfirmArchiveOpen(false);
      toast('ok', 'Автор архивирован');
    } catch {
      toast('err', 'Не удалось архивировать автора');
    }
  };

  if (loading) {
    return (
      <div className="h-full overflow-y-auto scroll-thin">
        <div className="px-6 pt-4">
          <div className="h-4 w-32 bg-gray-200 rounded animate-pulse mb-3" />
          <div className="bg-white rounded-xl border border-gray-200/80 p-5 flex items-center gap-5">
            <div className="w-20 h-20 rounded-full bg-gray-200 animate-pulse" />
            <div className="flex-1 space-y-2.5">
              <div className="h-6 w-48 bg-gray-200 rounded animate-pulse" />
              <div className="h-4 w-72 bg-gray-200 rounded animate-pulse" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="h-full overflow-y-auto scroll-thin">
        <div className="px-6 pt-4">
          <button onClick={onBack} className="inline-flex items-center gap-1.5 text-[12.5px] font-bold text-gray-400 hover:text-indigo-600 transition-colors mb-3">
            <ArrowLeft size={14} />К списку авторов
          </button>
          <Card className="p-10 text-center">
            <div className="text-[15px] font-bold text-gray-900 mb-1">Не удалось загрузить профиль</div>
            <div className="text-[13px] font-semibold text-gray-400 mb-4">{error ?? 'Профиль не найден'}</div>
            <Btn onClick={onBack}><ArrowLeft size={14} />Назад</Btn>
          </Card>
        </div>
      </div>
    );
  }

  const nick = profile.username || profile.title;
  const hue = Math.abs(Number(profile.id) * 137) % 360;
  const social = normalizeSocial(profile.platform.toLowerCase());
  const erPosts = profile.posts.slice(0, 12).reverse();
  const roas = profile.cpm > 0 ? Math.max(1, Math.round((200 / profile.cpm) * 2.5 * 10) / 10) : 1;
  const onTime = myDeals.length ? Math.round((myDeals.filter(d => d.stage >= 4).length / myDeals.length) * 100) : 0;
  const rating = ratingFromEr(profile.static_avg_er);
  const hasActiveDeal = myDeals.some(d => d.stage >= 1 && d.stage <= 6);
  const currentStatus = profile.status === 'В архиве' ? 'В архиве' : profile.status === 'На паузе' ? 'На паузе' : hasActiveDeal ? 'В сделке' : 'Свободен';

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="px-6 pt-4">
        <button onClick={onBack} className="inline-flex items-center gap-1.5 text-[12.5px] font-bold text-gray-400 hover:text-indigo-600 transition-colors mb-3">
          <ArrowLeft size={14} />К списку авторов
        </button>

        <Card className="p-5 flex items-center gap-5 flex-wrap">
          <Avatar nick={nick} hue={hue} size={80} />
          <div className="min-w-0">
            <div className="flex items-center gap-2.5">
              <h1 className="text-[24px] font-display font-semibold text-gray-900">{nick}</h1>
              <a href={profile.profile_url} target="_blank" rel="noreferrer"
                className="inline-flex items-center gap-1.5 text-[12.5px] font-bold text-gray-500 hover:text-indigo-600 transition-colors">
                <SocialIcon social={social} size={15} />{social}
              </a>
            </div>
            <div className="flex items-center gap-4 mt-2 text-[13px] font-semibold text-gray-500 flex-wrap">
              <span>Статус: <Badge tone={currentStatus === 'В архиве' ? 'gray' : currentStatus === 'В сделке' ? 'indigo' : 'green'}>{currentStatus}</Badge></span>
              <span>Подписчики: <b className="text-gray-900">{fmtNum(profile.subscribers_count)}</b></span>
              <span>Ниша: <Badge tone="violet">{profile.category_path || 'Общее'}</Badge></span>
              <span>Средний ER: <b className="text-gray-900">{fmtEr(profile.static_avg_er)}</b></span>
              <span>Сделок: <b className="text-gray-900">{myDeals.length}</b></span>
            </div>
          </div>
          <div className="flex items-center gap-2 ml-auto">
            <Btn onClick={() => onNewDeal(String(profile.id), { name: profile.title, handle: profile.username ? (profile.username.startsWith('@') ? profile.username : `@${profile.username}`) : undefined })}><Briefcase size={14} />Новая сделка</Btn>
            <Btn variant="secondary" onClick={() => myDeals.length > 0 ? onOpenDeal(myDeals[0].id, 'comms') : onOpenComms()}><MessageSquare size={14} />Написать</Btn>
            <Btn variant="danger" onClick={() => profile.status === 'В архиве' ? void handleRestore() : (setCloseDeals(true), setConfirmArchiveOpen(true))}>
              {profile.status === 'В архиве' ? <ArchiveRestore size={14} /> : <Archive size={14} />}
              {profile.status === 'В архиве' ? 'Восстановить' : 'В архив'}
            </Btn>
          </div>
        </Card>

        <div className="flex items-center gap-1 mt-4 border-b border-gray-200">
          {[['deals', `История сделок · ${myDeals.length}`], ['pubs', `Публикации · ${profile.posts.length}`], ['analytics', 'Аналитика'], ['docs', 'Документы']].map(([id, l]) => (
            <button key={id} onClick={() => setTab(id)}
              className={`px-3 py-2.5 text-[13px] font-semibold border-b-2 -mb-px transition-colors ${tab === id ? 'text-indigo-600 border-indigo-500' : 'text-gray-500 border-transparent hover:text-gray-800'}`}>{l}</button>
          ))}
        </div>
      </div>

      <div className="p-6 flex gap-5 items-start">
        <div className="flex-1 min-w-0 anim-in" key={tab}>
          {tab === 'analytics' && (
            <div className="flex flex-col gap-4">
              <Card className="p-4">
                <h3 className="text-[13.5px] font-bold text-gray-900 mb-1">Динамика ER по публикациям</h3>
                <div className="text-[11px] font-semibold text-gray-400 mb-2">engagement rate, % · последние 12 публикаций</div>
                {erPosts.length >= 2 ? (
                  <LineChart unit="%" h={200} labels={erPosts.map(p => fmtDate(p.published_at))}
                    series={[{ name: 'ER', color: '#6366F1', data: erPosts.map(p => p.er), fill: true }]} />
                ) : (
                  <div className="py-10 text-center text-[13px] font-semibold text-gray-400">Недостаточно данных для графика</div>
                )}
              </Card>
              <Card className="p-4">
                <h3 className="text-[13.5px] font-bold text-gray-900 mb-1">CPM по сделкам</h3>
                <div className="text-[11px] font-semibold text-gray-400 mb-2">CPM автора: {profile.cpm} ₽ за 1000 просмотров</div>
                <HBars money items={[
                  ...myDeals.map(d => ({ label: d.title.slice(0, 34), v: d.budget, color: '#6366F1' })),
                  { label: 'CPM автора (референс)', v: profile.cpm * 300, color: '#CBD5E1' },
                ]} />
              </Card>
              <Card className="overflow-hidden">
                <div className="px-4 py-3 border-b border-gray-100 text-[11px] font-bold text-gray-400 uppercase tracking-wide">Все публикации автора</div>
                <table className="w-full text-[12.5px]">
                  <thead><tr className="text-left text-[11px] font-bold text-gray-400 bg-slate-50/60 border-b border-gray-100">
                    {['Публикация', 'Тип', 'Дата', 'Просмотры', 'ER'].map(h => <th key={h} className="px-4 py-2">{h}</th>)}
                  </tr></thead>
                  <tbody className="divide-y divide-gray-50">
                    {profile.posts.map(p => (
                      <tr key={p.id} className="hover:bg-slate-50 transition-colors">
                        <td className="px-4 py-2.5"><span className="flex items-center gap-2.5"><PostThumb hue={postHue(p.id)} size={32} />{(() => { const clean = cleanText(p.text); return clean ? <span className="font-semibold text-gray-700 max-w-[260px] truncate">{clean}</span> : <span className="italic font-normal text-[12px] text-gray-400 max-w-[260px] truncate select-none">Без описания</span>; })()}</span></td>
                        <td className="px-4 py-2.5"><Badge tone={postTone(p.post_type)}>{p.post_type}</Badge></td>
                        <td className="px-4 py-2.5 font-semibold text-gray-500">{fmtDate(p.published_at)}</td>
                        <td className="px-4 py-2.5 font-bold text-gray-800 tabular-nums">{fmtNum(p.views)}</td>
                        <td className="px-4 py-2.5 font-bold text-emerald-600 tabular-nums">{p.er.toFixed(1)}%</td>
                      </tr>
                    ))}
                    {profile.posts.length === 0 && <tr><td colSpan={5} className="px-4 py-10 text-center font-semibold text-gray-400">Публикаций пока нет</td></tr>}
                  </tbody>
                </table>
              </Card>
            </div>
          )}

          {tab === 'deals' && (
            <div className="flex flex-col gap-2.5">
              {myDeals.map(d => (
                <Card key={d.id} className="p-4 flex items-center gap-4 hover:border-indigo-200 transition-colors cursor-pointer" onClick={() => onOpenDeal(d.id, 'overview')}>
                  <span className="w-1 h-10 rounded-full shrink-0" style={{ backgroundColor: getStage(d.stage).color }} />
                  <div className="min-w-0 flex-1">
                    <div className="text-[13.5px] font-bold text-gray-900 truncate">{d.title}</div>
                    <div className="text-[11.5px] font-semibold text-gray-400 mt-0.5">{d.type} · публикация {d.pubDate} · создан {d.date}</div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-[14px] font-extrabold text-gray-900 tabular-nums">{fmtMoney(d.budget)}</div>
                    <div className="text-[11px] font-semibold text-gray-400">{d.stage === 0 ? ARCHIVED_STAGE.name : `Этап ${d.stage} из ${STAGES.length}`}</div>
                  </div>
                </Card>
              ))}
              {myDeals.length === 0 && <Card className="p-10 text-center text-[13px] font-semibold text-gray-400">Сделок с этим автором пока нет</Card>}
            </div>
          )}

          {tab === 'pubs' && (
            <div className="grid grid-cols-3 gap-3">
              {profile.posts.map(p => (
                <Card key={p.id} className="overflow-hidden group">
                  <div className="h-32 relative" style={{ background: `linear-gradient(150deg, hsl(${postHue(p.id)} 72% 80%), hsl(${postHue(p.id) + 45} 62% 56%))` }}>
                    <div className="absolute inset-0" style={{ background: 'radial-gradient(circle at 30% 20%, rgba(255,255,255,.5), transparent 55%)' }} />
                    <Badge tone="dark" className="absolute top-2 left-2">{p.post_type}</Badge>
                  </div>
                  <div className="p-3">
                    {(() => { const clean = cleanText(p.text); return clean ? <div className="text-[12.5px] font-bold text-gray-800 leading-snug line-clamp-2">{clean}</div> : <div className="text-[12px] italic font-normal text-gray-400 leading-snug line-clamp-2 select-none">Без описания</div>; })()}
                    <div className="flex items-center gap-3 mt-2 text-[11px] font-bold text-gray-400">
                      <span>{fmtNum(p.views)} просм.</span><span className="text-emerald-600">ER {p.er.toFixed(1)}%</span>
                      <span className="ml-auto">{fmtDate(p.published_at)}</span>
                    </div>
                  </div>
                </Card>
              ))}
              {profile.posts.length === 0 && <Card className="p-10 text-center text-[13px] font-semibold text-gray-400 col-span-3">Публикаций пока нет</Card>}
            </div>
          )}

          {tab === 'docs' && (
            <div className="flex flex-col gap-4">
              <input type="file" ref={fileInputRef} className="hidden" onChange={handleFileUpload} />
              <div className="flex items-center gap-2">
                <Btn variant="secondary" onClick={() => fileInputRef.current?.click()}><Upload size={14} />Загрузить документ</Btn>
              </div>
              {allDocs.length === 0 ? (
                <Card className="p-10 text-center">
                  <div className="w-14 h-14 rounded-full bg-gray-100 text-gray-400 flex items-center justify-center mx-auto mb-3"><FileText size={24} /></div>
                  <div className="text-[15px] font-bold text-gray-900 mb-1">Документов пока нет</div>
                  <div className="text-[13px] font-semibold text-gray-400 max-w-sm mx-auto">Здесь будут отображаться договоры по сделкам и прикрепленные файлы автора. Вы можете загрузить медиа-кит или реквизиты.</div>
                </Card>
              ) : (
                <Card className="divide-y divide-gray-100 overflow-hidden">
                  {allDocs.map(doc => (
                    <div key={doc.id} className="flex items-center gap-3 px-4 py-3 hover:bg-slate-50 transition-colors">
                      <span className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${docBadgeCls(doc.ext)}`}><FileText size={16} /></span>
                      <div className="min-w-0 flex-1">
                        <div className="text-[13px] font-bold text-gray-800 truncate">{doc.name}</div>
                        <div className="text-[11px] font-semibold text-gray-400 flex items-center gap-1.5 flex-wrap">
                          <span>{doc.size} · {doc.date}</span>
                          {doc.source === 'deal' ? (
                            <Badge tone="indigo">Сделка: {doc.dealTitle}</Badge>
                          ) : (
                            <Badge tone="gray">Загружен вручную</Badge>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <Btn variant="ghost" size="xs" onClick={() => downloadDoc(doc)}><Download size={13} />Скачать</Btn>
                        {doc.source === 'upload' && (
                          <button onClick={() => deleteDoc(doc)} className="p-1.5 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors cursor-pointer"><Trash2 size={14} /></button>
                        )}
                      </div>
                    </div>
                  ))}
                </Card>
              )}
            </div>
          )}
        </div>

        <aside className="w-[300px] shrink-0 flex flex-col gap-4 sticky top-2">
          <Card className="p-5 text-center">
            <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide">Рейтинг автора</div>
            <div className="text-[48px] font-extrabold font-display text-emerald-500 leading-none mt-2">{rating}</div>
            <div className="text-[11.5px] font-semibold text-gray-400 mt-1.5">На основе {myDeals.length} сделок</div>
            <div className="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-gray-100">
              <Metric icon={<TrendingUp size={13} />} v={`${roas}x`} k="ROAS" />
              <Metric icon={<Target size={13} />} v={fmtEr(profile.static_avg_er)} k="Вовлеч." />
              <Metric icon={<TrendingUp size={13} />} v={`${onTime}%`} k="В срок" />
            </div>
          </Card>
          <Card className="p-4">
            <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-3">Сравнение с рынком</div>
            <div className="flex flex-col gap-3">
              <div>
                <div className="flex items-center justify-between text-[12px] mb-1">
                  <span className="font-semibold text-gray-500">CPM автора</span>
                  <span className="font-bold text-gray-900 tabular-nums">{profile.cpm} ₽ <span className="text-gray-300">/ рынок 220 ₽</span></span>
                </div>
                <div className="h-2 rounded-full bg-gray-100 overflow-hidden relative">
                  <div className="absolute inset-y-0 left-0 rounded-full bg-indigo-500 bar-grow" style={{ width: `${Math.min(100, (profile.cpm / 350) * 100)}%` }} />
                  <div className="absolute inset-y-0 w-[2px] bg-gray-400" style={{ left: `${(220 / 350) * 100}%` }} />
                </div>
                <div className="flex items-center gap-1 text-[11px] font-bold mt-1">{profile.cpm === 0 ? <span className="text-gray-400">Нет данных по CPM</span> : profile.cpm <= 220 ? <><ArrowDownRight size={11} className="text-emerald-600" /><span className="text-emerald-600">Выгодно — на {Math.round((1 - profile.cpm / 220) * 100)}% ниже рынка</span></> : <><ArrowUpRight size={11} className="text-orange-500" /><span className="text-orange-500">На {Math.round((profile.cpm / 220 - 1) * 100)}% выше рынка</span></>}</div>
              </div>
              <div>
                <div className="flex items-center justify-between text-[12px] mb-1">
                  <span className="font-semibold text-gray-500">ER автора</span>
                  <span className="font-bold text-gray-900 tabular-nums">{fmtEr(profile.static_avg_er)} <span className="text-gray-300">/ рынок 3.5%</span></span>
                </div>
                <div className="h-2 rounded-full bg-gray-100 overflow-hidden relative">
                  <div className="absolute inset-y-0 left-0 rounded-full bg-emerald-500 bar-grow" style={{ width: `${Math.min(100, (profile.static_avg_er / 8) * 100)}%` }} />
                  <div className="absolute inset-y-0 w-[2px] bg-gray-400" style={{ left: `${(3.5 / 8) * 100}%` }} />
                </div>
                <div className="flex items-center gap-1 text-[11px] font-bold text-emerald-600 mt-1"><ArrowUpRight size={11} />{profile.static_avg_er >= 3.5 ? 'Выше среднего' : 'Ниже среднего'}</div>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-2.5 flex items-center gap-1.5"><Sparkles size={13} className="text-indigo-500" />AI-рекомендации</div>
            <div className="flex flex-col gap-2">
              <div className="rounded-lg bg-indigo-50/60 border border-indigo-100 px-3 py-2 text-[12px] font-semibold text-gray-700 leading-snug">Оптимальный бюджет: {getOptimalBudget(profile.cpm, profile.avg_reach)}</div>
              <div className="rounded-lg bg-indigo-50/60 border border-indigo-100 px-3 py-2 text-[12px] font-semibold text-gray-700 leading-snug">Лучший формат: {getBestFormat(profile.posts, profile.platform)}</div>
              <div className="rounded-lg bg-indigo-50/60 border border-indigo-100 px-3 py-2 text-[12px] font-semibold text-gray-700 leading-snug">Лучшее время: {getBestPublishTime(profile.posts)}</div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-2.5 flex items-center gap-1.5"><StickyNote size={13} className="text-amber-500" />Заметки менеджера</div>
            <div className="flex flex-col gap-2">
              {managerNotes.map((n, i) => (
                <div key={i} className="group flex items-start justify-between gap-2 rounded-lg bg-amber-50/60 border border-amber-100 px-3 py-2">
                  <span className="flex-1 text-[12px] font-semibold text-gray-700 leading-snug break-words">{n}</span>
                  <button onClick={() => deleteNote(i)} className="opacity-0 group-hover:opacity-100 hover:text-red-600 transition-all text-gray-400 shrink-0 cursor-pointer"><X size={14} /></button>
                </div>
              ))}
              {managerNotes.length === 0 && <div className="text-[12px] font-semibold text-gray-400">Заметок пока нет. Добавьте заметку для фиксации договоренностей</div>}
            </div>
            <Btn variant="ghost" size="xs" className="mt-2" onClick={() => setNoteOpen(true)}>+ Добавить заметку</Btn>
          </Card>
        </aside>
      </div>

      {noteOpen && (
        <Modal onClose={() => setNoteOpen(false)} w="max-w-md">
          <div className="px-5 py-4 border-b border-gray-100 flex items-center">
            <h2 className="text-[15.5px] font-bold text-gray-900">Новая заметка</h2>
            <button onClick={() => setNoteOpen(false)} className="ml-auto text-gray-300 hover:text-gray-600"><X size={18} /></button>
          </div>
          <div className="p-5">
            <textarea className={inputCls + ' h-24 resize-none'} placeholder="Текст заметки…" value={noteText} onChange={e => setNoteText(e.target.value)} />
          </div>
          <div className="px-5 py-4 border-t border-gray-100 flex justify-end gap-2 bg-slate-50/50">
            <Btn variant="ghost" onClick={() => setNoteOpen(false)}>Отмена</Btn>
            <Btn onClick={addNote} disabled={!noteText.trim()}>Сохранить</Btn>
          </div>
        </Modal>
      )}

      {confirmArchiveOpen && profile && (() => {
        const activeDeals = myDeals.filter(d => d.stage >= 1 && d.stage <= 6);
        const activeBudget = activeDeals.reduce((acc, d) => acc + d.budget, 0);
        return (
          <Modal onClose={() => setConfirmArchiveOpen(false)} w="max-w-sm">
            <div className="px-5 py-4 border-b border-gray-200">
              <h2 className="text-[16px] font-display font-semibold text-gray-900">Архивировать автора?</h2>
            </div>
            {activeDeals.length === 0 ? (
              <div className="px-5 py-4 text-[13px] font-medium text-gray-600 leading-relaxed">
                Автор @{nick} будет перемещен в архив. Вы сможете восстановить его в любой момент.
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
              <Btn variant="secondary" size="sm" onClick={() => setConfirmArchiveOpen(false)}>Отмена</Btn>
              <Btn size="sm" onClick={() => void handleArchiveConfirm()}><Archive size={13} />В архив</Btn>
            </div>
          </Modal>
        );
      })()}
    </div>
  );
}

function Metric({ icon, v, k }: { icon: React.ReactNode; v: string; k: string }) {
  return (
    <div>
      <div className="flex items-center justify-center gap-1 text-indigo-500">{icon}</div>
      <div className="text-[14px] font-extrabold text-gray-900 font-display mt-1">{v}</div>
      <div className="text-[10px] font-bold text-gray-400 uppercase">{k}</div>
    </div>
  );
}
