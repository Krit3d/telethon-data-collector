import { Archive, ArchiveRestore, Briefcase, Download, Eye, FileText, ImageOff, Loader2, MessageSquare, Paperclip, Plus, Search, Send, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { SocialIcon } from '../components/icons';
import { Avatar, Badge, Btn, Card, inputCls, Modal, Tip, useToast } from '../components/ui';
import { ARCHIVED_STAGE, brandById, BRANDS, fmtMoney, getStage, resolveAuthor, STAGES, type Author, type Deal, type DealMsg, type Social } from '../data';
import { api, normalizeSocial, resolveMediaUrl, type CommunicationChannelItem, type CreatorRecord, type DealMessageItem, type FileUploadResponse } from '../services/api';

const STAGE_TONES: Record<number, string> = {
  6: 'green',
  5: 'sky',
  4: 'indigo',
  3: 'amber',
  2: 'orange',
  1: 'gray',
  0: 'gray',
};

function getChannelBadge(channelType?: string | null): { label: string; tone: 'sky' | 'amber' | 'green' | 'gray' } {
  switch (channelType?.toLowerCase()) {
    case 'telegram':
      return { label: 'Канал: Telegram', tone: 'sky' };
    case 'email':
      return { label: 'Канал: E-mail', tone: 'amber' };
    case 'whatsapp':
      return { label: 'Канал: WhatsApp', tone: 'green' };
    default:
      return { label: 'Внутренний чат', tone: 'gray' };
  }
}

export type CommsMessage = DealMsg & {
  media_url?: string | null;
  media_name?: string | null;
  media_type?: string | null;
};

function ChatImage({ src, alt }: { src: string; alt?: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div className="flex items-center gap-2 p-2 rounded-xl bg-black/10 text-[11.5px] font-medium text-current/80 mb-1.5">
        <ImageOff size={14} className="shrink-0 opacity-70" />
        <span className="truncate">Изображение недоступно</span>
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={alt || "Вложение"}
      onError={() => setFailed(true)}
      className="max-w-full max-h-64 rounded-xl object-cover mb-1.5 cursor-pointer"
      onClick={() => window.open(src, "_blank")}
    />
  );
}

function VideoPlayer({ src }: { src: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div className="flex items-center gap-2 p-2 rounded-xl bg-black/10 text-[11.5px] font-medium text-current/80 mb-1.5">
        <ImageOff size={14} className="shrink-0 opacity-70" />
        <span className="truncate">Видеофайл недоступен</span>
      </div>
    );
  }
  return (
    <video controls src={src} onError={() => setFailed(true)} className="max-w-full max-h-64 rounded-xl mb-1.5" />
  );
}

/* ================= СДЕЛКИ ================= */
export function DealsList({ deals, onOpenDeal }: { deals: Deal[]; onOpenDeal: (id: string) => void }) {
  const [q, setQ] = useState('');
  const [fStage, setFStage] = useState('all');
  const [fBrand, setFBrand] = useState('all');
  const list = deals.filter(d =>
    (!q || d.title.toLowerCase().includes(q.toLowerCase()) || resolveAuthor(d).nick.includes(q.toLowerCase())) &&
    (fStage === 'all' || d.stage === Number(fStage)) &&
    (fBrand === 'all' || d.brandId === fBrand)
  );
  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="px-6 pt-5 pb-4 bg-gradient-to-b from-white to-transparent">
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-[22px] font-display font-semibold text-gray-900">Сделки</h1>
            <p className="text-[12.5px] font-medium text-gray-400 mt-1">{deals.length} сделок · суммарный бюджет {fmtMoney(deals.reduce((a, d) => a + d.budget, 0))}</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300" />
              <input className={inputCls + ' !w-56 !pl-9'} placeholder="Название или автор…" value={q} onChange={e => setQ(e.target.value)} />
            </div>
            <select className={selCls} value={fStage} onChange={e => setFStage(e.target.value)}>
              <option value="all">Все этапы</option>
              <option value="0">{ARCHIVED_STAGE.name}</option>
              {STAGES.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
            <select className={selCls} value={fBrand} onChange={e => setFBrand(e.target.value)}>
              <option value="all">Все бренды</option>
              {BRANDS.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select>
          </div>
        </div>
      </div>
      <div className="px-6 pb-6">
        <Card className="overflow-hidden">
          <div className="overflow-x-auto scroll-thin">
            <table className="w-full text-[12.5px] min-w-[900px]">
              <thead>
                <tr className="text-left text-[11px] font-bold text-gray-400 uppercase tracking-wide border-b border-gray-100 bg-slate-50/60">
                  {['Сделка', 'Бренд', 'Автор', 'Тип', 'Бюджет', 'Этап', 'Публикация', ''].map((h, i) => <th key={i} className="px-4 py-2.5 whitespace-nowrap">{h}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {list.map(d => {
                  const a = resolveAuthor(d), b = brandById(d.brandId), s = getStage(d.stage);
                  return (
                    <tr key={d.id} onClick={() => onOpenDeal(d.id)} className="group cursor-pointer hover:bg-indigo-50/40 transition-colors">
                      <td className="px-4 py-3 max-w-[260px]"><span className="font-bold text-gray-800 truncate block group-hover:text-indigo-600">{d.title}</span></td>
                      <td className="px-4 py-3"><span className="flex items-center gap-2"><span className="w-6 h-6 rounded-md text-[9px] font-extrabold text-white flex items-center justify-center shrink-0" style={{ background: 'hsl(' + b.hue + ' 70% 50%)' }}>{b.letter}</span><span className="font-semibold text-gray-600 whitespace-nowrap">{b.name}</span></span></td>
                      <td className="px-4 py-3"><span className="flex items-center gap-2"><Avatar nick={a.nick} hue={a.hue} size={24} /><span className="font-bold text-gray-700">{a.nick}</span><SocialIcon social={a.social} size={12} /></span></td>
                      <td className="px-4 py-3"><Badge tone={d.type === 'Stories' ? 'violet' : d.type === 'Reels' ? 'sky' : d.type === 'Видео' ? 'red' : 'indigo'}>{d.type}</Badge></td>
                      <td className="px-4 py-3 font-extrabold text-gray-900 tabular-nums whitespace-nowrap">{fmtMoney(d.budget)}</td>
                      <td className="px-4 py-3"><Badge tone={STAGE_TONES[d.stage] ?? 'gray'} dot>{s.name}</Badge></td>
                      <td className="px-4 py-3 font-semibold text-gray-500 tabular-nums whitespace-nowrap">{d.pubDate}</td>
                      <td className="px-4 py-3"><Eye size={15} className="text-gray-300 group-hover:text-indigo-500 transition-colors" /></td>
                    </tr>
                  );
                })}
                {list.length === 0 && <tr><td colSpan={8} className="px-4 py-14 text-center font-semibold text-gray-400">Сделок по этим фильтрам нет</td></tr>}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ================= КОММУНИКАЦИИ ================= */
export function CommsList({ deals, onOpenDeal, initialAuthorId, onNewDeal }: { deals: Deal[]; onOpenDeal?: (id: string) => void; initialAuthorId?: string | null; onNewDeal?: (authorId: string, meta?: { name?: string; handle?: string }) => void }) {
  const toast = useToast();
  const [channels, setChannels] = useState<CommunicationChannelItem[]>([]);
  const [selectedAuthorId, setSelectedAuthorId] = useState<string | null>(null);
  const [msgs, setMsgs] = useState<CommsMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [q, setQ] = useState('');
  const [chatTab, setChatTab] = useState<'active' | 'archived'>('active');
  const [newChatOpen, setNewChatOpen] = useState(false);
  const [confirmArchive, setConfirmArchive] = useState(false);
  const [closeDeals, setCloseDeals] = useState(true);
  const [attachedFile, setAttachedFile] = useState<FileUploadResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    api.getCommunications()
      .then(items => {
        if (cancelled) return;
        if (initialAuthorId) {
          const target = items.find(c => String(c.author_id) === String(initialAuthorId));
          if (target) {
            setChannels(items);
            setSelectedAuthorId(String(initialAuthorId));
          } else {
            api.getCreatorProfile(initialAuthorId)
              .then(profile => {
                if (cancelled) return;
                const temp: CommunicationChannelItem = {
                  deal_id: null,
                  author_id: initialAuthorId,
                  author_name: profile.title,
                  author_handle: profile.username ? `@${stripAt(profile.username)}` : `@${profile.title}`,
                  platform: profile.platform,
                  deal_title: 'Новый контакт',
                  stage: 1,
                  last_message: 'Диалог не начат',
                  last_message_time: new Date().toISOString(),
                  unread_count: 0,
                  is_archived: false,
                  channel_type: 'internal',
                };
                setChannels([temp, ...items]);
                setSelectedAuthorId(String(initialAuthorId));
              })
              .catch(() => {
                if (cancelled) return;
                setChannels(items);
                setSelectedAuthorId(items.length > 0 ? String(items[0].author_id) : null);
              });
          }
        } else {
          setChannels(items);
          setSelectedAuthorId(prev => prev ?? (items.length > 0 ? String(items[0].author_id) : null));
        }
      })
      .catch(() => {
        if (!cancelled) toast('err', 'Не удалось загрузить каналы');
      });
    return () => { cancelled = true; };
  }, [initialAuthorId]);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    return channels.filter(c => {
      if (chatTab === 'active' ? c.is_archived : !c.is_archived) return false;
      if (!s) return true;
      return c.author_name.toLowerCase().includes(s) || c.author_handle.toLowerCase().includes(s);
    });
  }, [channels, q, chatTab]);

  const activeCount = useMemo(() => channels.filter(c => !c.is_archived).length, [channels]);
  const archivedCount = useMemo(() => channels.filter(c => c.is_archived).length, [channels]);

  const ch = filtered.find(c => String(c.author_id) === String(selectedAuthorId)) ?? filtered[0] ?? null;
  const a = ch ? channelAuthor(ch) : null;
  const targetDeal = ch ? deals.find(d => (d.authorId === String(ch.author_id) || d.id === String(ch.deal_id)) && d.stage >= 1 && d.stage <= 6) : undefined;
  const activeDeals = ch ? deals.filter(d => (d.authorId === String(ch.author_id) || d.id === String(ch.deal_id)) && d.stage >= 1 && d.stage <= 6) : [];
  const activeBudget = activeDeals.reduce((a, d) => a + d.budget, 0);

  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: 'smooth' });
  }, [msgs.length, selectedAuthorId]);

  useEffect(() => {
    if (!ch) return;
    const authorId = String(ch.author_id);
    let isMounted = true;

    const fetchMessages = () => {
      api.getCreatorMessages(authorId)
        .then(items => {
          if (!isMounted) return;
          const mapped = items.map(toDealMsg);
          setMsgs(prev => {
            if (prev.length === mapped.length && prev[prev.length - 1]?.id === mapped[mapped.length - 1]?.id) {
              return prev;
            }
            return mapped;
          });
          if (items.length > 0) {
            const last = items[items.length - 1];
            setChannels(prev => prev.map(c => {
              if (String(c.author_id) !== authorId) return c;
              return {
                ...c,
                last_message: last.text || '',
                last_message_time: last.created_at,
                unread_count: 0,
              };
            }));
          }
        })
        .catch(() => {});
    };

    fetchMessages();
    const timer = setInterval(fetchMessages, 4000);

    return () => {
      isMounted = false;
      clearInterval(timer);
    };
  }, [ch?.author_id]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    api.uploadChatFile(file)
      .then(result => {
        setAttachedFile(result);
        if (fileInputRef.current) fileInputRef.current.value = '';
        toast('ok', `Файл "${file.name}" прикреплен`);
      })
      .catch(() => toast('err', 'Ошибка загрузки файла'))
      .finally(() => setIsUploading(false));
  };

  const send = () => {
    if ((!draft.trim() && !attachedFile) || !ch || isUploading) return;
    const text = draft.trim();
    const file = attachedFile;
    setDraft('');
    setAttachedFile(null);
    api.sendCreatorMessage(String(ch.author_id), text || null, 'user', file ? { media_url: file.media_url, media_name: file.media_name, media_type: file.media_type } : undefined)
      .then(m => {
        const dm = toDealMsg(m);
        setMsgs(prev => [...prev, dm]);
        setChannels(prev => prev.map(c => String(c.author_id) === String(ch.author_id) ? { ...c, last_message: dm.text, last_message_time: m.created_at } : c));
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Не удалось отправить сообщение';
        toast('err', msg);
      });
  };

  const handleStartChat = (targetId: string) => {
    api.initCommunication(targetId)
      .then(channel => {
        setChannels(prev => prev.some(c => String(c.author_id) === String(channel.author_id)) ? prev : [channel, ...prev]);
        setSelectedAuthorId(String(channel.author_id));
        setNewChatOpen(false);
        setChatTab('active');
      })
      .catch(() => toast('err', 'Не удалось начать диалог'));
  };

  const handleArchiveConfirm = () => {
    if (!ch) return;
    const targetId = String(ch.author_id);
    api.updateCreatorStatus(targetId, 'В архиве', activeDeals.length > 0 ? closeDeals : false)
      .then(() => {
        setChannels(prev => prev.map(c => String(c.author_id) === String(ch.author_id) ? { ...c, is_archived: true } : c));
        setConfirmArchive(false);
        toast('info', 'Чат перемещён в архив');
      })
      .catch(() => toast('err', 'Не удалось архивировать чат'));
  };

  const handleRestoreChat = () => {
    if (!ch) return;
    const targetId = String(ch.author_id);
    api.updateCreatorStatus(targetId, 'Свободен')
      .then(() => {
        setChannels(prev => prev.map(c => String(c.author_id) === String(ch.author_id) ? { ...c, is_archived: false } : c));
        setChatTab('active');
        toast('ok', 'Чат восстановлен из архива');
      })
      .catch(() => toast('err', 'Не удалось восстановить чат'));
  };

  return (
    <div className="h-full flex">
      <aside className="w-[320px] shrink-0 border-r border-gray-200 bg-white flex flex-col">
        <div className="px-4 pt-4 pb-3 flex items-center justify-between gap-2 shrink-0">
          <div className="min-w-0 flex-1">
            <h1 className="text-[17px] font-display font-bold text-gray-900 leading-tight truncate">Коммуникации</h1>
            <p className="text-[11.5px] font-medium text-gray-400 mt-0.5 truncate">Чаты сделок · {activeCount} активных</p>
          </div>
          <Btn size="sm" onClick={() => setNewChatOpen(true)} className="!h-8 !px-2.5 !text-[12px] shrink-0 font-semibold whitespace-nowrap">
            <Plus size={13} className="shrink-0" />Новый чат
          </Btn>
        </div>
        <div className="px-3 pb-2 flex items-center gap-1">
          <button onClick={() => setChatTab('active')}
            className={'flex-1 flex items-center justify-center gap-1.5 h-8 rounded-lg text-[12.5px] font-semibold transition-colors ' + (chatTab === 'active' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-500 hover:bg-slate-50')}>
            Активные ({activeCount})
          </button>
          <button onClick={() => setChatTab('archived')}
            className={'flex-1 flex items-center justify-center gap-1.5 h-8 rounded-lg text-[12.5px] font-semibold transition-colors ' + (chatTab === 'archived' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-500 hover:bg-slate-50')}>
            Архив ({archivedCount})
          </button>
        </div>
        <div className="px-3 pb-2">
          <div className="relative">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-300" />
            <input className={inputCls + ' !h-8.5 !pl-8 !text-[12.5px]'} placeholder="Поиск чата…" value={q} onChange={e => setQ(e.target.value)} />
          </div>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto scroll-thin px-2 flex flex-col gap-0.5">
          {filtered.map(c => {
            const ca = channelAuthor(c);
            return (
              <button key={String(c.author_id)} onClick={() => setSelectedAuthorId(String(c.author_id))}
                className={'flex items-center gap-2.5 px-2.5 py-2.5 rounded-xl text-left transition-colors ' + (String(selectedAuthorId) === String(c.author_id) ? 'bg-indigo-50' : 'hover:bg-slate-50')}>
                <Avatar nick={ca.nick} hue={ca.hue} size={36} />
                <span className="min-w-0 flex-1">
                  <span className={'flex items-center justify-between ' + (String(selectedAuthorId) === String(c.author_id) ? 'text-indigo-700' : 'text-gray-800')}>
                    <span className="text-[13px] font-bold truncate">{ca.nick}</span>
                    <span className="text-[10px] font-bold text-gray-300 shrink-0">{formatMessageTime(c.last_message_time)}</span>
                  </span>
                  <span className="flex items-center gap-1.5 mt-0.5">
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-100 text-gray-500 shrink-0">{getChannelBadge(c.channel_type).label}</span>
                    <span className="text-[11.5px] font-medium text-gray-400 truncate">{c.last_message}</span>
                  </span>
                </span>
                {c.unread_count > 0 && <span className="w-5 h-5 rounded-full bg-indigo-500 text-white text-[10px] font-extrabold flex items-center justify-center shrink-0">{c.unread_count}</span>}
              </button>
            );
          })}
          {filtered.length === 0 && <div className="text-center text-[12.5px] font-semibold text-gray-400 py-10">{chatTab === 'active' ? 'Нет активных чатов' : 'Архив пуст'}</div>}
        </div>
      </aside>
      <div className="flex-1 min-w-0 flex flex-col bg-slate-50/60">
        <div className="px-5 py-3.5 bg-white border-b border-gray-200 flex items-center gap-3 shrink-0">
          <Avatar nick={a?.nick ?? ''} hue={a?.hue ?? 0} size={36} />
          <div className="min-w-0">
            <div className="flex items-center gap-2"><span className="text-[14px] font-bold text-gray-900">{a?.nick ?? ''}</span>{a && <SocialIcon social={a.social} size={13} />}</div>
            <div className="text-[11px] font-semibold text-gray-400 truncate">Сделка: {ch?.deal_title}</div>
            {ch && (
              <Badge tone={getChannelBadge(ch.channel_type).tone}>
                {getChannelBadge(ch.channel_type).label}
              </Badge>
            )}
          </div>
          {targetDeal ? (
            <>
              <Badge tone={STAGE_TONES[targetDeal.stage] ?? 'gray'} dot>{getStage(targetDeal.stage).name}</Badge>
              <Btn variant="secondary" size="sm" className="ml-auto" onClick={() => onOpenDeal?.(String(targetDeal.id))}><Briefcase size={13} />К сделке</Btn>
            </>
          ) : (
            <>
              <Badge tone="violet">Аутрич / Переговоры</Badge>
              <Btn size="sm" className="ml-auto" onClick={() => ch && onNewDeal?.(String(ch.author_id), { name: ch.author_name, handle: ch.author_handle })}><Plus size={13} />Оформить сделку</Btn>
            </>
          )}
          {ch && !ch.is_archived ? (
            <Tip label="В архив"><Btn variant="ghost" size="sm" onClick={() => { setCloseDeals(true); setConfirmArchive(true); }}><Archive size={14} /></Btn></Tip>
          ) : (
            <Btn variant="secondary" size="sm" onClick={handleRestoreChat}><ArchiveRestore size={13} />Восстановить из архива</Btn>
          )}
        </div>
        <div ref={chatRef} className="flex-1 min-h-0 overflow-y-auto scroll-thin p-5 flex flex-col gap-3">
          {!msgs.some(m => m.from === 'user' || m.from === 'author') && <div className="m-auto text-[12.5px] font-semibold text-gray-400">Начните диалог — автор увидит сообщение в {a?.social ?? ''}</div>}
          {msgs.map(m => {
            if (m.from === 'system') {
              return (
                <div key={m.id} className="flex justify-center my-1.5">
                  <span className="text-[11px] font-medium text-gray-400 bg-gray-100/80 border border-gray-200/60 px-3 py-0.5 rounded-full">
                    {m.text} · {m.time}
                  </span>
                </div>
              );
            }
            return (
              <div key={m.id} className={'flex ' + (m.from === 'user' ? 'justify-end' : 'justify-start')}>
                <div className={'relative max-w-[60%] rounded-2xl px-3.5 py-2.5 text-[13px] font-medium leading-snug shadow-sm ' + (m.from === 'user' ? 'tail-r bg-indigo-500 text-white rounded-br-md' : 'tail-l bg-white border border-gray-200 text-gray-800 rounded-bl-md')}>
                  {m.media_url && m.media_type === 'image' && (
                    <ChatImage src={resolveMediaUrl(m.media_url)} alt={m.media_name || undefined} />
                  )}
                  {m.media_url && m.media_type === 'video' && (
                    <VideoPlayer src={resolveMediaUrl(m.media_url)} />
                  )}
                  {m.media_url && m.media_type !== 'image' && m.media_type !== 'video' && (
                    <a href={resolveMediaUrl(m.media_url)} target="_blank" download={m.media_name || 'document'} className="flex items-center gap-2 p-2.5 rounded-xl bg-black/5 hover:bg-black/10 transition-colors mb-1.5">
                      <FileText size={16} className="shrink-0" />
                      <span className="text-[12.5px] font-semibold truncate">{m.media_name || 'Документ'}</span>
                      <Download size={14} className="shrink-0" />
                    </a>
                  )}
                  {m.text && <span className="block">{m.text}</span>}
                  <span className={'block text-[10px] font-semibold mt-1 text-right ' + (m.from === 'user' ? 'text-indigo-200' : 'text-gray-300')}>{m.time} {m.from === 'user' && '✓✓'}</span>
                </div>
              </div>
            );
          })}
        </div>
        <div className="p-3.5 border-t border-gray-200 bg-white shrink-0 flex items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept="image/*,video/*,.pdf,.doc,.docx,.xls,.xlsx,.txt"
            onChange={handleFileSelect}
          />
          {attachedFile && (
            <div className="flex items-center gap-2 rounded-xl border border-indigo-200 bg-indigo-50 px-2.5 py-1.5 text-[12px] font-medium text-indigo-700 shrink-0">
              <FileText size={14} className="shrink-0" />
              <span className="max-w-[160px] truncate">{attachedFile.media_name}</span>
              <span className="text-indigo-400 shrink-0">{formatFileSize(attachedFile.file_size)}</span>
              <button onClick={() => setAttachedFile(null)} className="text-indigo-400 hover:text-indigo-700 shrink-0"><X size={13} /></button>
            </div>
          )}
          <button disabled={isUploading} className={'w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ' + (isUploading ? 'text-indigo-500 cursor-default' : 'text-gray-400 hover:bg-gray-100 hover:text-indigo-600')} onClick={() => fileInputRef.current?.click()}>
            {isUploading ? <Loader2 size={17} className="animate-spin" /> : <Paperclip size={17} />}
          </button>
          <textarea rows={1} value={draft}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder={'Сообщение для ' + (a?.nick ?? '') + '…'}
            className="flex-1 resize-none rounded-xl border border-gray-200 px-3.5 py-2.5 text-[13px] outline-none focus:border-indigo-400 focus:ring-4 focus:ring-indigo-100 transition-shadow" />
          <Btn onClick={send} disabled={(!draft.trim() && !attachedFile) || isUploading} className="!rounded-xl !w-9 !h-9.5 !p-0 shrink-0"><Send size={15} /></Btn>
        </div>
      </div>
      {newChatOpen && <NewChatModal channels={channels} onClose={() => setNewChatOpen(false)} onStart={handleStartChat} />}
      {confirmArchive && ch && (
        <Modal onClose={() => setConfirmArchive(false)} w="max-w-sm">
          <div className="px-5 py-4 border-b border-gray-200">
            <h2 className="text-[16px] font-display font-semibold text-gray-900">Архивировать диалог?</h2>
          </div>
          {activeDeals.length === 0 ? (
            <div className="px-5 py-4 text-[13px] font-medium text-gray-600 leading-relaxed">
              Чат и автор @{a?.nick ?? ''} будут перемещены в архив. Вы сможете восстановить их в любой момент.
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
            <Btn variant="secondary" size="sm" onClick={() => setConfirmArchive(false)}>Отмена</Btn>
            <Btn size="sm" onClick={handleArchiveConfirm}><Archive size={13} />В архив</Btn>
          </div>
        </Modal>
      )}
    </div>
  );
}

function NewChatModal({ channels, onClose, onStart }: {
  channels: CommunicationChannelItem[];
  onClose: () => void;
  onStart: (id: string) => void;
}) {
  const [creators, setCreators] = useState<CreatorRecord[]>([]);
  const [q, setQ] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    api.getCreators()
      .then(records => { if (!cancelled) { setCreators(records); setLoading(false); } })
      .catch(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const list = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return creators;
    return creators.filter(c => (c.name || c.Name || c.handle || c.username || '').toLowerCase().includes(s));
  }, [creators, q]);

  return (
    <Modal onClose={onClose} w="max-w-xl">
      <div className="px-5 py-4 border-b border-gray-200 flex items-center justify-between">
        <h2 className="text-[16px] font-display font-semibold text-gray-900">Новый чат</h2>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600"><X size={18} /></button>
      </div>
      <div className="px-5 py-3 border-b border-gray-200">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300" />
          <input className={inputCls + ' !pl-9'} placeholder="Поиск по нику или имени…" value={q} onChange={e => setQ(e.target.value)} />
        </div>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto scroll-thin p-3 flex flex-col gap-1.5">
        {loading && <div className="text-center text-[13px] font-semibold text-gray-400 py-10">Загрузка авторов…</div>}
        {!loading && list.length === 0 && <div className="text-center text-[13px] font-semibold text-gray-400 py-10">Авторы не найдены</div>}
        {list.map(creator => {
          const id = String(creator.accountid || creator.accountId || creator.id || creator.username || creator.handle || '');
          const hasChat = channels.some(c => String(c.author_id) === id);
          const nick = creator.name || creator.Name || creator.handle || creator.username || 'Без имени';
          const social = normalizeSocial(creator.platform || creator.Platform || 'Instagram');
          const followers = Number(creator.followers ?? creator.subscribers_count ?? 0);
          const hue = hashHue(id);
          return (
            <div key={id} className="flex items-center gap-3 px-3 py-2.5 rounded-xl border border-gray-100 hover:border-indigo-200 transition-colors">
              <Avatar nick={nick} hue={hue} size={38} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-[13.5px] font-bold text-gray-900 truncate">{nick}</span>
                  <SocialIcon social={social} size={13} />
                </div>
                <div className="text-[11.5px] font-semibold text-gray-400">{social} · {fmtBig(followers)} подписчиков</div>
              </div>
              {hasChat ? (
                <Badge tone="gray">Чат начат</Badge>
              ) : (
                <Btn size="sm" onClick={() => onStart(id)}><MessageSquare size={13} />Написать</Btn>
              )}
            </div>
          );
        })}
      </div>
    </Modal>
  );
}

const selCls = inputCls + ' !w-44 !h-9.5 text-[12.5px] font-semibold';

const fmtBig = (n: number) => n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1).replace('.0', '').replace('.', ',')}M` : n.toLocaleString('ru-RU');

const hashHue = (seed: string): number => {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) & 0x7fffffff;
  return h % 360;
};

const stripAt = (value: string): string => value.startsWith('@') ? value.slice(1) : value;

const formatFileSize = (bytes: number): string => {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1).replace('.0', '').replace('.', ',')} МБ`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1).replace('.0', '').replace('.', ',')} КБ`;
  return `${bytes} Б`;
};

const formatMessageTime = (isoString: string): string => {
  const date = new Date(isoString);
  if (isNaN(date.getTime())) return isoString;
  const now = new Date();
  const sameDay = date.getFullYear() === now.getFullYear() && date.getMonth() === now.getMonth() && date.getDate() === now.getDate();
  if (sameDay) {
    return date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
  }
  return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' });
};

const channelAuthor = (c: CommunicationChannelItem): Pick<Author, 'nick' | 'social' | 'hue'> => {
  return {
    nick: c.author_handle || c.author_name,
    hue: hashHue(String(c.author_id)),
    social: (c.platform === 'instagram' ? 'Instagram' : c.platform === 'telegram' ? 'Telegram' : c.platform === 'youtube' ? 'YouTube' : 'Instagram') as Social,
  };
};

const toDealMsg = (m: DealMessageItem): CommsMessage => ({
  id: String(m.id),
  from: m.sender_type === 'creator' ? 'author' : (m.sender_type === 'system' ? 'system' : 'user'),
  text: m.text || '',
  time: new Date(m.created_at).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
  media_url: m.media_url,
  media_name: m.media_name,
  media_type: m.media_type,
});
