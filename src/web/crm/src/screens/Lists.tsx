import { Archive, ArchiveRestore, Briefcase, Eye, MessageSquare, Paperclip, Plus, Search, Send, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { SocialIcon } from '../components/icons';
import { Avatar, Badge, Btn, Card, inputCls, Modal, Tip, useToast } from '../components/ui';
import { brandById, BRANDS, fmtMoney, resolveAuthor, STAGES, type Author, type Deal, type DealMsg, type Social } from '../data';
import { api, normalizeSocial, type CommunicationChannelItem, type CreatorRecord, type DealMessageItem } from '../services/api';

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
                  const a = resolveAuthor(d), b = brandById(d.brandId), s = STAGES.find(x => x.id === d.stage)!;
                  return (
                    <tr key={d.id} onClick={() => onOpenDeal(d.id)} className="group cursor-pointer hover:bg-indigo-50/40 transition-colors">
                      <td className="px-4 py-3 max-w-[260px]"><span className="font-bold text-gray-800 truncate block group-hover:text-indigo-600">{d.title}</span></td>
                      <td className="px-4 py-3"><span className="flex items-center gap-2"><span className="w-6 h-6 rounded-md text-[9px] font-extrabold text-white flex items-center justify-center shrink-0" style={{ background: 'hsl(' + b.hue + ' 70% 50%)' }}>{b.letter}</span><span className="font-semibold text-gray-600 whitespace-nowrap">{b.name}</span></span></td>
                      <td className="px-4 py-3"><span className="flex items-center gap-2"><Avatar nick={a.nick} hue={a.hue} size={24} /><span className="font-bold text-gray-700">{a.nick}</span><SocialIcon social={a.social} size={12} /></span></td>
                      <td className="px-4 py-3"><Badge tone={d.type === 'Stories' ? 'violet' : d.type === 'Reels' ? 'sky' : d.type === 'Видео' ? 'red' : 'indigo'}>{d.type}</Badge></td>
                      <td className="px-4 py-3 font-extrabold text-gray-900 tabular-nums whitespace-nowrap">{fmtMoney(d.budget)}</td>
                      <td className="px-4 py-3"><Badge tone={d.stage === 5 ? 'green' : d.stage === 4 ? 'indigo' : d.stage === 2 ? 'orange' : d.stage === 3 ? 'amber' : 'gray'} dot>{s.name}</Badge></td>
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
export function CommsList({ deals, onOpenDeal, initialAuthorId, onNewDeal }: { deals: Deal[]; onOpenDeal?: (id: string) => void; initialAuthorId?: string | null; onNewDeal?: (id: string) => void }) {
  const toast = useToast();
  const [channels, setChannels] = useState<CommunicationChannelItem[]>([]);
  const [selectedDealId, setSelectedDealId] = useState<number | null>(null);
  const [msgs, setMsgs] = useState<DealMsg[]>([]);
  const [draft, setDraft] = useState('');
  const [q, setQ] = useState('');
  const [chatTab, setChatTab] = useState<'active' | 'archived'>('active');
  const [newChatOpen, setNewChatOpen] = useState(false);
  const [confirmArchive, setConfirmArchive] = useState(false);
  const [closeDeals, setCloseDeals] = useState(true);

  useEffect(() => {
    let cancelled = false;
    api.getCommunications()
      .then(items => {
        if (cancelled) return;
        if (initialAuthorId) {
          const target = items.find(c => c.author_id === Number(initialAuthorId));
          if (target) {
            setChannels(items);
            setSelectedDealId(target.deal_id);
          } else {
            api.getCreatorProfile(initialAuthorId)
              .then(profile => {
                if (cancelled) return;
                const temp: CommunicationChannelItem = {
                  deal_id: -Number(initialAuthorId),
                  author_id: Number(initialAuthorId),
                  author_name: profile.title,
                  author_handle: profile.username ? `@${profile.username.replace(/^@/, '')}` : `@${profile.title}`,
                  platform: profile.platform,
                  deal_title: 'Новый контакт',
                  stage: 1,
                  last_message: 'Диалог не начат',
                  last_message_time: new Date().toISOString(),
                  unread_count: 0,
                  is_archived: false,
                };
                setChannels([temp, ...items]);
                setSelectedDealId(temp.deal_id);
              })
              .catch(() => {
                if (cancelled) return;
                setChannels(items);
                setSelectedDealId(items.length > 0 ? items[0].deal_id : null);
              });
          }
        } else {
          setChannels(items);
          setSelectedDealId(prev => prev ?? (items.length > 0 ? items[0].deal_id : null));
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

  const ch = filtered.find(c => c.deal_id === selectedDealId) ?? filtered[0] ?? null;
  const a = ch ? channelAuthor(ch) : null;
  const targetDeal = ch ? deals.find(d => d.authorId === String(ch.author_id) || d.id === String(ch.deal_id)) : undefined;
  const activeDeals = ch ? deals.filter(d => (d.authorId === String(ch.author_id) || d.id === String(ch.deal_id)) && d.stage >= 1 && d.stage <= 4) : [];
  const activeBudget = activeDeals.reduce((a, d) => a + d.budget, 0);

  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: 'smooth' });
  }, [msgs.length, selectedDealId]);

  useEffect(() => {
    if (!ch) return;
    const load = ch.deal_id > 0
      ? api.getDealMessages(ch.deal_id)
      : api.getCreatorMessages(String(ch.author_id));
    load
      .then(items => {
        setMsgs(items.map(toDealMsg));
        setChannels(prev => prev.map(c => c.deal_id === ch.deal_id ? { ...c, unread_count: 0 } : c));
      })
      .catch(() => toast('err', 'Не удалось загрузить сообщения'));
  }, [ch?.deal_id]);

  const send = () => {
    if (!draft.trim() || !ch) return;
    const text = draft.trim();
    setDraft('');
    const sendReq = ch.deal_id > 0
      ? api.sendDealMessage(ch.deal_id, text)
      : api.sendCreatorMessage(String(ch.author_id), text);
    sendReq
      .then(m => {
        const dm = toDealMsg(m);
        setMsgs(prev => [...prev, dm]);
        setChannels(prev => prev.map(c => c.deal_id === ch.deal_id ? { ...c, last_message: dm.text, last_message_time: dm.time, deal_id: m.deal_id } : c));
        if (ch.deal_id <= 0) setSelectedDealId(m.deal_id);
      })
      .catch(() => toast('err', 'Не удалось отправить сообщение'));
  };

  const handleStartChat = (creator: CreatorRecord) => {
    const targetId = (creator.handle || creator.username || creator.name || creator.Name || creator.accountid || creator.id || '').toString().replace(/^@/, '');
    api.initCommunication(targetId)
      .then(channel => {
        setChannels(prev => prev.some(c => c.deal_id === channel.deal_id) ? prev : [channel, ...prev]);
        setSelectedDealId(channel.deal_id);
        setNewChatOpen(false);
        setChatTab('active');
      })
      .catch(() => toast('err', 'Не удалось начать диалог'));
  };

  const handleArchiveConfirm = () => {
    if (!ch) return;
    const targetId = ch.author_handle ? ch.author_handle.replace(/^@/, '') : String(Math.abs(ch.author_id));
    api.updateCreatorStatus(targetId, 'В архиве', activeDeals.length > 0 ? closeDeals : false)
      .then(() => {
        setChannels(prev => prev.map(c => c.deal_id === ch.deal_id ? { ...c, is_archived: true } : c));
        setConfirmArchive(false);
        toast('info', 'Чат перемещён в архив');
      })
      .catch(() => toast('err', 'Не удалось архивировать чат'));
  };

  const handleRestoreChat = () => {
    if (!ch) return;
    const targetId = ch.author_handle ? ch.author_handle.replace(/^@/, '') : String(Math.abs(ch.author_id));
    api.updateCreatorStatus(targetId, 'Свободен')
      .then(() => {
        setChannels(prev => prev.map(c => c.deal_id === ch.deal_id ? { ...c, is_archived: false } : c));
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
              <button key={c.deal_id} onClick={() => setSelectedDealId(c.deal_id)}
                className={'flex items-center gap-2.5 px-2.5 py-2.5 rounded-xl text-left transition-colors ' + (selectedDealId === c.deal_id ? 'bg-indigo-50' : 'hover:bg-slate-50')}>
                <Avatar nick={ca.nick} hue={ca.hue} size={36} />
                <span className="min-w-0 flex-1">
                  <span className={'flex items-center justify-between ' + (selectedDealId === c.deal_id ? 'text-indigo-700' : 'text-gray-800')}>
                    <span className="text-[13px] font-bold truncate">{ca.nick}</span>
                    <span className="text-[10px] font-bold text-gray-300 shrink-0">{formatMessageTime(c.last_message_time)}</span>
                  </span>
                  <span className="block text-[11.5px] font-medium text-gray-400 truncate mt-0.5">{c.last_message}</span>
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
          </div>
          {targetDeal && targetDeal.budget > 0 ? (
            <>
              <Badge tone={targetDeal.stage === 5 ? 'green' : targetDeal.stage === 4 ? 'indigo' : targetDeal.stage === 2 ? 'orange' : targetDeal.stage === 3 ? 'amber' : 'gray'} dot>{STAGES.find(x => x.id === targetDeal.stage)?.name ?? 'Этап'}</Badge>
              <Btn variant="secondary" size="sm" className="ml-auto" onClick={() => onOpenDeal?.(String(targetDeal.id))}><Briefcase size={13} />К сделке</Btn>
            </>
          ) : (
            <>
              <Badge tone="violet">Аутрич / Переговоры</Badge>
              <Btn size="sm" className="ml-auto" onClick={() => ch && onNewDeal?.(String(ch.author_id))}><Plus size={13} />Оформить сделку</Btn>
            </>
          )}
          {ch && !ch.is_archived ? (
            <Tip label="В архив"><Btn variant="ghost" size="sm" onClick={() => { setCloseDeals(true); setConfirmArchive(true); }}><Archive size={14} /></Btn></Tip>
          ) : (
            <Btn variant="secondary" size="sm" onClick={handleRestoreChat}><ArchiveRestore size={13} />Восстановить из архива</Btn>
          )}
        </div>
        <div ref={chatRef} className="flex-1 min-h-0 overflow-y-auto scroll-thin p-5 flex flex-col gap-3">
          {msgs.length === 0 && <div className="m-auto text-[12.5px] font-semibold text-gray-400">Начните диалог — автор увидит сообщение в {a?.social ?? ''}</div>}
          {msgs.map(m => (
            <div key={m.id} className={'flex ' + (m.from === 'user' ? 'justify-end' : 'justify-start')}>
              <div className={'relative max-w-[60%] rounded-2xl px-3.5 py-2.5 text-[13px] font-medium leading-snug shadow-sm ' + (m.from === 'user' ? 'tail-r bg-indigo-500 text-white rounded-br-md' : 'tail-l bg-white border border-gray-200 text-gray-800 rounded-bl-md')}>
                {m.text}
                <span className={'block text-[10px] font-semibold mt-1 text-right ' + (m.from === 'user' ? 'text-indigo-200' : 'text-gray-300')}>{m.time} {m.from === 'user' && '✓✓'}</span>
              </div>
            </div>
          ))}
        </div>
        <div className="p-3.5 border-t border-gray-200 bg-white shrink-0 flex items-end gap-2">
          <button className="w-9 h-9 rounded-lg text-gray-400 hover:bg-gray-100 hover:text-indigo-600 flex items-center justify-center shrink-0" onClick={() => toast('info', 'Прикрепление файла…')}><Paperclip size={17} /></button>
          <textarea rows={1} value={draft}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder={'Сообщение для ' + (a?.nick ?? '') + '…'}
            className="flex-1 resize-none rounded-xl border border-gray-200 px-3.5 py-2.5 text-[13px] outline-none focus:border-indigo-400 focus:ring-4 focus:ring-indigo-100 transition-shadow" />
          <Btn onClick={send} disabled={!draft.trim()} className="!rounded-xl !w-9 !h-9.5 !p-0 shrink-0"><Send size={15} /></Btn>
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
  onStart: (creator: CreatorRecord) => void;
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
          const id = String(creator.accountid || creator.accountId || creator.id || '');
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
                <Btn size="sm" onClick={() => onStart(creator)}><MessageSquare size={13} />Написать</Btn>
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

const channelAuthor = (c: CommunicationChannelItem): Pick<Author, 'nick' | 'social' | 'hue'> => ({
  nick: c.author_handle || c.author_name,
  hue: Math.abs(c.author_id * 137) % 360,
  social: (c.platform === 'instagram' ? 'Instagram' : c.platform === 'telegram' ? 'Telegram' : c.platform === 'youtube' ? 'YouTube' : 'Instagram') as Social,
});

const toDealMsg = (m: DealMessageItem): DealMsg => ({
  id: String(m.id),
  from: m.sender_type === 'creator' ? 'author' : 'user',
  text: m.text,
  time: new Date(m.created_at).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
});
