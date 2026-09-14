import { useEffect, useMemo, useRef, useState } from 'react';
import { Search, Send, Paperclip, Eye, Briefcase, Archive } from 'lucide-react';
import { BRANDS, STAGES, fmtMoney, resolveAuthor, brandById, type Author, type Deal, type DealMsg, type Social } from '../data';
import { Badge, Avatar, Card, Btn, useToast, inputCls } from '../components/ui';
import { SocialIcon } from '../components/icons';
import { api, type CommunicationChannelItem, type DealMessageItem } from '../services/api';

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
export function CommsList({ deals, onOpenDeal }: { deals: Deal[]; onOpenDeal?: (id: string) => void }) {
  const toast = useToast();
  const [channels, setChannels] = useState<CommunicationChannelItem[]>([]);
  const [selectedDealId, setSelectedDealId] = useState<number | null>(null);
  const [msgs, setMsgs] = useState<DealMsg[]>([]);
  const [draft, setDraft] = useState('');
  const [q, setQ] = useState('');

  useEffect(() => {
    api.getCommunications()
      .then(items => {
        setChannels(items);
        if (selectedDealId === null && items.length > 0) {
          setSelectedDealId(items[0].deal_id);
        }
      })
      .catch(() => toast('err', 'Не удалось загрузить каналы'));
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return channels;
    return channels.filter(c => c.author_name.toLowerCase().includes(s) || c.author_handle.toLowerCase().includes(s));
  }, [channels, q]);

  const ch = filtered.find(c => c.deal_id === selectedDealId) ?? filtered[0] ?? null;
  const a = ch ? channelAuthor(ch) : null;

  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: 'smooth' });
  }, [msgs.length, selectedDealId]);

  useEffect(() => {
    if (!ch) return;
    api.getDealMessages(ch.deal_id)
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
    api.sendDealMessage(ch.deal_id, text)
      .then(m => {
        const dm = toDealMsg(m);
        setMsgs(prev => [...prev, dm]);
        setChannels(prev => prev.map(c => c.deal_id === ch.deal_id ? { ...c, last_message: dm.text, last_message_time: dm.time } : c));
      })
      .catch(() => toast('err', 'Не удалось отправить сообщение'));
  };

  return (
    <div className="h-full flex">
      <aside className="w-[320px] shrink-0 border-r border-gray-200 bg-white flex flex-col">
        <div className="px-4 pt-5 pb-3">
          <h1 className="text-[17px] font-display font-semibold text-gray-900">Коммуникации</h1>
          <p className="text-[11.5px] font-medium text-gray-400 mt-0.5">Чаты сделок · {channels.length} активных</p>
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
        </div>
      </aside>
      <div className="flex-1 min-w-0 flex flex-col bg-slate-50/60">
        <div className="px-5 py-3.5 bg-white border-b border-gray-200 flex items-center gap-3 shrink-0">
          <Avatar nick={a?.nick ?? ''} hue={a?.hue ?? 0} size={36} />
          <div className="min-w-0">
            <div className="flex items-center gap-2"><span className="text-[14px] font-bold text-gray-900">{a?.nick ?? ''}</span>{a && <SocialIcon social={a.social} size={13} />}</div>
            <div className="text-[11px] font-semibold text-gray-400 truncate">Сделка: {ch?.deal_title}</div>
          </div>
          <Btn variant="secondary" size="sm" className="ml-auto" onClick={() => ch && onOpenDeal?.(String(ch.deal_id))}><Briefcase size={13} />Сделка</Btn>
          <Btn variant="ghost" size="sm" onClick={() => toast('info', (a?.nick ?? '') + ' перемещён в архив чатов')}><Archive size={14} /></Btn>
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
    </div>
  );
}
const selCls = inputCls + ' !w-44 !h-9.5 text-[12.5px] font-semibold';

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
