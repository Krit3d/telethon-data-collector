import { useEffect, useRef, useState } from 'react';
import { X, Share2, MoreHorizontal, Pencil, Paperclip, Send, Upload, FileText, Check, ShieldCheck, BookOpen, Bold, Italic, List, Image as ImageIcon, FilePlus2, MessageCircle } from 'lucide-react';
import { STAGES, fmtMoney, authorById, brandById, type Deal, type DealMsg } from '../data';
import { SidePanel, Badge, Avatar, Btn, Tabs, useToast, Tip, CopyBtn, Field, inputCls } from '../components/ui';
import { SocialIcon, FakeQR } from '../components/icons';

export default function DealPanel({ deal, deals, initialTab, onClose, onUpdate, onOpenKB, onGenerateContract }: {
  deal: Deal;
  deals: Deal[];
  initialTab?: string;
  onClose: () => void;
  onUpdate: (patch: Partial<Deal>) => void;
  onOpenKB: (brandId?: string) => void;
  onGenerateContract: (deal: Deal) => void;
}) {
  const toast = useToast();
  const [tab, setTab] = useState(initialTab ?? 'overview');
  const [msg, setMsg] = useState('');
  const [uploading, setUploading] = useState<number | null>(null);
  const [checkOk, setCheckOk] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);
  const a = authorById(deal.authorId), b = brandById(deal.brandId);
  const stage = STAGES.find(s => s.id === deal.stage)!;

  useEffect(() => { chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: 'smooth' }); }, [deal.msgs.length, tab]);

  const send = () => {
    if (!msg.trim()) return;
    const t = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
    const m: DealMsg = { id: 'm' + Date.now(), kind: 'out', text: msg.trim(), time: t };
    onUpdate({ msgs: [...deal.msgs, m] });
    setMsg('');
    toast('ok', 'Сообщение отправлено автору');
  };

  const startUpload = () => {
    setUploading(0);
    const iv = setInterval(() => {
      setUploading(p => {
        if (p === null) return p;
        if (p >= 100) { clearInterval(iv); setUploading(null); toast('ok', 'Файл загружен: Презентация_кампании.pdf'); return null; }
        return p + 12;
      });
    }, 120);
  };

  const eridCode = deal.erid ?? 'ERID-1695115200-X2K9M4P7Q1';

  return (
    <SidePanel onClose={onClose} w="w-[640px]">
      {/* Шапка */}
      <div className="px-5 py-4 border-b border-gray-100 shrink-0">
        <div className="flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <h2 className="text-[16px] font-bold text-gray-900 leading-snug">{deal.title}</h2>
            <div className="flex items-center gap-2 mt-2">
              <div className="relative">
                <select
                  value={deal.stage}
                  onChange={e => { onUpdate({ stage: Number(e.target.value) }); toast('ok', `Статус: «${STAGES.find(s => s.id === Number(e.target.value))?.name}»`); }}
                  className="appearance-none pl-6.5 pr-7 h-8 rounded-lg border border-gray-200 bg-white text-[12.5px] font-bold text-gray-800 cursor-pointer hover:border-gray-300 transition-colors outline-none">
                  {STAGES.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
                </select>
                <span className="absolute left-2.5 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full pointer-events-none" style={{ background: stage.color }} />
                <span className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none text-[10px]">▾</span>
              </div>
              <Badge tone="gray">{deal.date}</Badge>
              <Badge tone={deal.type === 'Stories' ? 'violet' : deal.type === 'Reels' ? 'sky' : deal.type === 'Видео' ? 'red' : 'indigo'}>{deal.type}</Badge>
            </div>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <Btn variant="outline" size="sm" onClick={() => setTab('comms')}><MessageCircle size={14} />Написать автору</Btn>
            <Btn variant="ghost" size="sm" onClick={() => toast('info', 'Режим редактирования включён')}><Pencil size={14} />Редактировать</Btn>
            <Btn variant="ghost" size="sm" onClick={() => { navigator.clipboard?.writeText(location.href + '#' + deal.id).catch(() => {}); toast('ok', 'Ссылка на сделку скопирована'); }}><Share2 size={14} />Поделиться</Btn>
            <button className="w-8 h-8 rounded-lg text-gray-400 hover:bg-gray-100 flex items-center justify-center" onClick={() => toast('info', 'Действия: в архив, дублировать, удалить')}><MoreHorizontal size={16} /></button>
            <button onClick={onClose} className="w-8 h-8 rounded-lg text-gray-300 hover:text-gray-600 hover:bg-gray-100 flex items-center justify-center"><X size={18} /></button>
          </div>
        </div>
      </div>

      <Tabs active={tab} onChange={setTab} className="px-3 shrink-0" items={[
        { id: 'overview', label: 'Обзор' },
        { id: 'comms', label: `Коммуникации · ${deal.msgs.length}`, icon: <MessageCircle size={13} /> },
        { id: 'docs', label: 'Документы' },
        { id: 'erid', label: 'ERID' },
        { id: 'kb', label: 'База знаний', icon: <BookOpen size={13} /> },
      ]} />

      <div className="flex-1 min-h-0 overflow-y-auto scroll-thin">
        {/* ===== ОБЗОР ===== */}
        {tab === 'overview' && (
          <div className="p-5 flex flex-col gap-5 anim-in">
            <section>
              <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-3">Информация о сделке</h3>
              <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                <Info label="Название" v={deal.title} />
                <Info label="Бренд" v={<span className="flex items-center gap-2"><span className="w-6 h-6 rounded-md flex items-center justify-center text-[10px] font-extrabold text-white" style={{ background: `hsl(${b.hue} 70% 50%)` }}>{b.letter}</span>{b.name}</span>} />
                <Info label="Автор" v={<button onClick={() => toast('info', `Профиль ${a.nick}`)} className="flex items-center gap-2 hover:text-indigo-600 transition-colors"><Avatar nick={a.nick} hue={a.hue} size={22} /><SocialIcon social={a.social} size={12} />{a.nick}</button>} />
                <Info label="Бюджет" v={<b className="text-[15px] font-extrabold">{fmtMoney(deal.budget)}</b>} />
                <Info label="Тип контента" v={deal.type} />
                <Info label="Дата публикации" v={deal.pubDate} />
              </div>
              <div className="mt-4">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[12px] font-semibold text-gray-500">Описание · ТЗ</span>
                  <div className="flex items-center gap-0.5">
                    {[Bold, Italic, List, ImageIcon].map((I, i) => (
                      <button key={i} className="w-7 h-7 rounded-md text-gray-400 hover:bg-gray-100 hover:text-gray-700 flex items-center justify-center" onClick={() => toast('info', 'Форматирование применено')}><I size={13} /></button>
                    ))}
                  </div>
                </div>
                <div className="rounded-xl border border-gray-200 bg-slate-50/50 px-3.5 py-3 text-[13px] font-medium text-gray-700 leading-relaxed">{deal.desc}</div>
              </div>
            </section>

            <section>
              <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-3">Этапы сделки</h3>
              <div className="flex items-start">
                {STAGES.map((s, i) => (
                  <div key={s.id} className="flex-1 flex flex-col items-center relative">
                    {i > 0 && <span className={`absolute right-1/2 top-[13px] w-full h-[3px] rounded ${i < deal.stage ? 'bg-indigo-400' : 'bg-gray-200'}`} style={{ zIndex: 0 }} />}
                    <span className={`relative z-10 w-[26px] h-[26px] rounded-full flex items-center justify-center text-[11px] font-extrabold border-2 transition-all ${
                      i + 1 < deal.stage ? 'bg-indigo-500 border-indigo-500 text-white'
                        : i + 1 === deal.stage ? 'bg-white border-indigo-500 text-indigo-600 pulse-ring'
                          : 'bg-white border-gray-200 text-gray-300'}`}>
                      {i + 1 < deal.stage ? <Check size={13} /> : i + 1}
                    </span>
                    <span className={`text-[10px] font-bold mt-1.5 text-center leading-tight ${i + 1 === deal.stage ? 'text-indigo-600' : 'text-gray-400'}`}>{s.name}</span>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-3">Сводка по условиям</h3>
              <div className="rounded-xl border border-gray-200 divide-y divide-gray-100">
                <Row k="Условия сотрудничества" v={deal.terms} />
                <Row k="Эксклюзивность" v={deal.exclusive ? 'Да' : 'Нет'} />
                <Row k="Количество правок" v={String(deal.edits)} />
              </div>
            </section>

            <div className="flex gap-2">
              <Btn className="flex-1" onClick={() => onGenerateContract(deal)}><FilePlus2 size={14} />Сгенерировать договор</Btn>
              <Btn variant="secondary" onClick={() => toast('ok', `Напоминание создано: follow-up ${a.nick}`)}>Напомнить автору</Btn>
            </div>
          </div>
        )}

        {/* ===== КОММУНИКАЦИИ ===== */}
        {tab === 'comms' && (
          <div className="flex flex-col h-full anim-in">
            <div className="px-5 py-3 border-b border-indigo-100 bg-indigo-50/50 flex items-center gap-3 shrink-0">
              <div className="relative shrink-0">
                <Avatar nick={a.nick} hue={a.hue} size={36} />
                <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-500 border-2 border-white" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-[13px] font-bold text-gray-900">{a.nick}</span>
                  <SocialIcon social={a.social} size={13} />
                </div>
                <div className="text-[11px] font-semibold text-gray-400">Связь с автором по сделке · обычно отвечает в течение часа</div>
              </div>
              <Badge tone="indigo" className="ml-auto shrink-0">{deal.msgs.length} сообщ.</Badge>
            </div>
            <div ref={chatRef} className="flex-1 min-h-0 overflow-y-auto scroll-thin p-5 flex flex-col gap-3 bg-slate-50/50">
              {deal.msgs.length === 0 && <div className="m-auto text-center text-[12.5px] font-semibold text-gray-400">Сообщений пока нет — напишите автору первым</div>}
              {deal.msgs.map(m => m.kind === 'sys' ? (
                <div key={m.id} className="self-center max-w-[80%] text-center">
                  <span className="inline-flex items-center gap-1.5 text-[11px] font-bold text-violet-600 bg-violet-50 border border-violet-100 rounded-full px-3 py-1">
                    <FileText size={11} />{m.text} · {m.time}
                  </span>
                </div>
              ) : (
                <div key={m.id} className={`flex ${m.kind === 'out' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`relative max-w-[75%] rounded-2xl px-3.5 py-2.5 text-[13px] font-medium leading-snug shadow-sm ${
                    m.kind === 'out' ? 'tail-r bg-indigo-500 text-white rounded-br-md' : 'tail-l bg-white border border-gray-200 text-gray-800 rounded-bl-md'}`}>
                    {m.text}
                    <span className={`block text-[10px] font-semibold mt-1 text-right ${m.kind === 'out' ? 'text-indigo-200' : 'text-gray-300'}`}>{m.time} {m.kind === 'out' && '✓✓'}</span>
                  </div>
                </div>
              ))}
            </div>
            <div className="p-3.5 border-t border-gray-100 bg-white shrink-0">
              <div className="flex items-end gap-2">
                <button className="w-9 h-9 rounded-lg text-gray-400 hover:bg-gray-100 hover:text-indigo-600 flex items-center justify-center shrink-0 transition-colors"
                  onClick={() => toast('info', 'Прикрепление файла…')}><Paperclip size={17} /></button>
                <textarea
                  rows={1} value={msg}
                  onChange={e => setMsg(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
                  placeholder="Сообщение автору…"
                  className="flex-1 resize-none rounded-xl border border-gray-200 px-3.5 py-2.5 text-[13px] outline-none focus:border-indigo-400 focus:ring-4 focus:ring-indigo-100 transition-shadow" />
                <Btn onClick={send} disabled={!msg.trim()} className="!rounded-xl !w-9 !h-9.5 !p-0 shrink-0"><Send size={15} /></Btn>
              </div>
            </div>
          </div>
        )}

        {/* ===== ДОКУМЕНТЫ ===== */}
        {tab === 'docs' && (
          <div className="p-5 flex flex-col gap-4 anim-in">
            <div className="flex gap-2">
              <Btn variant="secondary" onClick={startUpload}><Upload size={14} />Загрузить файл</Btn>
              <Btn onClick={() => onGenerateContract(deal)}><FilePlus2 size={14} />Сгенерировать договор из шаблона</Btn>
            </div>
            {uploading !== null && (
              <div className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-3">
                <div className="flex items-center justify-between text-[12px] font-bold text-indigo-700 mb-1.5">
                  <span>Презентация_кампании.pdf</span><span className="tabular-nums">{Math.min(uploading, 100)}%</span>
                </div>
                <div className="h-1.5 rounded-full bg-indigo-100 overflow-hidden">
                  <div className="h-full rounded-full bg-indigo-500 transition-all duration-100" style={{ width: `${Math.min(uploading, 100)}%` }} />
                </div>
              </div>
            )}
            <div className="rounded-xl border border-gray-200 divide-y divide-gray-100">
              {[
                { n: 'Договор_2024-09-15.pdf', s: '245 KB', d: 'загружен 14.09.2024', t: 'pdf' },
                { n: 'ТЗ_крем_для_лица.docx', s: '128 KB', d: 'загружен 14.09.2024', t: 'doc' },
                { n: 'Скриншот_профиля.png', s: '1.2 MB', d: 'загружен 13.09.2024', t: 'img' },
              ].map(f => (
                <div key={f.n} className="flex items-center gap-3 px-4 py-3 group hover:bg-slate-50 transition-colors">
                  <span className={`w-9 h-9 rounded-lg flex items-center justify-center text-[9px] font-extrabold text-white shrink-0 ${f.t === 'pdf' ? 'bg-red-500' : f.t === 'doc' ? 'bg-blue-500' : 'bg-emerald-500'}`}>
                    {f.t === 'pdf' ? 'PDF' : f.t === 'doc' ? 'DOC' : 'IMG'}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] font-bold text-gray-800 truncate group-hover:text-indigo-600 transition-colors">{f.n}</div>
                    <div className="text-[11px] font-semibold text-gray-400">{f.s} · {f.d}</div>
                  </div>
                  <Btn variant="ghost" size="xs" onClick={() => toast('info', `Скачивание ${f.n}`)}>Скачать</Btn>
                </div>
              ))}
            </div>
            <div>
              <div className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-2">Превью · Договор_2024-09-15.pdf</div>
              <div className="doc-page rounded-lg border border-gray-100 p-6">
                <div className="text-center mb-4">
                  <div className="text-[13px] font-extrabold text-gray-900">ДОГОВОР ОКАЗАНИЯ УСЛУГ № 2024-0915</div>
                  <div className="text-[11px] font-semibold text-gray-400 mt-1">г. Москва · 15.09.2024</div>
                </div>
                <div className="space-y-2">
                  {[100, 96, 88, 100, 92, 60, 0, 98, 100, 74].map((w, i) => w > 0 && (
                    <div key={i} className="h-2 rounded bg-gray-100" style={{ width: `${w}%` }} />
                  ))}
                </div>
                <div className="mt-4 flex items-center justify-between text-[10px] font-bold text-gray-300">
                  <span>Страница 1 из 4</span><span>Маркировка: 38-ФЗ, ст. 18.1</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ===== ERID ===== */}
        {tab === 'erid' && (
          <div className="p-5 flex flex-col gap-4 anim-in">
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-4 flex gap-4">
              <FakeQR seed={eridCode} size={96} className="rounded-lg border border-gray-200 shrink-0" />
              <div className="min-w-0">
                <Badge tone="green" dot>ERID-маркер сгенерирован</Badge>
                <div className="font-mono text-[12.5px] font-bold text-gray-900 mt-2 break-all">{eridCode}</div>
                <div className="flex items-center gap-3 mt-2">
                  <CopyBtn text={eridCode} />
                  <Tip label="Соответствует требованиям 38-ФЗ «О рекламе» — данные переданы в ОРД"><Badge tone="green">38-ФЗ ✓</Badge></Tip>
                </div>
              </div>
            </div>
            <div className="rounded-xl border border-gray-200 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[12px] font-bold text-gray-400 uppercase tracking-wide">Проверка размещения</div>
                  <div className={`text-[13px] font-bold mt-1 ${checkOk ? 'text-emerald-600' : 'text-amber-600'}`}>
                    {checkOk ? '✓ Размещение подтверждено' : 'Ожидает публикации'}
                  </div>
                </div>
                {!checkOk && <Btn variant="outline" onClick={() => { setCheckOk(true); toast('ok', 'Проверка выполнена: нарушений не найдено'); }}><ShieldCheck size={14} />Проверить вручную</Btn>}
              </div>
            </div>
            <div>
              <div className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-2">История</div>
              <div className="rounded-xl border border-gray-200 overflow-hidden">
                <table className="w-full text-[12.5px]">
                  <thead><tr className="text-left text-[11px] font-bold text-gray-400 bg-slate-50/60 border-b border-gray-100">
                    <th className="px-4 py-2">Дата генерации</th><th className="px-2 py-2">Сгенерировал</th><th className="px-4 py-2 text-right">Статус</th>
                  </tr></thead>
                  <tbody className="divide-y divide-gray-50">
                    <tr><td className="px-4 py-2.5 font-bold text-gray-700">{deal.date}</td><td className="px-2 py-2.5 font-medium text-gray-500">Анна Соколова</td><td className="px-4 py-2.5 text-right"><Badge tone="green">Активен</Badge></td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ===== БАЗА ЗНАНИЙ ===== */}
        {tab === 'kb' && (
          <div className="p-5 flex flex-col gap-4 anim-in">
            <div className="flex items-center gap-3">
              <span className="w-11 h-11 rounded-xl flex items-center justify-center text-[15px] font-extrabold text-white shrink-0"
                style={{ background: `linear-gradient(135deg, hsl(${b.hue} 70% 52%), hsl(${b.hue + 25} 65% 42%))` }}>{b.letter}</span>
              <div>
                <div className="text-[15px] font-bold text-gray-900">{b.name}</div>
                <div className="text-[12px] font-medium text-gray-400">{b.shortDesc}</div>
              </div>
              <Badge tone="green" className="ml-auto" dot>{b.status}</Badge>
            </div>
            <section className="rounded-xl border border-gray-200 p-4">
              <h4 className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-2">Ключевые преимущества продукта</h4>
              <ul className="space-y-1.5">{b.usp.slice(0, 3).map(u => (
                <li key={u} className="flex gap-2 text-[12.5px] font-medium text-gray-700"><Check size={14} className="text-emerald-500 shrink-0 mt-0.5" />{u}</li>
              ))}</ul>
            </section>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-3.5">
                <h4 className="text-[11px] font-bold text-emerald-700 uppercase tracking-wide mb-2">Что можно говорить</h4>
                <ul className="space-y-1.5">{b.allowed.slice(0, 3).map(r => (
                  <li key={r} className="flex gap-1.5 text-[11.5px] font-medium text-gray-700 leading-snug"><Check size={12} className="text-emerald-500 shrink-0 mt-0.5" />{r}</li>
                ))}</ul>
              </div>
              <div className="rounded-xl border border-red-200 bg-red-50/40 p-3.5">
                <h4 className="text-[11px] font-bold text-red-600 uppercase tracking-wide mb-2">Что запрещено</h4>
                <ul className="space-y-1.5">{b.forbidden.slice(0, 3).map(r => (
                  <li key={r} className="flex gap-1.5 text-[11.5px] font-medium text-gray-700 leading-snug"><X size={12} className="text-red-500 shrink-0 mt-0.5" />{r}</li>
                ))}</ul>
              </div>
            </div>
            <section className="rounded-xl border border-gray-200 p-4">
              <h4 className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-2">Условия сотрудничества</h4>
              <div className="flex flex-wrap gap-1.5">
                {b.payTypes.filter(p => p.on).map(p => (
                  <Badge key={p.id} tone={p.def ? 'indigo' : 'gray'}>{p.def && <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />}{p.label}</Badge>
                ))}
              </div>
              <div className="text-[12px] font-semibold text-gray-500 mt-2.5">Для этой сделки: <b className="text-gray-800">{deal.terms}</b></div>
            </section>
            <Btn variant="outline" onClick={() => onOpenKB(b.id)}><BookOpen size={14} />Открыть полную базу знаний</Btn>
          </div>
        )}
      </div>
    </SidePanel>
  );
}

function Info({ label, v }: { label: string; v: React.ReactNode }) {
  return (
    <div>
      <div className="text-[11px] font-semibold text-gray-400 mb-0.5">{label}</div>
      <div className="text-[13px] font-semibold text-gray-800">{v}</div>
    </div>
  );
}
function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between gap-4 px-4 py-2.5">
      <span className="text-[12.5px] font-semibold text-gray-400">{k}</span>
      <span className="text-[12.5px] font-bold text-gray-800 text-right">{v}</span>
    </div>
  );
}
