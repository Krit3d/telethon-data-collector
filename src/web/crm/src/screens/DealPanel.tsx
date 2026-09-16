import { Bold, BookOpen, Check, Download, FilePlus2, Italic, List, MessageCircle, MoreHorizontal, Share2, ShieldCheck, Trash2, Upload, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { FakeQR, SocialIcon } from '../components/icons';
import { Avatar, Badge, Btn, CopyBtn, Modal, SidePanel, Tabs, Tip, useToast } from '../components/ui';
import { ARCHIVED_STAGE, brandById, fmtMoney, getStage, resolveAuthor, STAGES, type Deal, type DealMsg } from '../data';
import { api, type DealMessageItem } from '../services/api';

type DocType = 'pdf' | 'doc' | 'img' | 'other';
interface DealDocument {
  id: string;
  name: string;
  size: string;
  date: string;
  type: DocType;
  url?: string;
}

export default function DealPanel({ deal, deals, initialTab, onClose, onUpdate, onOpenKB, onGenerateContract, onDelete, onOpenAuthor, onOpenComms }: {
  deal: Deal;
  deals: Deal[];
  initialTab?: string;
  onClose: () => void;
  onUpdate: (patch: Partial<Deal>) => void;
  onOpenKB: (brandId?: string) => void;
  onGenerateContract: (deal: Deal) => void;
  onDelete?: (dealId: string) => void;
  onOpenAuthor?: (authorId: string) => void;
  onOpenComms?: (authorId: string) => void;
}) {
  const toast = useToast();
  const [tab, setTab] = useState(initialTab ?? 'overview');
  const [checkOk, setCheckOk] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const tzKey = `creatorflow_tz_${deal.id}`;
  const [termsText, setTermsText] = useState(() => {
    try {
      const saved = localStorage.getItem(tzKey);
      if (saved) return saved;
    } catch {
      // ignore
    }
    return deal.desc || deal.terms || '';
  });
  const chatRef = useRef<HTMLDivElement>(null);
  const termsRef = useRef<HTMLDivElement>(null);
  const a = resolveAuthor(deal), b = brandById(deal.brandId);
  const stage = getStage(deal.stage);

  const storageKey = `creatorflow_docs_${deal.id}`;
  const [docs, setDocs] = useState<DealDocument[]>(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      return saved ? (JSON.parse(saved) as DealDocument[]) : [];
    } catch {
      return [];
    }
  });
  const [selectedDocId, setSelectedDocId] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedDoc = docs.find(d => d.id === selectedDocId) ?? docs[0];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    const type: DocType = ext === 'pdf' ? 'pdf' : (ext === 'doc' || ext === 'docx') ? 'doc' : (ext === 'png' || ext === 'jpg' || ext === 'jpeg' || ext === 'webp' || ext === 'svg') ? 'img' : 'other';
    const size = file.size >= 1024 * 1024 ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : `${Math.max(1, Math.round(file.size / 1024))} KB`;
    const reader = new FileReader();
    reader.onload = () => {
      const doc: DealDocument = {
        id: `doc-${Date.now()}`,
        name: file.name,
        size,
        date: `загружен ${new Date().toLocaleDateString('ru-RU')}`,
        type,
        url: typeof reader.result === 'string' ? reader.result : undefined,
      };
      setDocs(prev => {
        const next = [doc, ...prev];
        try {
          localStorage.setItem(storageKey, JSON.stringify(next.filter(d => d.url)));
        } catch {
          // quota exceeded
        }
        return next;
      });
      setSelectedDocId(doc.id);
      toast('ok', `Файл ${file.name} успешно загружен`);
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const deleteDoc = (doc: DealDocument) => {
    setDocs(prev => {
      const next = prev.filter(d => d.id !== doc.id);
      try {
        localStorage.setItem(storageKey, JSON.stringify(next));
      } catch {
        // ignore
      }
      return next;
    });
    if (selectedDocId === doc.id) setSelectedDocId('');
    toast('ok', `Файл ${doc.name} удалён`);
  };

  const downloadDoc = (doc: DealDocument) => {
    if (doc.url) {
      const a = document.createElement('a');
      a.href = doc.url;
      a.download = doc.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } else {
      const blob = new Blob([`Документ: ${doc.name}\nРазмер: ${doc.size}\nДата: ${doc.date}\n\nОписание документа по сделке «${deal.title}».`], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = doc.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
    toast('ok', `Скачивание ${doc.name}`);
  };

  useEffect(() => { chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: 'smooth' }); }, [deal.msgs.length, tab]);

  useEffect(() => {
    if (tab !== 'comms' || !deal.authorId) return;
    api.getCreatorMessages(String(deal.authorId))
      .then(items => onUpdate({ msgs: items.map(toDealMsg) }))
      .catch(() => toast('err', 'Не удалось загрузить сообщения автора'));
  }, [tab, deal.authorId]);

  const confirmDelete = () => {
    setDeleting(true);
    api.deleteDeal(Number(deal.id))
      .then(() => { onDelete?.(deal.id); onClose(); })
      .catch(() => { setDeleting(false); setConfirmDeleteOpen(false); toast('err', 'Не удалось удалить сделку'); });
  };

  const eridCode = deal.erid ?? 'ERID-1695115200-X2K9M4P7Q1';

  const saveTerms = () => {
    const newContent = termsRef.current?.innerHTML ?? termsText;
    try {
      localStorage.setItem(tzKey, newContent);
    } catch {
      // ignore
    }
    onUpdate({ desc: newContent });
    setTermsText(newContent);
    toast('ok', 'ТЗ сохранено');
  };

  const remindAuthor = () => {
    api.sendCreatorMessage(String(deal.authorId), `Здравствуйте! Напоминаем по сделке «${deal.title}». Подскажите, пожалуйста, статус подготовки материалов?`, 'user')
      .then(m => {
        onUpdate({ msgs: [...deal.msgs, toDealMsg(m)] });
        onOpenComms?.(String(deal.authorId));
        toast('ok', 'Напоминание отправлено в чат');
      })
      .catch(() => toast('err', 'Не удалось отправить напоминание'));
  };

  const copyLink = () => {
    const base = window.location.origin + window.location.pathname;
    const currentHash = window.location.hash.split('?')[0] || '#/deals';
    const url = `${base}${currentHash}?dealId=${deal.id}`;
    try {
      navigator.clipboard.writeText(url);
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = url;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    }
    setMenuOpen(false);
    toast('ok', 'Ссылка на сделку скопирована');
  };

  return (
    <SidePanel onClose={onClose} w="w-[640px]">
      {/* Шапка */}
      <div className="px-5 py-4 border-b border-gray-100 shrink-0">
        <div className="flex items-start gap-3">
          <h2 className="flex-1 min-w-0 text-[17px] font-bold text-gray-900 leading-snug break-words">{deal.title}</h2>
          <button onClick={onClose} className="w-8 h-8 rounded-lg text-gray-300 hover:text-gray-600 hover:bg-gray-100 flex items-center justify-center shrink-0"><X size={18} /></button>
        </div>
        <div className="flex items-center gap-2 mt-2">
          <div className="relative">
            <select
              value={deal.stage}
              onChange={e => {
                const newStage = Number(e.target.value);
                const oldStage = deal.stage;
                onUpdate({ stage: newStage });
                api.updateDeal(Number(deal.id), { stage: newStage })
                  .then(() => toast('ok', `Статус: «${getStage(newStage).name}»`))
                  .catch(() => { onUpdate({ stage: oldStage }); toast('err', 'Не удалось обновить стадию'); });
              }}
              className="appearance-none pl-6.5 pr-7 h-8 rounded-lg border border-gray-200 bg-white text-[12.5px] font-bold text-gray-800 cursor-pointer hover:border-gray-300 transition-colors outline-none">
              {deal.stage === 0 && <option value={0}>{ARCHIVED_STAGE.name}</option>}
              {STAGES.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full pointer-events-none" style={{ background: stage.color }} />
            <span className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none text-[10px]">▾</span>
          </div>
          <Badge tone="gray">{deal.date}</Badge>
          <Badge tone={deal.type === 'Stories' ? 'violet' : deal.type === 'Reels' ? 'sky' : deal.type === 'Видео' ? 'red' : 'indigo'}>{deal.type}</Badge>
          <div className="flex items-center gap-1 ml-auto shrink-0">
            <Btn variant="outline" size="sm" onClick={() => onOpenComms?.(String(deal.authorId))}><MessageCircle size={14} />Написать автору</Btn>
            <div className="relative">
              <button className="w-8 h-8 rounded-lg text-gray-400 hover:bg-gray-100 flex items-center justify-center" onClick={() => setMenuOpen(!menuOpen)}><MoreHorizontal size={16} /></button>
              {menuOpen && (
                <>
                  <div className="fixed inset-0 z-20" onClick={() => setMenuOpen(false)} />
                  <div className="absolute right-0 top-full mt-1 z-30 bg-white rounded-xl border border-gray-200 shadow-lg min-w-[210px] py-1">
                    <button onClick={copyLink} className="w-full flex items-center gap-2 px-3 py-2 text-[13px] font-medium text-gray-700 hover:bg-gray-50 transition-colors"><Share2 size={14} />Копировать ссылку</button>
                    <button onClick={() => { setMenuOpen(false); setConfirmDeleteOpen(true); }} className="w-full flex items-center gap-2 px-3 py-2 text-[13px] font-medium text-red-600 hover:bg-red-50 transition-colors"><Trash2 size={14} />Удалить сделку</button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <Tabs active={tab} onChange={setTab} className="px-3 shrink-0" items={[
        { id: 'overview', label: 'Обзор' },
        { id: 'comms', label: `Коммуникации · ${deal.msgs.length}`, icon: <MessageCircle size={13} /> },
        { id: 'docs', label: 'Документы' },
        // { id: 'erid', label: 'ERID' },
        // { id: 'kb', label: 'База знаний', icon: <BookOpen size={13} /> },
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
                <Info label="Автор" v={<button onClick={() => onOpenAuthor?.(deal.authorId)} className="flex items-center gap-2 hover:text-indigo-600 transition-colors"><Avatar nick={a.nick} hue={a.hue} size={22} /><SocialIcon social={a.social} size={12} />{a.nick}</button>} />
                <Info label="Бюджет" v={<b className="text-[15px] font-extrabold">{fmtMoney(deal.budget)}</b>} />
                <Info label="Тип контента" v={deal.type} />
                <Info label="Дата публикации" v={deal.pubDate} />
              </div>
              <div className="mt-4">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[12px] font-semibold text-gray-500">Описание · ТЗ</span>
                  <div className="flex items-center gap-0.5">
                    <button onMouseDown={e => e.preventDefault()} className="w-7 h-7 rounded-md text-gray-400 hover:bg-gray-100 hover:text-gray-700 flex items-center justify-center" onClick={() => document.execCommand('bold')}><Bold size={13} /></button>
                    <button onMouseDown={e => e.preventDefault()} className="w-7 h-7 rounded-md text-gray-400 hover:bg-gray-100 hover:text-gray-700 flex items-center justify-center" onClick={() => document.execCommand('italic')}><Italic size={13} /></button>
                    <button onMouseDown={e => e.preventDefault()} className="w-7 h-7 rounded-md text-gray-400 hover:bg-gray-100 hover:text-gray-700 flex items-center justify-center" onClick={() => document.execCommand('insertUnorderedList')}><List size={13} /></button>
                  </div>
                </div>
                <div
                  ref={termsRef}
                  contentEditable
                  suppressContentEditableWarning
                  onBlur={saveTerms}
                  className="rounded-xl border border-gray-200 bg-slate-50/50 px-3.5 py-3 text-[13px] font-medium text-gray-700 leading-relaxed [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-1.5 [&_li]:my-0.5 [&_b]:font-bold [&_strong]:font-bold [&_i]:italic [&_em]:italic outline-none focus:ring-2 focus:ring-indigo-100"
                  dangerouslySetInnerHTML={{ __html: termsText }}
                />
              </div>
            </section>

            <section>
              <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-3">Этапы сделки</h3>
              <div className="flex items-start">
                {STAGES.map((s, i) => (
                  <div key={s.id} className="flex-1 flex flex-col items-center relative">
                    {i > 0 && <span className={`absolute right-1/2 top-[13px] w-full h-[3px] rounded ${i < deal.stage ? 'bg-indigo-400' : 'bg-gray-200'}`} style={{ zIndex: 0 }} />}
                    <span className={`relative z-10 w-[26px] h-[26px] rounded-full flex items-center justify-center text-[11px] font-extrabold border-2 transition-all ${i + 1 < deal.stage ? 'bg-indigo-500 border-indigo-500 text-white'
                        : i + 1 === deal.stage ? 'bg-white border-indigo-500 text-indigo-600 pulse-ring'
                          : 'bg-white border-gray-200 text-gray-300'}`}>
                      {i + 1 < deal.stage ? <Check size={13} /> : i + 1}
                    </span>
                    <span className={`text-[10px] font-bold mt-1.5 text-center leading-tight truncate max-w-[85px] ${i + 1 === deal.stage ? 'text-indigo-600' : 'text-gray-400'}`}>{s.name}</span>
                  </div>
                ))}
              </div>
              {deal.stage === 0 && (
                <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3 text-[12.5px] font-semibold text-slate-600 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-slate-400 shrink-0" />
                  Сделка находится в архиве (сорвана)
                </div>
              )}
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
              <Btn variant="secondary" onClick={remindAuthor}>Напомнить автору</Btn>
            </div>
          </div>
        )}

        {/* ===== КОММУНИКАЦИИ ===== */}
        {tab === 'comms' && (
          <div className="p-5 anim-in">
            <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100 flex items-center gap-3">
                <span className="w-9 h-9 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center shrink-0"><MessageCircle size={16} /></span>
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-bold text-gray-900">Диалог с автором</div>
                  <div className="text-[11px] font-semibold text-gray-400">Связь с {a.nick} · {deal.msgs.length} сообщ.</div>
                </div>
                <Badge tone="indigo" className="shrink-0">Канал: Внутренний чат</Badge>
              </div>
              <div className="px-4 py-3 flex flex-col gap-2.5">
                {deal.msgs.slice(-3).map(m => (
                  <div key={m.id} className="flex items-start gap-2">
                    <span className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${m.from === 'user' ? 'bg-indigo-500' : 'bg-gray-300'}`} />
                    <div className="min-w-0">
                      <div className="text-[12.5px] font-medium text-gray-700 leading-snug line-clamp-2">{m.text}</div>
                      <div className="text-[10.5px] font-semibold text-gray-300 mt-0.5">{m.time} · {m.from === 'user' ? 'Вы' : a.nick}</div>
                    </div>
                  </div>
                ))}
                {deal.msgs.length === 0 && <div className="text-[12.5px] font-semibold text-gray-400">Сообщений пока нет — напишите автору первым</div>}
              </div>
              <div className="px-4 py-3 border-t border-gray-100 bg-slate-50/50">
                <Btn className="w-full" onClick={() => onOpenComms?.(String(deal.authorId))}><MessageCircle size={14} />Перейти в диалог с автором</Btn>
              </div>
            </div>
          </div>
        )}

        {/* ===== ДОКУМЕНТЫ ===== */}
        {tab === 'docs' && (
          <div className="p-5 flex flex-col gap-4 anim-in">
            <div className="flex gap-2">
              <Btn variant="secondary" onClick={() => fileInputRef.current?.click()}><Upload size={14} />Загрузить файл</Btn>
              <Btn onClick={() => onGenerateContract(deal)}><FilePlus2 size={14} />Сгенерировать договор из шаблона</Btn>
            </div>
            <input ref={fileInputRef} type="file" accept=".pdf,.doc,.docx,.png,.jpg,.jpeg,.txt" className="hidden" onChange={handleFileChange} />
            {docs.length === 0 ? (
              <div className="rounded-xl border border-dashed border-gray-300 bg-slate-50/50 p-6 text-center">
                <div className="text-[13px] font-semibold text-gray-600">Нет прикрепленных файлов. Загрузите документ или сгенерируйте договор из шаблона.</div>
              </div>
            ) : (
              <div className="rounded-xl border border-gray-200 divide-y divide-gray-100">
                {docs.map(doc => (
                  <div key={doc.id} onClick={() => setSelectedDocId(doc.id)} className={`flex items-center gap-3 px-4 py-3 group transition-colors cursor-pointer ${doc.id === selectedDocId ? 'border-l-2 border-indigo-200 bg-indigo-50/30' : 'hover:bg-slate-50'}`}>
                    <span className={`w-9 h-9 rounded-lg flex items-center justify-center text-[9px] font-extrabold text-white shrink-0 ${doc.type === 'pdf' ? 'bg-red-500' : doc.type === 'doc' ? 'bg-blue-500' : doc.type === 'img' ? 'bg-emerald-500' : 'bg-gray-500'}`}>
                      {doc.type === 'pdf' ? 'PDF' : doc.type === 'doc' ? 'DOC' : doc.type === 'img' ? 'IMG' : 'FILE'}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="text-[13px] font-bold text-gray-800 truncate group-hover:text-indigo-600 transition-colors">{doc.name}</div>
                      <div className="text-[11px] font-semibold text-gray-400">{doc.size} · {doc.date}</div>
                    </div>
                    <span onClick={e => e.stopPropagation()} className="flex items-center gap-1.5">
                      <Btn variant="ghost" size="xs" onClick={() => downloadDoc(doc)}><Download size={13} />Скачать</Btn>
                      <button onClick={() => deleteDoc(doc)} className="w-7 h-7 rounded-md text-gray-400 hover:bg-red-50 hover:text-red-600 flex items-center justify-center transition-colors" title="Удалить файл"><Trash2 size={14} /></button>
                    </span>
                  </div>
                ))}
              </div>
            )}
            {selectedDoc && (
              <div>
                <div className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-2">Превью · {selectedDoc.name}</div>
                {selectedDoc.type === 'img' && selectedDoc.url ? (
                  <img src={selectedDoc.url} alt={selectedDoc.name} className="max-h-[360px] w-full object-contain rounded-lg border border-gray-100" />
                ) : selectedDoc.type === 'pdf' && selectedDoc.url ? (
                  <iframe src={selectedDoc.url} title={selectedDoc.name} className="w-full h-[360px] rounded-lg border border-gray-100" />
                ) : selectedDoc.type === 'pdf' ? (
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
                ) : (
                  <div className="rounded-xl border border-gray-200 p-5">
                    <div className="flex items-center gap-3">
                      <span className={`w-11 h-11 rounded-xl flex items-center justify-center text-[10px] font-extrabold text-white shrink-0 ${selectedDoc.type === 'doc' ? 'bg-blue-500' : 'bg-gray-500'}`}>
                        {selectedDoc.type === 'doc' ? 'DOC' : 'FILE'}
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="text-[14px] font-bold text-gray-900 truncate">{selectedDoc.name}</div>
                        <div className="text-[11px] font-semibold text-gray-400">{selectedDoc.size} · {selectedDoc.date}</div>
                      </div>
                      <Badge tone={selectedDoc.type === 'doc' ? 'blue' : 'gray'}>{selectedDoc.type === 'doc' ? 'DOCX' : 'Файл'}</Badge>
                    </div>
                    <div className="mt-4 space-y-2">
                      {[100, 96, 88, 100, 92, 60, 0, 98, 100, 74].map((w, i) => w > 0 && (
                        <div key={i} className="h-2 rounded bg-gray-100" style={{ width: `${w}%` }} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
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
      {confirmDeleteOpen && (
        <Modal onClose={() => setConfirmDeleteOpen(false)}>
          <div className="px-5 py-4">
            <h3 className="text-[16px] font-bold text-gray-900">Удалить сделку?</h3>
            <p className="text-[13px] font-medium text-gray-600 leading-relaxed mt-2">Вы уверены, что хотите удалить сделку «{deal.title}»? Вся переписка и условия будут удалены безвозвратно.</p>
          </div>
          <div className="flex justify-end gap-2 px-5 py-3 border-t border-gray-100 shrink-0">
            <Btn variant="secondary" onClick={() => setConfirmDeleteOpen(false)}>Отмена</Btn>
            <Btn variant="danger" disabled={deleting} onClick={confirmDelete}>Удалить</Btn>
          </div>
        </Modal>
      )}
    </SidePanel>
  );
}

const toDealMsg = (m: DealMessageItem): DealMsg => ({
  id: String(m.id),
  from: m.sender_type === 'creator' ? 'author' : (m.sender_type === 'system' ? 'system' : 'user'),
  text: m.text || '',
  time: new Date(m.created_at).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
});

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
