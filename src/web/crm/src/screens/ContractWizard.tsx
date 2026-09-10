import { useState } from 'react';
import { X, ChevronLeft, ChevronRight, FileText, Check, AlertTriangle, ZoomIn, ZoomOut, Download, PenTool, Mail, Send, Link2, BookOpen } from 'lucide-react';
import { TEMPLATES, authorById, brandById, fmtMoney, type Deal } from '../data';
import { Modal, Badge, Btn, Field, inputCls, useToast, Toggle, Avatar } from '../components/ui';

const STEPS = ['Выбор шаблона', 'Заполнение данных', 'Предпросмотр', 'Отправка'];

export default function ContractWizard({ deal, onClose, onDone }: { deal: Deal; onClose: () => void; onDone: () => void }) {
  const toast = useToast();
  const [step, setStep] = useState(0);
  const [tpl, setTpl] = useState<string | null>(null);
  const [exclusive, setExclusive] = useState(deal.exclusive);
  const [edits, setEdits] = useState(deal.edits);
  const [eridOn, setEridOn] = useState(true);
  const [nda, setNda] = useState(false);
  const [zoom, setZoom] = useState(100);
  const [method, setMethod] = useState('email');
  const [toTimeline, setToTimeline] = useState(true);
  const b = brandById(deal.brandId), a = authorById(deal.authorId);

  const checklist = [
    { ok: true, t: 'Предмет договора указан' },
    { ok: true, t: 'Права на контент определены' },
    { ok: eridOn, t: 'ERID-маркировка включена (38-ФЗ)' },
    { ok: true, t: 'Порядок оплаты прописан' },
    { ok: nda, t: 'NDA о неразглашении условий', warn: true },
  ];
  const allOk = checklist.every(c => c.ok);

  return (
    <Modal onClose={onClose} w="max-w-[820px]">
      {/* Шапка wizard */}
      <div className="px-5 py-4 border-b border-gray-100 shrink-0">
        <div className="flex items-center">
          <h2 className="text-[15.5px] font-bold text-gray-900">Создание договора</h2>
          <span className="text-[12px] font-semibold text-gray-400 ml-3 truncate max-w-[400px]">{deal.title}</span>
          <button onClick={onClose} className="ml-auto text-gray-300 hover:text-gray-600 shrink-0"><X size={18} /></button>
        </div>
        <div className="flex items-center mt-3.5">
          {STEPS.map((s, i) => (
            <div key={s} className="flex items-center flex-1 last:flex-none">
              <button onClick={() => i < step && setStep(i)} className={`flex items-center gap-2 ${i < step ? 'cursor-pointer' : 'cursor-default'}`}>
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-[10.5px] font-extrabold border-2 transition-all ${
                  i < step ? 'bg-emerald-500 border-emerald-500 text-white' : i === step ? 'bg-indigo-500 border-indigo-500 text-white' : 'bg-white border-gray-200 text-gray-400'}`}>
                  {i < step ? <Check size={11} /> : i + 1}
                </span>
                <span className={`text-[11.5px] font-bold whitespace-nowrap ${i === step ? 'text-indigo-600' : i < step ? 'text-emerald-600' : 'text-gray-400'}`}>{s}</span>
              </button>
              {i < STEPS.length - 1 && <span className={`flex-1 h-[2px] rounded mx-3 ${i < step ? 'bg-emerald-400' : 'bg-gray-200'}`} />}
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto scroll-thin p-5">
        {/* ШАГ 1 */}
        {step === 0 && (
          <div className="grid grid-cols-2 gap-3 anim-in">
            {TEMPLATES.map(t => (
              <button key={t.id} onClick={() => setTpl(t.id)}
                className={`relative text-left rounded-xl border-2 p-4 transition-all hover:-translate-y-0.5 hover:shadow-md ${tpl === t.id ? 'border-indigo-500 bg-indigo-50/40 shadow-md' : 'border-gray-200 hover:border-gray-300'}`}>
                {t.badge && <Badge tone="indigo" className="absolute -top-2.5 right-3 shadow-sm">{t.badge}</Badge>}
                <div className="flex items-center gap-2.5">
                  <span className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${tpl === t.id ? 'bg-indigo-500 text-white' : 'bg-slate-100 text-gray-500'}`}><FileText size={16} /></span>
                  <span className="text-[13.5px] font-bold text-gray-900">{t.name}</span>
                </div>
                <p className="text-[11.5px] font-medium text-gray-400 mt-2 leading-snug">{t.desc}</p>
                <div className="mt-3 rounded-lg bg-slate-50 border border-gray-100 px-3 py-2 space-y-1">
                  {t.preview.slice(0, 3).map((l, i) => <div key={i} className="text-[10.5px] font-mono text-gray-400 truncate">{l}</div>)}
                </div>
              </button>
            ))}
          </div>
        )}

        {/* ШАГ 2 */}
        {step === 1 && (
          <div className="grid grid-cols-[1fr_280px] gap-5 anim-in">
            <div className="flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-4">
                <Field label="Бренд"><div className={inputCls + ' flex items-center gap-2 !font-bold'}><span className="w-5 h-5 rounded-md text-white text-[9px] font-extrabold flex items-center justify-center" style={{ background: `hsl(${b.hue} 70% 50%)` }}>{b.letter}</span>{b.name}</div></Field>
                <Field label="Автор"><div className={inputCls + ' flex items-center gap-2 !font-bold'}><Avatar nick={a.nick} hue={a.hue} size={18} />{a.nick}</div></Field>
                <Field label="Бюджет (автозаполнение)"><div className={inputCls + ' !font-extrabold tabular-nums'}>{fmtMoney(deal.budget)}</div></Field>
                <Field label="Сроки"><div className={inputCls}>до {deal.pubDate} · публикация</div></Field>
              </div>
              <div className="rounded-xl border border-gray-200 divide-y divide-gray-100">
                <div className="flex items-center justify-between px-4 py-3">
                  <div><div className="text-[13px] font-bold text-gray-800">Эксклюзивность</div><div className="text-[11px] font-medium text-gray-400">Автор не работает с конкурентами 30 дней</div></div>
                  <Toggle on={exclusive} onChange={setExclusive} />
                </div>
                <div className="flex items-center justify-between px-4 py-3">
                  <div><div className="text-[13px] font-bold text-gray-800">Количество правок</div><div className="text-[11px] font-medium text-gray-400">Бесплатные правки по ТЗ</div></div>
                  <div className="flex items-center gap-2">
                    <button className="w-7 h-7 rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-50 font-bold" onClick={() => setEdits(e => Math.max(0, e - 1))}>−</button>
                    <span className="w-6 text-center text-[14px] font-extrabold tabular-nums">{edits}</span>
                    <button className="w-7 h-7 rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-50 font-bold" onClick={() => setEdits(e => e + 1)}>+</button>
                  </div>
                </div>
                <div className="flex items-center justify-between px-4 py-3">
                  <div><div className="text-[13px] font-bold text-gray-800">ERID-обязательность</div><div className="text-[11px] font-medium text-gray-400">Пункт о маркировке по 38-ФЗ (включено по умолчанию)</div></div>
                  <Toggle on={eridOn} onChange={setEridOn} />
                </div>
              </div>
            </div>
            <aside className="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 h-fit">
              <div className="flex items-center gap-1.5 text-[11px] font-bold text-indigo-700 uppercase tracking-wide mb-3"><BookOpen size={13} />Подсказки из базы знаний</div>
              <div className="flex flex-col gap-2.5">
                <div className="rounded-lg bg-white border border-indigo-100 p-3 text-[11.5px] font-medium text-gray-700 leading-relaxed">
                  Рекомендуем пункт о <b>неэксклюзивности</b> — {a.nick} часто работает с конкурентами категории (по истории сделок).
                </div>
                <div className="rounded-lg bg-white border border-indigo-100 p-3 text-[11.5px] font-medium text-gray-700 leading-relaxed">
                  {b.name} требует <b>обязательное упоминание ERID</b> в первых 3 кадрах (из базы знаний бренда).
                </div>
                {b.mentions[0] && (
                  <div className="rounded-lg bg-white border border-indigo-100 p-3 text-[11.5px] font-medium text-gray-700 leading-relaxed">
                    Обязательное упоминание: <b>{b.mentions[0]}</b>
                  </div>
                )}
              </div>
            </aside>
          </div>
        )}

        {/* ШАГ 3 */}
        {step === 2 && (
          <div className="grid grid-cols-[1fr_260px] gap-5 anim-in">
            <div>
              <div className="flex items-center gap-1 mb-2.5">
                <button className="w-8 h-8 rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-50 flex items-center justify-center" onClick={() => setZoom(z => Math.max(60, z - 10))}><ZoomOut size={14} /></button>
                <span className="text-[12px] font-bold text-gray-500 w-12 text-center tabular-nums">{zoom}%</span>
                <button className="w-8 h-8 rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-50 flex items-center justify-center" onClick={() => setZoom(z => Math.min(140, z + 10))}><ZoomIn size={14} /></button>
                <div className="ml-auto flex gap-1.5">
                  <Btn variant="secondary" size="xs" onClick={() => toast('ok', 'PDF сохранён в Документы сделки')}><Download size={12} />Download</Btn>
                  <Btn variant="secondary" size="xs" onClick={() => { setStep(3); }}><PenTool size={12} />Send for e-signature</Btn>
                </div>
              </div>
              <div className="rounded-xl border border-gray-200 bg-slate-100/70 p-5 overflow-auto scroll-thin max-h-[420px]">
                <div className="doc-page rounded-md mx-auto p-7 transition-all duration-200" style={{ width: `${zoom * 4.6}px`, maxWidth: '100%', minWidth: 300 }}>
                  <div className="text-center">
                    <div className="text-[13px] font-extrabold text-gray-900">{TEMPLATES.find(t => t.id === tpl)?.name.toUpperCase() ?? 'ДОГОВОР ОКАЗАНИЯ УСЛУГ'}</div>
                    <div className="text-[10.5px] font-semibold text-gray-400 mt-1">№ 2024-{deal.id.toUpperCase()} · г. Москва · {deal.date}</div>
                  </div>
                  <div className="mt-4 space-y-2 text-[10px] leading-relaxed text-gray-600 font-medium">
                    <p><b>Заказчик:</b> ООО «{b.name}» (ИНН 7712345678, ОГРН 1157746123456)</p>
                    <p><b>Исполнитель:</b> {a.nick}, самозанятый (ИНН 771234567890)</p>
                    <p><b>1. Предмет договора.</b> Исполнитель создаёт и размещает рекламный материал формата «{deal.type}» — «{deal.title}».</p>
                    <p><b>2. Стоимость и порядок оплаты.</b> {deal.terms}. Оплата в течение 10 рабочих дней после подписания Акта.</p>
                    <p><b>3. Правки.</b> Заказчик вправе запросить до {edits} бесплатных правок в рамках ТЗ.{exclusive ? ' Исполнитель гарантирует эксклюзивность на 30 дней.' : ''}</p>
                    {eridOn && <p className="bg-yellow-50 border border-yellow-200 rounded px-2 py-1"><b>4. Маркировка рекламы.</b> Материал маркируется токеном ERID в соответствии со ст. 18.1 38-ФЗ «О рекламе». Данные передаются в ОРД.</p>}
                    {nda && <p><b>{eridOn ? '5' : '4'}. Конфиденциальность (NDA).</b> Стороны не разглашают условия договора третьим лицам.</p>}
                    <div className="pt-3 flex justify-between text-[9.5px] text-gray-400 font-bold">
                      <span>Заказчик: _________ /подпись/</span><span>Исполнитель: _________ /подпись/</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <aside className="h-fit rounded-xl border border-gray-200 p-4">
              <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-3">Чек-лист compliance</div>
              <div className="flex flex-col gap-2">
                {checklist.map((c, i) => (
                  <div key={i} className={`flex items-start gap-2 rounded-lg px-2.5 py-2 border ${c.ok ? 'border-emerald-100 bg-emerald-50/50' : c.warn ? 'border-amber-200 bg-amber-50/60' : 'border-red-100 bg-red-50/50'}`}>
                    {c.ok ? <Check size={13} className="text-emerald-500 shrink-0 mt-0.5" /> : c.warn ? <AlertTriangle size={13} className="text-amber-500 shrink-0 mt-0.5" /> : <AlertTriangle size={13} className="text-red-400 shrink-0 mt-0.5" />}
                    <div className="min-w-0 flex-1">
                      <div className="text-[11.5px] font-bold text-gray-700 leading-snug">{c.t}</div>
                      {c.warn && !nda && <Btn size="xs" variant="secondary" className="mt-1.5" onClick={() => { setNda(true); toast('ok', 'Пункт NDA добавлен в договор'); }}>Добавить NDA</Btn>}
                    </div>
                  </div>
                ))}
              </div>
              <div className={`mt-3 text-[11px] font-bold text-center rounded-lg py-2 ${allOk ? 'bg-emerald-50 text-emerald-600' : 'bg-amber-50 text-amber-600'}`}>
                {allOk ? '✓ Договор готов к отправке' : 'Есть пункты, требующие внимания'}
              </div>
            </aside>
          </div>
        )}

        {/* ШАГ 4 */}
        {step === 3 && (
          <div className="max-w-md mx-auto anim-in">
            <h3 className="text-[14px] font-bold text-gray-900 mb-3">Способ отправки на подпись</h3>
            <div className="flex flex-col gap-2.5">
              {[
                { id: 'email', icon: <Mail size={16} />, t: 'Email', d: 'PDF на почту автора с запросом подписи' },
                { id: 'tg', icon: <Send size={16} />, t: 'Telegram', d: 'Файл в чат сделки + напоминание через 24 ч' },
                { id: 'link', icon: <Link2 size={16} />, t: 'Ссылка для подписи', d: 'Веб-подпись (аналог DocuSign), срок действия 7 дней' },
              ].map(m => (
                <button key={m.id} onClick={() => setMethod(m.id)}
                  className={`flex items-center gap-3 rounded-xl border-2 px-4 py-3 text-left transition-all ${method === m.id ? 'border-indigo-500 bg-indigo-50/40' : 'border-gray-200 hover:border-gray-300'}`}>
                  <span className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${method === m.id ? 'bg-indigo-500 text-white' : 'bg-slate-100 text-gray-500'}`}>{m.icon}</span>
                  <span><span className="block text-[13px] font-bold text-gray-900">{m.t}</span><span className="block text-[11.5px] font-medium text-gray-400">{m.d}</span></span>
                  <span className={`ml-auto w-4.5 h-4.5 w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0 ${method === m.id ? 'border-indigo-500 bg-indigo-500' : 'border-gray-300'}`}>
                    {method === m.id && <Check size={11} className="text-white" />}
                  </span>
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2.5 mt-4 text-[12.5px] font-semibold text-gray-600 cursor-pointer">
              <input type="checkbox" checked={toTimeline} onChange={e => setToTimeline(e.target.checked)} className="w-4 h-4 accent-indigo-500" />
              Добавить в таймлайн сделки
            </label>
          </div>
        )}
      </div>

      <div className="px-5 py-4 border-t border-gray-100 flex items-center justify-between bg-slate-50/50 shrink-0">
        <Btn variant="ghost" onClick={() => (step === 0 ? onClose() : setStep(s => s - 1))}>
          <ChevronLeft size={14} />{step === 0 ? 'Отмена' : 'Назад'}
        </Btn>
        <div className="flex items-center gap-2">
          {step === 0 && <Btn disabled={!tpl} onClick={() => setStep(1)}>Далее<ChevronRight size={14} /></Btn>}
          {step === 1 && <Btn onClick={() => setStep(2)}>Сформировать договор<ChevronRight size={14} /></Btn>}
          {step === 2 && <Btn onClick={() => setStep(3)}>К отправке<ChevronRight size={14} /></Btn>}
          {step === 3 && <Btn onClick={() => { toast('ok', `Договор отправлен (${method === 'email' ? 'Email' : method === 'tg' ? 'Telegram' : 'ссылка для подписи'})`); onDone(); }}>
            <Send size={14} />Отправить
          </Btn>}
        </div>
      </div>
    </Modal>
  );
}
