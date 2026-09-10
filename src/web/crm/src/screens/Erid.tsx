import { useMemo, useState } from 'react';
import { Download, ShieldCheck, ShieldAlert, ExternalLink, X, Check, Eye } from 'lucide-react';
import { ERIDS, authorById, brandById, dealById, BRANDS, type EridRec, type Deal } from '../data';
import { Badge, Avatar, Card, Btn, SidePanel, useToast, inputCls, Tip, CopyBtn } from '../components/ui';
import { SocialIcon, FakeQR } from '../components/icons';

const statusTone: Record<EridRec['status'], string> = {
  'Активен': 'green', 'Ожидает публикации': 'amber', 'Опубликован': 'blue', 'Архивирован': 'gray',
};

export default function Erid({ deals, onOpenDeal }: { deals: Deal[]; onOpenDeal: (id: string) => void }) {
  const toast = useToast();
  const [recs, setRecs] = useState<EridRec[]>(ERIDS);
  const [fBrand, setFBrand] = useState('all');
  const [fStatus, setFStatus] = useState('all');
  const [sel, setSel] = useState<EridRec | null>(null);
  const [verdict, setVerdict] = useState<Record<string, 'ok' | 'bad'>>({});

  const list = useMemo(() => recs.filter(r =>
    (fBrand === 'all' || r.brandId === fBrand) && (fStatus === 'all' || r.status === fStatus)
  ), [recs, fBrand, fStatus]);

  const setVerdictFor = (id: string, v: 'ok' | 'bad') => {
    setVerdict(x => ({ ...x, [id]: v }));
    if (v === 'ok') {
      toast('ok', 'Корректность размещения подтверждена');
    } else {
      toast('err', 'Зафиксировано нарушение — создана задача менеджеру',
        () => { setVerdict(x => { const next = { ...x }; delete next[id]; return next; }); toast('info', 'Вердикт отменён'); });
    }
  };

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="px-6 pt-5 pb-4 bg-gradient-to-b from-white to-transparent">
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-[22px] font-display font-semibold text-gray-900">Реестр ERID-маркеров</h1>
            <p className="text-[12.5px] font-medium text-gray-400 mt-1">Compliance по 38-ФЗ «О рекламе» · {recs.length} маркеров · {recs.filter(r => r.status === 'Опубликован').length} опубликовано</p>
          </div>
          <div className="flex items-center gap-2">
            <select className={inputSel} value="all" onChange={() => toast('info', 'Период: весь')}>
              <option value="all">Период: весь</option><option>Сентябрь 2024</option><option>Август 2024</option>
            </select>
            <select className={inputSel} value={fBrand} onChange={e => setFBrand(e.target.value)}>
              <option value="all">Все бренды</option>{BRANDS.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select>
            <select className={inputSel} value={fStatus} onChange={e => setFStatus(e.target.value)}>
              <option value="all">Все статусы</option>
              {Object.keys(statusTone).map(s => <option key={s}>{s}</option>)}
            </select>
            <Btn variant="secondary" onClick={() => toast('ok', 'Реестр выгружен в erid_registry_19.09.xlsx')}><Download size={14} />Экспорт в Excel</Btn>
          </div>
        </div>
      </div>

      <div className="px-6 pb-6">
        <Card className="overflow-hidden">
          <div className="overflow-x-auto scroll-thin">
            <table className="w-full text-[12.5px] min-w-[880px]">
              <thead>
                <tr className="text-left text-[11px] font-bold text-gray-400 uppercase tracking-wide border-b border-gray-100 bg-slate-50/60">
                  {['ERID-код', 'Сделка', 'Автор', 'Бренд', 'Дата генерации', 'Статус размещения', ''].map((h, i) => <th key={i} className="px-4 py-2.5 whitespace-nowrap">{h}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {list.map(r => {
                  const a = authorById(r.authorId), b = brandById(r.brandId), d = dealById(deals, r.dealId);
                  return (
                    <tr key={r.id} onClick={() => setSel(r)} className="group cursor-pointer hover:bg-indigo-50/40 transition-colors">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <Tip label="Соответствует требованиям 38-ФЗ «О рекламе» (ст. 18.1) — данные переданы в ОРД">
                            <ShieldCheck size={14} className="text-emerald-500 shrink-0" />
                          </Tip>
                          <span className="font-mono text-[11.5px] font-bold text-gray-800">{r.code}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 max-w-[220px]"><span className="font-bold text-gray-700 truncate block group-hover:text-indigo-600 transition-colors">{d?.title ?? '—'}</span></td>
                      <td className="px-4 py-3"><span className="flex items-center gap-2"><Avatar nick={a.nick} hue={a.hue} size={24} /><span className="font-bold text-gray-700">{a.nick}</span><SocialIcon social={a.social} size={12} /></span></td>
                      <td className="px-4 py-3 font-semibold text-gray-500">{b.name}</td>
                      <td className="px-4 py-3 font-semibold text-gray-500 tabular-nums">{r.date}</td>
                      <td className="px-4 py-3"><Badge tone={statusTone[r.status]} dot>{r.status}</Badge></td>
                      <td className="px-4 py-3"><span className="text-gray-300 group-hover:text-indigo-500 transition-colors"><Eye size={15} /></span></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* Детальная карточка */}
      {sel && (
        <SidePanel onClose={() => setSel(null)} w="w-[560px]">
          <div className="px-5 py-4 border-b border-gray-100 flex items-start shrink-0">
            <div>
              <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide">ERID-маркер</div>
              <div className="font-mono text-[13.5px] font-bold text-gray-900 mt-1">{sel.code}</div>
              <div className="flex items-center gap-2 mt-2">
                <Badge tone={statusTone[sel.status]} dot>{sel.status}</Badge>
                <Tip label="Требование 38-ФЗ: маркер должен быть размещён в публикации"><Badge tone="green">38-ФЗ</Badge></Tip>
              </div>
            </div>
            <button onClick={() => setSel(null)} className="ml-auto text-gray-300 hover:text-gray-600"><X size={18} /></button>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto scroll-thin p-5 flex flex-col gap-4">
            <div className="flex gap-5">
              <div className="border border-gray-200 rounded-xl p-3 bg-white shadow-sm shrink-0"><FakeQR seed={sel.code} size={132} /></div>
              <div className="flex-1 flex flex-col gap-2.5 min-w-0">
                <div>
                  <div className="text-[11px] font-semibold text-gray-400">Сделка</div>
                  <button onClick={() => onOpenDeal(sel.dealId)} className="text-[13px] font-bold text-indigo-600 hover:underline text-left">{dealById(deals, sel.dealId)?.title}</button>
                </div>
                <div>
                  <div className="text-[11px] font-semibold text-gray-400">Автор / Бренд</div>
                  <div className="text-[13px] font-bold text-gray-800">{authorById(sel.authorId).nick} · {brandById(sel.brandId).name}</div>
                </div>
                {sel.link ? (
                  <a className="inline-flex items-center gap-1.5 text-[12.5px] font-bold text-indigo-600 hover:underline" href="#" onClick={e => { e.preventDefault(); toast('info', 'Открываем публикацию…'); }}>
                    <ExternalLink size={13} />{sel.link}
                  </a>
                ) : (
                  <div className="text-[12px] font-semibold text-amber-600 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2">Публикация ещё не вышла — ссылка появится после размещения</div>
                )}
                <CopyBtn text={sel.code} className="mt-auto" />
              </div>
            </div>

            {/* Скриншот поста */}
            <Card className="overflow-hidden">
              <div className="px-4 py-2.5 border-b border-gray-100 text-[11px] font-bold text-gray-400 uppercase tracking-wide">Скриншот размещения</div>
              <div className="p-4 bg-slate-50">
                {sel.link ? (
                  <div className="relative rounded-xl border-[2.5px] border-red-500 overflow-hidden bg-white shadow-md max-w-[300px] mx-auto">
                    <div className="h-44 relative" style={{ background: `linear-gradient(140deg, hsl(${authorById(sel.authorId).hue} 70% 82%), hsl(${authorById(sel.authorId).hue + 40} 60% 60%))` }}>
                      <div className="absolute top-2 left-2 flex items-center gap-1.5 bg-white/90 rounded-full px-2 py-0.5"><Avatar nick={authorById(sel.authorId).nick} hue={authorById(sel.authorId).hue} size={14} /><span className="text-[9px] font-bold text-gray-800">{authorById(sel.authorId).nick}</span></div>
                    </div>
                    <div className="px-3 py-2.5">
                      <div className="text-[10.5px] font-medium text-gray-700 leading-snug">Реклама. <mark className="bg-yellow-200 px-0.5 rounded text-[10px] font-mono font-bold">ERID: {sel.code.slice(-10)}</mark></div>
                    </div>
                    <span className="absolute -top-3 right-3 bg-red-500 text-white text-[9px] font-extrabold px-2 py-0.5 rounded-full">ERID найден</span>
                  </div>
                ) : (
                  <div className="py-8 text-center text-[12px] font-semibold text-gray-400">Скриншот будет добавлен после публикации</div>
                )}
              </div>
            </Card>

            {/* Ручной вердикт */}
            <Card className="p-4">
              <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-2.5">Ручной вердикт</div>
              <div className="flex gap-2">
                <Btn variant="secondary" className={verdict[sel.id] === 'ok' ? '!bg-emerald-500 !text-white !border-emerald-500 hover:!bg-emerald-600' : ''} onClick={() => setVerdictFor(sel.id, 'ok')}>
                  <Check size={14} />Подтвердить корректность
                </Btn>
                <Btn variant="danger" className={verdict[sel.id] === 'bad' ? '!bg-red-500 !text-white !border-red-500' : ''} onClick={() => setVerdictFor(sel.id, 'bad')}>
                  <ShieldAlert size={14} />Нарушение
                </Btn>
              </div>
              {verdict[sel.id] && <div className={`text-[11.5px] font-bold mt-2 ${verdict[sel.id] === 'ok' ? 'text-emerald-600' : 'text-red-500'}`}>{verdict[sel.id] === 'ok' ? '✓ Размещение подтверждено' : '✗ Нарушение зафиксировано, задача создана'}</div>}
            </Card>

            {/* История проверок */}
            <Card className="overflow-hidden">
              <div className="px-4 py-2.5 border-b border-gray-100 text-[11px] font-bold text-gray-400 uppercase tracking-wide">История проверок</div>
              <table className="w-full text-[12.5px]">
                <tbody className="divide-y divide-gray-50">
                  {sel.checks.map((c, i) => (
                    <tr key={i}>
                      <td className="px-4 py-2.5 font-bold text-gray-700 tabular-nums whitespace-nowrap">{c.date}</td>
                      <td className="px-2 py-2.5 font-medium text-gray-500">{c.result}</td>
                      <td className="px-4 py-2.5 text-right font-bold text-gray-700 whitespace-nowrap">{c.by}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        </SidePanel>
      )}
    </div>
  );
}
const inputSel = inputCls + ' !w-40 !h-8.5 text-[12.5px] font-semibold';
