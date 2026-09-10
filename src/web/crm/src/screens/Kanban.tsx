import { useMemo, useState } from 'react';
import { MessageCircle, FileText, Plus, X, ChevronRight, ChevronLeft, KanbanSquare, BookOpen } from 'lucide-react';
import { STAGES, BRANDS, fmtMoney, authorById, brandById, type Deal } from '../data';
import { Badge, Avatar, useToast, Btn, inputCls } from '../components/ui';
import { SocialIcon } from '../components/icons';

const typeTone: Record<Deal['type'], string> = { Stories: 'violet', Reels: 'sky', Пост: 'indigo', Видео: 'red' };

export default function Kanban({ deals, setDeals, onOpenDeal, onNewDeal, onOpenKB }: {
  deals: Deal[];
  setDeals: (updater: (prev: Deal[]) => Deal[]) => void;
  onOpenDeal: (id: string, tab?: string) => void;
  onNewDeal: () => void;
  onOpenKB: (brandId?: string) => void;
}) {
  const toast = useToast();
  const [fBrand, setFBrand] = useState('all');
  const [fType, setFType] = useState('all');
  const [dragId, setDragId] = useState<string | null>(null);
  const [overCol, setOverCol] = useState<number | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  const filtered = useMemo(() => deals.filter(d =>
    (fBrand === 'all' || d.brandId === fBrand) && (fType === 'all' || d.type === fType)
  ), [deals, fBrand, fType]);

  const total = filtered.reduce((a, d) => a + d.budget, 0);
  const ctx = filtered.find(d => d.stage === 2) ?? filtered[0];

  const drop = (stage: number) => {
    if (!dragId) return;
    const deal = deals.find(d => d.id === dragId);
    if (deal && deal.stage !== stage) {
      setDeals(prev => prev.map(d => d.id === dragId ? { ...d, stage } : d));
      toast('ok', `Сделка перемещена в «${STAGES.find(s => s.id === stage)?.name}»`);
    }
    setDragId(null); setOverCol(null);
  };

  return (
    <div className="h-full flex flex-col relative">
      <div className="px-6 pt-5 pb-3 bg-gradient-to-b from-white to-transparent shrink-0">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-[22px] font-display font-semibold text-gray-900">Воронка сделок</h1>
            <p className="text-[12.5px] font-medium text-gray-400 mt-0.5">
              {filtered.length} сделок на {fmtMoney(total)} · сентябрь 2024
            </p>
          </div>
          <div className="flex items-center gap-2">
            <select className={selCls} value="all" onChange={() => toast('info', 'Период: весь сентябрь')}>
              <option value="all">Период: весь</option><option>Сентябрь 2024</option><option>Август 2024</option>
            </select>
            <select className={selCls} value={fBrand} onChange={e => setFBrand(e.target.value)}>
              <option value="all">Все бренды</option>
              {BRANDS.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select>
            <select className={selCls} value={fType} onChange={e => setFType(e.target.value)}>
              <option value="all">Тип: любой</option>
              {(['Stories', 'Reels', 'Пост', 'Видео'] as const).map(t => <option key={t}>{t}</option>)}
            </select>
            <Btn onClick={onNewDeal}><Plus size={15} />Новая сделка</Btn>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 flex gap-3 px-4 pb-4">
        <div className="flex-1 min-w-0 overflow-x-auto scroll-thin">
          <div className="flex gap-3 h-full min-w-max">
            {STAGES.map(stage => {
              const cards = filtered.filter(d => d.stage === stage.id);
              return (
                <div key={stage.id}
                  onDragOver={e => { e.preventDefault(); setOverCol(stage.id); }}
                  onDragLeave={() => setOverCol(c => (c === stage.id ? null : c))}
                  onDrop={() => drop(stage.id)}
                  className={`w-[272px] shrink-0 rounded-xl border border-gray-200/70 bg-slate-50/80 flex flex-col transition-colors ${overCol === stage.id ? 'drop-target' : ''}`}>
                  <div className="px-3 pt-3 pb-2 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ background: stage.color }} />
                    <span className="text-[12.5px] font-bold text-gray-700">{stage.name}</span>
                    <span className="ml-auto text-[11px] font-extrabold text-gray-400 bg-white border border-gray-200 rounded-md px-1.5 py-0.5 tabular-nums">{cards.length}</span>
                  </div>
                  <div className="flex-1 min-h-0 overflow-y-auto scroll-thin px-2 pb-2 flex flex-col gap-2">
                    {cards.map(d => {
                      const a = authorById(d.authorId);
                      return (
                        <div key={d.id}
                          draggable
                          onDragStart={e => { setDragId(d.id); e.dataTransfer.effectAllowed = 'move'; }}
                          onDragEnd={() => { setDragId(null); setOverCol(null); }}
                          onClick={() => onOpenDeal(d.id)}
                          className={`group relative bg-white rounded-xl border border-gray-200/80 p-3 pl-3.5 cursor-grab active:cursor-grabbing shadow-[0_1px_2px_rgba(16,24,40,.05)] transition-all duration-150 hover:-translate-y-0.5 hover:shadow-[0_10px_24px_-10px_rgba(16,24,40,.25)] hover:border-indigo-200 ${dragId === d.id ? 'dragging-card' : ''}`}>
                          <span className="absolute left-0 top-3 bottom-3 w-[3px] rounded-full" style={{ background: stage.color }} />
                          <div className="flex items-center gap-2">
                            <Avatar nick={a.nick} hue={a.hue} size={26} />
                            <span className="text-[12px] font-bold text-gray-800 truncate">{a.nick}</span>
                            <SocialIcon social={a.social} size={13} className="shrink-0" />
                            <Badge tone={typeTone[d.type]} className="ml-auto shrink-0">{d.type}</Badge>
                          </div>
                          <div className="text-[13px] font-semibold text-gray-900 leading-snug mt-2 line-clamp-2">{d.title}</div>
                          <div className="text-[14px] font-extrabold text-gray-900 mt-1.5 tabular-nums">{fmtMoney(d.budget)}</div>
                          <div className="flex items-center gap-3 mt-2.5 pt-2 border-t border-gray-100 text-[11px] font-semibold text-gray-400">
                            <span className="inline-flex items-center gap-1"><MessageCircle size={12} />{d.msgs.length}</span>
                            {d.file && <span className="inline-flex items-center gap-1 truncate"><FileText size={12} />{d.file.length > 14 ? d.file.slice(0, 13) + '…' : d.file}</span>}
                            <span className="ml-auto shrink-0">{d.date}</span>
                          </div>
                          {d.erid && <Badge tone="green" className="absolute -top-2 right-2 shadow-sm">ERID ✓</Badge>}
                        </div>
                      );
                    })}
                    {cards.length === 0 && (
                      <div className="flex-1 flex flex-col items-center justify-center text-center px-4 py-6 text-gray-300">
                        <KanbanSquare size={22} className="mb-2" />
                        <div className="text-[11.5px] font-semibold">Перетащите сделку сюда</div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Правая панель «Контекст сделки» */}
        {!collapsed && ctx && (
          <aside className="w-[300px] shrink-0 rounded-xl border border-gray-200/80 bg-white p-4 flex flex-col gap-3 self-start anim-in">
            <div className="flex items-center justify-between">
              <h3 className="text-[13.5px] font-bold text-gray-900">Контекст сделки</h3>
              <button onClick={() => setCollapsed(true)} className="text-gray-300 hover:text-gray-500"><ChevronRight size={16} /></button>
            </div>
            {(() => { const b = brandById(ctx.brandId); const a = authorById(ctx.authorId); return (
              <>
                <div className="rounded-xl border border-gray-100 bg-slate-50/60 p-3">
                  <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-2">Бренд</div>
                  <div className="flex items-center gap-2.5">
                    <span className="w-9 h-9 rounded-lg flex items-center justify-center text-[13px] font-extrabold text-white shrink-0"
                      style={{ background: `linear-gradient(135deg, hsl(${b.hue} 70% 52%), hsl(${b.hue + 25} 65% 42%))` }}>{b.letter}</span>
                    <div className="min-w-0">
                      <div className="text-[13px] font-bold text-gray-900 truncate">{b.name}</div>
                      <div className="text-[11px] font-semibold text-gray-400">{b.category}</div>
                    </div>
                  </div>
                </div>
                <div className="rounded-xl border border-gray-100 bg-slate-50/60 p-3">
                  <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-2">Тип сотрудничества</div>
                  <div className="text-[12.5px] font-bold text-gray-800 leading-snug">{ctx.terms}</div>
                </div>
                <div className="rounded-xl border-2 border-indigo-200 bg-indigo-50/60 p-3 relative overflow-hidden">
                  <span className="absolute left-0 top-0 bottom-0 w-1 bg-indigo-500" />
                  <div className="flex items-center justify-between mb-2 pl-1.5">
                    <span className="text-[11px] font-extrabold text-indigo-700 uppercase tracking-wide inline-flex items-center gap-1.5"><MessageCircle size={12} />Коммуникации</span>
                    <span className="text-[10px] font-extrabold text-indigo-500 bg-white border border-indigo-200 rounded-md px-1.5 py-0.5 tabular-nums">{ctx.msgs.length} сообщ.</span>
                  </div>
                  <div className="flex items-center gap-2 pl-1.5">
                    <Avatar nick={a.nick} hue={a.hue} size={26} />
                    <span className="text-[12.5px] font-bold text-gray-800">{a.nick}</span>
                    <SocialIcon social={a.social} size={13} />
                  </div>
                  <div className="mt-2 pl-1.5 text-[11.5px] font-medium text-gray-500 italic truncate">
                    {ctx.msgs.length ? `«${ctx.msgs[ctx.msgs.length - 1].text}»` : 'Сообщений пока нет — напишите первым'}
                  </div>
                  <Btn size="sm" className="w-full mt-2.5" onClick={() => onOpenDeal(ctx.id, 'comms')}><MessageCircle size={13} />Написать автору</Btn>
                </div>
                <Btn variant="outline" onClick={() => onOpenKB(ctx.brandId)}><BookOpen size={14} />Открыть базу знаний бренда</Btn>
              </>
            ); })()}
          </aside>
        )}
        {collapsed && (
          <button onClick={() => setCollapsed(false)}
            className="self-start bg-white border border-gray-200 rounded-xl shadow-sm px-1 py-4 text-gray-400 hover:text-indigo-600 hover:border-indigo-200 transition-colors">
            <ChevronLeft size={16} />
          </button>
        )}
      </div>
      {overCol === null && dragId && <span className="hidden">{dragId}</span>}
    </div>
  );
}
const selCls = inputCls + ' !w-40 !h-8.5 text-[12.5px] font-semibold';
