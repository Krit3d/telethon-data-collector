import { useState } from 'react';
import { ArrowLeft, Briefcase, MessageSquare, Archive, ArrowDownRight, ArrowUpRight, StickyNote, FileText, TrendingUp, Target } from 'lucide-react';
import { AUTHORS, PUBLICATIONS, fmtNum, fmtMoney, authorById, type Deal } from '../data';
import { Badge, Avatar, Card, Btn, useToast } from '../components/ui';
import { SocialIcon, PostThumb } from '../components/icons';
import { LineChart, HBars } from '../components/charts';

export default function AuthorProfile({ authorId, deals, onBack, onNewDeal, onOpenDeal }: {
  authorId: string; deals: Deal[]; onBack: () => void; onNewDeal: (authorId: string) => void; onOpenDeal: (id: string) => void;
}) {
  const toast = useToast();
  const [tab, setTab] = useState('analytics');
  const a = authorById(authorId);
  const pubs = PUBLICATIONS.filter(p => p.authorId === authorId);
  const myDeals = deals.filter(d => d.authorId === authorId);

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="px-6 pt-4">
        <button onClick={onBack} className="inline-flex items-center gap-1.5 text-[12.5px] font-bold text-gray-400 hover:text-indigo-600 transition-colors mb-3">
          <ArrowLeft size={14} />К списку авторов
        </button>

        {/* Шапка профиля */}
        <Card className="p-5 flex items-center gap-5 flex-wrap">
          <Avatar nick={a.nick} hue={a.hue} size={80} />
          <div className="min-w-0">
            <div className="flex items-center gap-2.5">
              <h1 className="text-[24px] font-display font-semibold text-gray-900">{a.nick}</h1>
              <a href="#" onClick={e => { e.preventDefault(); toast('info', `Открываем профиль ${a.social}…`); }}
                className="inline-flex items-center gap-1.5 text-[12.5px] font-bold text-gray-500 hover:text-indigo-600 transition-colors">
                <SocialIcon social={a.social} size={15} />{a.social}
              </a>
            </div>
            <div className="flex items-center gap-4 mt-2 text-[13px] font-semibold text-gray-500 flex-wrap">
              <span>Подписчики: <b className="text-gray-900">{fmtNum(a.followers)}</b></span>
              <span>Ниша: <Badge tone="violet">{a.niche}</Badge></span>
              <span>Средний ER: <b className="text-gray-900">{a.er}%</b></span>
              <span>Сделок: <b className="text-gray-900">{a.deals}</b></span>
            </div>
          </div>
          <div className="flex items-center gap-2 ml-auto">
            <Btn onClick={() => onNewDeal(a.id)}><Briefcase size={14} />Новая сделка</Btn>
            <Btn variant="secondary" onClick={() => toast('ok', `Чат с ${a.nick} открыт`)}><MessageSquare size={14} />Написать</Btn>
            <Btn variant="danger" onClick={() => { toast('info', `${a.nick} архивирован`); onBack(); }}><Archive size={14} />Архивировать</Btn>
          </div>
        </Card>

        {/* Табы */}
        <div className="flex items-center gap-1 mt-4 border-b border-gray-200">
          {[['deals', `История сделок · ${myDeals.length}`], ['pubs', `Публикации · ${pubs.length}`], ['analytics', 'Аналитика'], ['docs', 'Документы']].map(([id, l]) => (
            <button key={id} onClick={() => setTab(id)}
              className={`px-3 py-2.5 text-[13px] font-semibold border-b-2 -mb-px transition-colors ${tab === id ? 'text-indigo-600 border-indigo-500' : 'text-gray-500 border-transparent hover:text-gray-800'}`}>{l}</button>
          ))}
        </div>
      </div>

      <div className="p-6 flex gap-5 items-start">
        {/* Левая колонка */}
        <div className="flex-1 min-w-0 anim-in" key={tab}>
          {tab === 'analytics' && (
            <div className="flex flex-col gap-4">
              <Card className="p-4">
                <h3 className="text-[13.5px] font-bold text-gray-900 mb-1">Динамика ER по публикациям</h3>
                <div className="text-[11px] font-semibold text-gray-400 mb-2">engagement rate, % · последние 8 публикаций</div>
                <LineChart unit="%" h={200} labels={pubs.length ? pubs.map(p => p.date.slice(0, 5)) : ['—']}
                  series={[{ name: 'ER', color: '#6366F1', data: pubs.length ? pubs.map(p => Math.round((p.likes + p.comments) / p.views * 1000) / 10) : [0], fill: true }]} />
              </Card>
              <Card className="p-4">
                <h3 className="text-[13.5px] font-bold text-gray-900 mb-1">CPM по сделкам</h3>
                <div className="text-[11px] font-semibold text-gray-400 mb-2">пунктир — средний рыночный CPM по нише «{a.niche}» (220 ₽)</div>
                <HBars money items={[
                  ...myDeals.map(d => ({ label: d.title.slice(0, 34), v: d.budget, color: '#6366F1' })),
                  { label: 'Средний рыночный CPM (референс)', v: 220 * 300, color: '#CBD5E1' },
                ]} />
              </Card>
              <Card className="overflow-hidden">
                <div className="px-4 py-3 border-b border-gray-100 text-[11px] font-bold text-gray-400 uppercase tracking-wide">Все публикации автора</div>
                <table className="w-full text-[12.5px]">
                  <thead><tr className="text-left text-[11px] font-bold text-gray-400 bg-slate-50/60 border-b border-gray-100">
                    {['Публикация', 'Тип', 'Дата', 'Просмотры', 'ER', 'ROAS'].map(h => <th key={h} className="px-4 py-2">{h}</th>)}
                  </tr></thead>
                  <tbody className="divide-y divide-gray-50">
                    {pubs.map(p => (
                      <tr key={p.id} className="hover:bg-slate-50 transition-colors">
                        <td className="px-4 py-2.5"><span className="flex items-center gap-2.5"><PostThumb hue={p.hue} size={32} /><span className="font-semibold text-gray-700 max-w-[260px] truncate">{p.text}</span></span></td>
                        <td className="px-4 py-2.5"><Badge tone={p.type === 'Stories' ? 'violet' : p.type === 'Reels' ? 'sky' : p.type === 'Видео' ? 'red' : 'indigo'}>{p.type}</Badge></td>
                        <td className="px-4 py-2.5 font-semibold text-gray-500">{p.date}</td>
                        <td className="px-4 py-2.5 font-bold text-gray-800 tabular-nums">{fmtNum(p.views)}</td>
                        <td className="px-4 py-2.5 font-bold text-emerald-600 tabular-nums">{((p.likes + p.comments) / p.views * 100).toFixed(1)}%</td>
                        <td className="px-4 py-2.5 font-bold text-gray-800 tabular-nums">{(p.revenue / p.cost).toFixed(1)}x</td>
                      </tr>
                    ))}
                    {pubs.length === 0 && <tr><td colSpan={6} className="px-4 py-10 text-center font-semibold text-gray-400">Публикаций пока нет</td></tr>}
                  </tbody>
                </table>
              </Card>
            </div>
          )}

          {tab === 'deals' && (
            <div className="flex flex-col gap-2.5">
              {myDeals.map(d => (
                <Card key={d.id} className="p-4 flex items-center gap-4 hover:border-indigo-200 transition-colors cursor-pointer" onClick={() => onOpenDeal(d.id)}>
                  <span className={`w-1 h-10 rounded-full shrink-0 ${d.stage === 5 ? 'bg-emerald-500' : d.stage === 4 ? 'bg-indigo-500' : d.stage === 2 ? 'bg-orange-400' : d.stage === 3 ? 'bg-amber-400' : 'bg-slate-300'}`} />
                  <div className="min-w-0 flex-1">
                    <div className="text-[13.5px] font-bold text-gray-900 truncate">{d.title}</div>
                    <div className="text-[11.5px] font-semibold text-gray-400 mt-0.5">{d.type} · публикация {d.pubDate} · создан {d.date}</div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-[14px] font-extrabold text-gray-900 tabular-nums">{fmtMoney(d.budget)}</div>
                    <div className="text-[11px] font-semibold text-gray-400">Этап {d.stage} из 5</div>
                  </div>
                </Card>
              ))}
              {myDeals.length === 0 && <Card className="p-10 text-center text-[13px] font-semibold text-gray-400">Сделок с этим автором пока нет</Card>}
            </div>
          )}

          {tab === 'pubs' && (
            <div className="grid grid-cols-3 gap-3">
              {pubs.map(p => (
                <Card key={p.id} className="overflow-hidden group">
                  <div className="h-32 relative" style={{ background: `linear-gradient(150deg, hsl(${p.hue} 72% 80%), hsl(${p.hue + 45} 62% 56%))` }}>
                    <div className="absolute inset-0" style={{ background: 'radial-gradient(circle at 30% 20%, rgba(255,255,255,.5), transparent 55%)' }} />
                    <Badge tone="dark" className="absolute top-2 left-2">{p.type}</Badge>
                  </div>
                  <div className="p-3">
                    <div className="text-[12.5px] font-bold text-gray-800 leading-snug line-clamp-2">{p.text}</div>
                    <div className="flex items-center gap-3 mt-2 text-[11px] font-bold text-gray-400">
                      <span>{fmtNum(p.views)} просм.</span><span className="text-emerald-600">ER {((p.likes + p.comments) / p.views * 100).toFixed(1)}%</span>
                      <span className="ml-auto">{p.date}</span>
                    </div>
                  </div>
                </Card>
              ))}
              {pubs.length === 0 && <Card className="p-10 text-center text-[13px] font-semibold text-gray-400 col-span-3">Публикаций пока нет</Card>}
            </div>
          )}

          {tab === 'docs' && (
            <Card className="divide-y divide-gray-100 overflow-hidden">
              {[...myDeals.filter(d => d.file).map(d => d.file!), 'Анкета_автора.pdf', 'Медиа_кит_2024.pdf'].map((f, i) => (
                <div key={i} className="flex items-center gap-3 px-4 py-3 hover:bg-slate-50 transition-colors">
                  <span className="w-9 h-9 rounded-lg bg-slate-100 text-gray-500 flex items-center justify-center shrink-0"><FileText size={16} /></span>
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] font-bold text-gray-800 truncate">{f}</div>
                    <div className="text-[11px] font-semibold text-gray-400">PDF · загружен в сентябре 2024</div>
                  </div>
                  <Btn variant="ghost" size="xs" onClick={() => toast('info', `Скачивание ${f}`)}>Скачать</Btn>
                </div>
              ))}
            </Card>
          )}
        </div>

        {/* Правая колонка */}
        <aside className="w-[300px] shrink-0 flex flex-col gap-4 sticky top-2">
          <Card className="p-5 text-center">
            <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide">Рейтинг автора</div>
            <div className="text-[48px] font-extrabold font-display text-emerald-500 leading-none mt-2">A−</div>
            <div className="text-[11.5px] font-semibold text-gray-400 mt-1.5">На основе {a.deals} сделок</div>
            <div className="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-gray-100">
              <Metric icon={<TrendingUp size={13} />} v={`${a.roas}x`} k="ROAS" />
              <Metric icon={<Target size={13} />} v={`${a.er}%`} k="Вовлеч." />
              <Metric icon={<TrendingUp size={13} />} v="95%" k="В срок" />
            </div>
          </Card>
          <Card className="p-4">
            <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-3">Сравнение с рынком</div>
            <div className="flex flex-col gap-3">
              <div>
                <div className="flex items-center justify-between text-[12px] mb-1">
                  <span className="font-semibold text-gray-500">CPM автора</span>
                  <span className="font-bold text-gray-900 tabular-nums">{a.cpm} ₽ <span className="text-gray-300">/ рынок 220 ₽</span></span>
                </div>
                <div className="h-2 rounded-full bg-gray-100 overflow-hidden relative">
                  <div className="absolute inset-y-0 left-0 rounded-full bg-indigo-500 bar-grow" style={{ width: `${(a.cpm / 350) * 100}%` }} />
                  <div className="absolute inset-y-0 w-[2px] bg-gray-400" style={{ left: `${(220 / 350) * 100}%` }} />
                </div>
                <div className="flex items-center gap-1 text-[11px] font-bold text-emerald-600 mt-1"><ArrowDownRight size={11} />Выгодно — на {Math.round((1 - a.cpm / 220) * 100)}% ниже рынка</div>
              </div>
              <div>
                <div className="flex items-center justify-between text-[12px] mb-1">
                  <span className="font-semibold text-gray-500">ER автора</span>
                  <span className="font-bold text-gray-900 tabular-nums">{a.er}% <span className="text-gray-300">/ рынок 3.5%</span></span>
                </div>
                <div className="h-2 rounded-full bg-gray-100 overflow-hidden relative">
                  <div className="absolute inset-y-0 left-0 rounded-full bg-emerald-500 bar-grow" style={{ width: `${(a.er / 8) * 100}%` }} />
                  <div className="absolute inset-y-0 w-[2px] bg-gray-400" style={{ left: `${(3.5 / 8) * 100}%` }} />
                </div>
                <div className="flex items-center gap-1 text-[11px] font-bold text-emerald-600 mt-1"><ArrowUpRight size={11} />Выше среднего</div>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wide mb-2.5 flex items-center gap-1.5"><StickyNote size={13} className="text-amber-500" />Заметки менеджера</div>
            <div className="flex flex-col gap-2">
              {[
                `Оптимальный бюджет: ${Math.round(a.cpm * 250 / 1000)}–${Math.round(a.cpm * 350 / 1000)}K ₽ для максимального ROAS`,
                'Лучший формат: ' + (a.social === 'YouTube' ? 'Видео' : a.social === 'TikTok' ? 'Короткие видео' : 'Reels'),
                'Лучшее время публикации: Вторник 19:00 (МСК)',
              ].map((n, i) => (
                <div key={i} className="rounded-lg bg-amber-50/60 border border-amber-100 px-3 py-2 text-[12px] font-semibold text-gray-700 leading-snug">{n}</div>
              ))}
            </div>
            <Btn variant="ghost" size="xs" className="mt-2" onClick={() => toast('ok', 'Заметка добавлена')}>+ Добавить заметку</Btn>
          </Card>
        </aside>
      </div>
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
