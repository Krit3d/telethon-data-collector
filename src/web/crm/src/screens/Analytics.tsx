import { useMemo, useState } from 'react';
import { TrendingUp, Clock, FileCheck2, CalendarCheck2, Search, GitCompareArrows, StickyNote, AlertTriangle } from 'lucide-react';
import { AUTHORS, BRANDS, PUBLICATIONS, ANALYTICS_NOTES, fmtMoney, authorById, type Deal } from '../data';
import { Badge, Avatar, Card, Btn, useToast, Toggle, Stat } from '../components/ui';
import { SocialIcon, PostThumb } from '../components/icons';
import { LineChart, HBars, Donut, Funnel, Gauge, VBars, Scatter, RussiaMap } from '../components/charts';

const PERIODS = ['Last 30 days', 'Q3 2024', 'YTD'];
const NET_COLORS: Record<string, string> = { Instagram: '#E1306C', VK: '#0077FF', Telegram: '#2AABEE', TikTok: '#111827', YouTube: '#FF0000', 'Дзен': '#333' };

export default function Analytics({ deals }: { deals: Deal[] }) {
  const toast = useToast();
  const [view, setView] = useState<'overview' | 'extended'>('overview');
  const [period, setPeriod] = useState(PERIODS[1]);
  const [brands, setBrands] = useState<string[]>([]);
  const [compare, setCompare] = useState(false);
  const [authorQ, setAuthorQ] = useState('');
  const [nets, setNets] = useState<string[]>([]);
  const authorHits = useMemo(() => authorQ ? AUTHORS.filter(a => a.nick.includes(authorQ.toLowerCase())).slice(0, 4) : [], [authorQ]);

  const funnelDeals = [
    { label: 'Первичный контакт', v: 48 },
    { label: 'Переговоры', v: 31 },
    { label: 'Согласование ТЗ', v: 19 },
    { label: 'Подписание договора', v: 12 },
    { label: 'Контент создан', v: 8 },
  ];
  const months = ['Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен'];

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      {/* Sticky фильтры */}
      <div className="sticky top-0 z-20 px-6 pt-4 pb-3 bg-slate-50/90 backdrop-blur-md border-b border-gray-200/70">
        <div className="flex items-center gap-3 flex-wrap">
          <div>
            <h1 className="text-[22px] font-display font-semibold text-gray-900 leading-tight">Аналитика</h1>
            <p className="text-[12px] font-medium text-gray-400 mt-0.5">Сводка по кампаниям, авторам и compliance</p>
          </div>
          <div className="ml-auto flex items-center gap-2 flex-wrap">
            <div className="flex rounded-lg border border-gray-200 bg-white p-0.5">
              {PERIODS.map(p => (
                <button key={p} onClick={() => { setPeriod(p); toast('info', `Период: ${p}`); }}
                  className={`px-2.5 h-7.5 rounded-md text-[11.5px] font-bold transition-all ${period === p ? 'bg-indigo-500 text-white shadow-sm' : 'text-gray-500 hover:text-gray-800'}`}>{p}</button>
              ))}
            </div>
            {BRANDS.map(b => {
              const on = brands.includes(b.id);
              return (
                <button key={b.id} onClick={() => setBrands(x => on ? x.filter(v => v !== b.id) : [...x, b.id])}
                  className={`px-2.5 h-8.5 rounded-lg border text-[11.5px] font-bold transition-all active:scale-95 ${on ? 'bg-indigo-500 border-indigo-500 text-white' : 'bg-white border-gray-200 text-gray-500 hover:border-indigo-300'}`}>
                  {b.name}
                </button>
              );
            })}
            <label className="flex items-center gap-2 text-[11.5px] font-bold text-gray-500 cursor-pointer">
              <Toggle on={compare} onChange={setCompare} /><GitCompareArrows size={13} />Сравнить периоды
            </label>
          </div>
        </div>
        <div className="flex items-center gap-1.5 mt-2.5">
          <div className="flex rounded-lg border border-gray-200 bg-white p-0.5">
            {([['overview', 'Обзор'], ['extended', 'Расширенная']] as const).map(([id, l]) => (
              <button key={id} onClick={() => setView(id)}
                className={`px-3 h-7.5 rounded-md text-[11.5px] font-bold transition-all ${view === id ? 'bg-gray-900 text-white' : 'text-gray-500 hover:text-gray-800'}`}>{l}</button>
            ))}
          </div>
          {view === 'extended' && (
            <>
              <div className="relative">
                <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-300" />
                <input className="h-8.5 pl-7.5 pr-3 rounded-lg border border-gray-200 bg-white text-[12px] font-semibold outline-none focus:border-indigo-400 focus:ring-4 focus:ring-indigo-100 w-48"
                  placeholder="Автор…" value={authorQ} onChange={e => setAuthorQ(e.target.value)} />
                {authorHits.length > 0 && (
                  <div className="absolute top-9.5 left-0 w-56 bg-white rounded-xl border border-gray-200 shadow-xl py-1.5 z-30 pop-in">
                    {authorHits.map(a => (
                      <button key={a.id} onClick={() => { setAuthorQ(a.nick); toast('info', `Фильтр по автору: ${a.nick}`); }}
                        className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-slate-50 text-left">
                        <Avatar nick={a.nick} hue={a.hue} size={20} /><span className="text-[12px] font-bold text-gray-700">{a.nick}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              {(['Instagram', 'VK', 'Telegram', 'YouTube'] as const).map(n => {
                const on = nets.includes(n);
                return (
                  <button key={n} onClick={() => setNets(x => on ? x.filter(v => v !== n) : [...x, n])}
                    className={`flex items-center gap-1.5 px-2.5 h-8.5 rounded-full border text-[11.5px] font-bold transition-all ${on ? 'border-indigo-400 bg-indigo-50 text-indigo-700' : 'bg-white border-gray-200 text-gray-500 hover:border-gray-300'}`}>
                    <SocialIcon social={n} size={12} />{n}
                  </button>
                );
              })}
            </>
          )}
        </div>
      </div>

      {view === 'overview' ? (
        <div className="p-6 anim-in" key={period}>
          <div className="grid grid-cols-4 gap-4 mb-5 stagger">
            <Stat label="Выручка за период" value={fmtMoney(4_860_000)} sub="+18% к прошлому периоду" tone="green" icon={<TrendingUp size={15} />} />
            <Stat label="Активных сделок" value={deals.length} sub={`${deals.filter(d => d.stage === 5).length} завершено`} tone="indigo" icon={<FileCheck2 size={15} />} />
            <Stat label="Средний ROAS" value="2.4x" sub="цель ≥ 2.0x" tone="blue" icon={<TrendingUp size={15} />} />
            <Stat label="Публикаций" value="24" sub="6 ожидают синхронизации" tone="amber" icon={<CalendarCheck2 size={15} />} />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Card className="p-4">
              <CardHead title="Воронка сделок" sub="конверсия между этапами" />
              <Funnel steps={funnelDeals} />
            </Card>
            <Card className="p-4">
              <CardHead title="Динамика доходов" sub="2024 vs 2023, по месяцам" right={<Badge tone="green">+18% YoY</Badge>} />
              <LineChart money labels={months} series={[
                { name: '2024', color: '#6366F1', data: [2_100_000, 2_400_000, 2_850_000, 3_100_000, 4_200_000, 4_860_000], fill: true },
                ...(compare ? [{ name: '2023', color: '#CBD5E1', data: [1_800_000, 1_950_000, 2_300_000, 2_600_000, 3_400_000, 4_100_000], dash: true }] : []),
              ]} />
            </Card>
            <Card className="p-4">
              <CardHead title="Топ авторов" sub="по выручке за период" />
              <HBars money items={[
                { label: '@gadget_review', v: 2_410_000, color: NET_COLORS.YouTube, sub: <Avatar nick="@gadget_review" hue={200} size={20} /> },
                { label: '@tech_guru', v: 1_290_000, color: NET_COLORS.YouTube, sub: <Avatar nick="@tech_guru" hue={210} size={20} /> },
                { label: '@travel_diary', v: 720_000, color: NET_COLORS.Instagram, sub: <Avatar nick="@travel_diary" hue={175} size={20} /> },
                { label: '@beauty_blog', v: 486_000, color: NET_COLORS.Instagram, sub: <Avatar nick="@beauty_blog" hue={330} size={20} /> },
                { label: '@mama_blog', v: 205_000, color: NET_COLORS.Instagram, sub: <Avatar nick="@mama_blog" hue={45} size={20} /> },
              ]} />
            </Card>
            <div className="grid grid-rows-[auto_auto] gap-4">
              <Card className="p-4">
                <CardHead title="Распределение по соцсетям" sub="доля публикаций" />
                <Donut size={132} thickness={19} centerTop="24" centerBottom="публикации" parts={[
                  { label: 'Instagram', v: 45, color: '#E1306C' }, { label: 'VK', v: 25, color: '#0077FF' },
                  { label: 'Telegram', v: 15, color: '#2AABEE' }, { label: 'TikTok', v: 10, color: '#111827' },
                  { label: 'YouTube', v: 5, color: '#FF0000' },
                ]} />
              </Card>
              <Card className="p-4 flex items-center justify-around">
                <Gauge value={92} label="ERID-compliance" />
                <div className="flex flex-col gap-2.5 pr-2">
                  {[
                    ['Среднее время ответа', '24 мин', <Clock key="i" size={13} />],
                    ['Создание договора', '1.8 ч', <FileCheck2 key="i" size={13} />],
                    ['Выполнено в срок', '94%', <CalendarCheck2 key="i" size={13} />],
                  ].map(([k, v, ic], i) => (
                    <div key={i} className="flex items-center gap-2.5">
                      <span className="w-7 h-7 rounded-lg bg-slate-100 text-gray-500 flex items-center justify-center">{ic}</span>
                      <div>
                        <div className="text-[10.5px] font-bold text-gray-400 uppercase">{k as string}</div>
                        <div className="text-[14px] font-extrabold text-gray-900 font-display">{v as string}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-6 flex gap-5 items-start anim-in" key={period + 'ext'}>
          <div className="flex-1 min-w-0 grid grid-cols-3 gap-4">
            <Card className="p-4 col-span-2">
              <CardHead title="Динамика охвата" sub="по дням, сравнение периодов" />
              <LineChart labels={Array.from({ length: 14 }, (_, i) => `${i + 6}.09`)} series={[
                { name: 'Текущий', color: '#6366F1', data: [120, 145, 138, 172, 190, 181, 205, 232, 224, 248, 262, 255, 280, 301].map(v => v * 1000), fill: true },
                { name: 'Прошлый', color: '#CBD5E1', data: [98, 110, 125, 118, 140, 152, 149, 168, 175, 190, 186, 205, 214, 228].map(v => v * 1000), dash: true },
              ]} />
            </Card>
            <Card className="p-4">
              <CardHead title="Бюджет по соцсетям" sub="распределение, ₽" />
              <Donut size={120} thickness={17} centerTop="1.2M" centerBottom="бюджет" parts={[
                { label: 'Instagram', v: 480000, color: '#E1306C' }, { label: 'YouTube', v: 390000, color: '#FF0000' },
                { label: 'Telegram', v: 210000, color: '#2AABEE' }, { label: 'VK', v: 120000, color: '#0077FF' },
              ]} />
            </Card>
            <Card className="p-4">
              <CardHead title="Топ-5 публикаций по ROAS" />
              <HBars suffix="x" items={[...PUBLICATIONS].map(p => ({
                label: authorById(p.authorId).nick, v: Math.round(p.revenue / p.cost * 10) / 10,
                color: NET_COLORS[p.social], sub: <PostThumb hue={p.hue} size={20} />,
              })).sort((x, y) => y.v - x.v).slice(0, 5)} />
            </Card>
            <Card className="p-4">
              <CardHead title="Воронка: бюджет → выручка" sub="в денежном выражении" />
              <Funnel money steps={[
                { label: 'Бюджет', v: 1_200_000 }, { label: 'Охват', v: 4_200_000 }, { label: 'Клики', v: 62_000 },
                { label: 'Конверсии', v: 2_700 }, { label: 'Выручка', v: 4_860_000 },
              ]} />
            </Card>
            <Card className="p-4">
              <CardHead title="ERID Compliance" sub="проверено 24 из 26" right={<Badge tone="green">зелёная зона</Badge>} />
              <div className="flex items-center gap-4">
                <Gauge value={92} label="соответствие" />
                <div className="flex flex-col gap-1.5 flex-1">
                  <div className="flex items-center gap-2 text-[11.5px] font-bold text-red-500"><AlertTriangle size={12} />3 публикации без маркера</div>
                  <div className="text-[11px] font-medium text-gray-400 leading-snug">Задачи созданы: @mama_blog (03.09), @vegan_food (01.09), @crypto_talk (25.08)</div>
                </div>
              </div>
            </Card>
            <Card className="p-4 col-span-2">
              <CardHead title="Сравнение авторов" sub="X: CPM · Y: ER · размер: ROAS · цвет: соцсеть" />
              <Scatter xLabel="CPM" yLabel="ER"
                points={AUTHORS.slice(0, 8).map(a => ({ x: a.cpm, y: a.er, r: 8 + a.roas * 4, color: NET_COLORS[a.social] ?? '#6366F1', label: a.nick }))} />
            </Card>
            <Card className="p-4">
              <CardHead title="Доход по авторам" sub="стек по месяцам" />
              <VBars money items={months.map((m, i) => ({
                label: m, parts: [
                  { v: [820, 940, 1100, 1250, 1680, 1940][i] * 1000, color: '#6366F1' },
                  { v: [480, 520, 610, 700, 980, 1120][i] * 1000, color: '#A5B4FC' },
                  { v: [210, 260, 300, 360, 520, 640][i] * 1000, color: '#E0E7FF' },
                ],
              }))} h={170} />
            </Card>
            <Card className="p-4 col-span-2"><CardHead title="География аудитории" sub="тепловая карта России" /><RussiaMap /></Card>
            <Card className="p-4">
              <CardHead title="Форматы: конверсия" sub="средняя по нишам" />
              <HBars suffix="%" items={[
                { label: 'Stories', v: 6.9, color: '#E1306C' }, { label: 'Reels', v: 5.4, color: '#6366F1' },
                { label: 'Видео', v: 4.2, color: '#FF0000' }, { label: 'Пост', v: 3.0, color: '#0077FF' },
              ]} />
              <div className="mt-3 text-[11px] font-semibold text-gray-400">Stories конвертят в 2.3× лучше постов</div>
            </Card>
          </div>

          <aside className="w-[330px] shrink-0 sticky top-[130px]">
            <Card className="overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100 flex items-center gap-2 bg-amber-50/50">
                <StickyNote size={15} className="text-amber-500" />
                <span className="text-[12.5px] font-bold text-gray-800">Заметки и напоминания</span>
                <Btn size="xs" variant="ghost" className="ml-auto" onClick={() => toast('ok', 'Заметка создана')}>+ Новая</Btn>
              </div>
              <div className="p-3 flex flex-col gap-2.5 stagger">
                {ANALYTICS_NOTES.map((n, i) => (
                  <div key={i} className={`rounded-xl border px-3.5 py-3 hover:shadow-sm transition-shadow ${
                    n.tone === 'red' ? 'border-red-100 bg-red-50/50' : n.tone === 'green' ? 'border-emerald-100 bg-emerald-50/50' : n.tone === 'amber' ? 'border-amber-100 bg-amber-50/50' : 'border-indigo-100 bg-indigo-50/50'}`}>
                    <div className="text-[12.5px] font-semibold text-gray-700 leading-relaxed">{n.icon} {n.text}</div>
                    <div className="text-[10.5px] font-bold text-gray-400 mt-1.5">{n.by === 'система' || n.by === 'данные' ? n.by : `заметка менеджера · ${n.by}`}</div>
                  </div>
                ))}
              </div>
            </Card>
          </aside>
        </div>
      )}
    </div>
  );
}

function CardHead({ title, sub, right }: { title: string; sub?: string; right?: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between mb-3">
      <div>
        <h3 className="text-[13.5px] font-bold text-gray-900">{title}</h3>
        {sub && <div className="text-[11px] font-semibold text-gray-400 mt-0.5">{sub}</div>}
      </div>
      {right}
    </div>
  );
}
