import { useState } from 'react';
import { ExternalLink, RefreshCw, X, Eye, Heart, MessageSquare, Repeat2, Bookmark, MousePointerClick, ShoppingCart, AlertTriangle } from 'lucide-react';
import { PUBLICATIONS, BRANDS, DEALS_INIT, authorById, fmtNum, fmtMoney, type Publication, type Social } from '../data';
import { Badge, Avatar, Card, Btn, SidePanel, useToast, inputCls, Tip, CopyBtn, Stat } from '../components/ui';
import { SocialIcon, PostThumb, FakeQR } from '../components/icons';
import { LineChart, Donut, Gauge, Funnel, HBars } from '../components/charts';

const syncUi: Record<Publication['sync'], { label: string; cls: string }> = {
  syncing: { label: 'Синхронизация', cls: 'text-indigo-500' },
  waiting: { label: 'Ожидание', cls: 'text-gray-400' },
  ok: { label: 'Актуально', cls: 'text-emerald-600' },
  error: { label: 'Ошибка', cls: 'text-red-500' },
};
const erColor = (er: number) => er > 5 ? 'text-emerald-600 font-extrabold' : er >= 2 ? 'text-amber-600 font-extrabold' : 'text-red-500 font-extrabold';
const roasColor = (r: number) => r >= 2 ? 'text-emerald-600' : r >= 1 ? 'text-amber-600' : 'text-red-500';

export default function Publications({ onOpenDeal }: { onOpenDeal: (id: string) => void }) {
  const toast = useToast();
  const [pubs, setPubs] = useState(PUBLICATIONS);
  const [fBrand, setFBrand] = useState('all');
  const [fSocial, setFSocial] = useState('all');
  const [fSync, setFSync] = useState('all');
  const [sel, setSel] = useState<Publication | null>(null);
  const [refreshing, setRefreshing] = useState<string | null>(null);

  const list = pubs.filter(p =>
    (fBrand === 'all' || DEALS_INIT.some(d => d.authorId === p.authorId && d.brandId === fBrand)) &&
    (fSocial === 'all' || p.social === fSocial) &&
    (fSync === 'all' || p.sync === fSync)
  );

  const totalViews = list.reduce((a, p) => a + p.views, 0);
  const totalRevenue = list.reduce((a, p) => a + p.revenue, 0);
  const avgEr = list.reduce((a, p) => a + (p.likes + p.comments) / p.views, 0) / Math.max(list.length, 1) * 100;
  const avgRoas = list.reduce((a, p) => a + p.revenue / p.cost, 0) / Math.max(list.length, 1);

  const refresh = (p: Publication) => {
    setRefreshing(p.id);
    setTimeout(() => {
      setPubs(x => x.map(y => y.id === p.id ? { ...y, sync: 'ok' as const, views: Math.round(y.views * (1 + Math.random() * 0.06)) } : y));
      setRefreshing(null);
      toast('ok', 'Метрики обновлены только что');
    }, 1400);
  };

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="sticky top-0 z-20 px-6 pt-4 pb-3 bg-slate-50/90 backdrop-blur-md border-b border-gray-200/70">
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-[22px] font-display font-semibold text-gray-900">Публикации и метрики</h1>
            <p className="text-[12px] font-medium text-gray-400 mt-0.5">Трекинг результатов · синхронизация с AppsFlyer, Метрикой и GA4</p>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <select className={selCls} value={fBrand} onChange={e => setFBrand(e.target.value)}>
              <option value="all">Все бренды</option>{BRANDS.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select>
            <select className={selCls} value={fSocial} onChange={e => setFSocial(e.target.value)}>
              <option value="all">Все соцсети</option>
              {(['Instagram', 'VK', 'Telegram', 'TikTok', 'YouTube', 'Дзен'] as Social[]).map(s => <option key={s}>{s}</option>)}
            </select>
            <select className={selCls} value={fSync} onChange={e => setFSync(e.target.value)}>
              <option value="all">Синхронизация: любая</option>
              <option value="ok">Актуально</option><option value="syncing">Синхронизация</option>
              <option value="waiting">Ожидание</option><option value="error">Ошибка</option>
            </select>
            <Btn variant="secondary" onClick={() => toast('ok', 'Отчёт publications_q3.pdf сформирован')}>Экспорт отчёта</Btn>
          </div>
        </div>
        <div className="grid grid-cols-5 gap-3 mt-3">
          <Stat label="Всего публикаций" value={list.length} tone="indigo" />
          <Stat label="Общий охват" value={fmtNum(totalViews)} tone="blue" />
          <Stat label="Средний ER" value={avgEr.toFixed(1) + '%'} tone={avgEr >= 4 ? 'green' : 'amber'} />
          <Stat label="Общая выручка" value={fmtMoney(totalRevenue)} tone="green" />
          <Stat label="Средний ROAS" value={avgRoas.toFixed(1) + 'x'} tone={avgRoas >= 2 ? 'green' : 'amber'} />
        </div>
      </div>

      <div className="p-6 pt-4">
        <Card className="overflow-hidden">
          <div className="overflow-x-auto scroll-thin">
            <table className="w-full text-[12px] min-w-[1340px]">
              <thead>
                <tr className="text-left text-[10.5px] font-bold text-gray-400 uppercase tracking-wide border-b border-gray-100 bg-slate-50/60">
                  {['Публикация', 'Автор', '', 'Тип', 'Дата', 'ERID', 'Просмотры', 'Вовлечение', 'ER', 'Клики', 'Конв.', 'Выручка', 'ROAS', 'Синхр.', ''].map((h, i) => <th key={i} className="px-3 py-2.5 whitespace-nowrap">{h}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {list.map(p => {
                  const a = authorById(p.authorId);
                  const er = (p.likes + p.comments) / p.views * 100;
                  const roas = p.revenue / p.cost;
                  return (
                    <tr key={p.id} className="group hover:bg-indigo-50/40 transition-colors cursor-pointer" onClick={() => setSel(p)}>
                      <td className="px-3 py-2.5">
                        <span className="flex items-center gap-2.5">
                          <PostThumb hue={p.hue} size={44} />
                          <span className="min-w-0">
                            <span className="block font-bold text-gray-800 max-w-[210px] truncate group-hover:text-indigo-600 transition-colors">{p.text}</span>
                            <a href="#" onClick={e => { e.preventDefault(); e.stopPropagation(); toast('info', 'Открываем публикацию…'); }}
                              className="inline-flex items-center gap-1 text-[10.5px] font-bold text-indigo-500 hover:underline"><ExternalLink size={10} />Открыть</a>
                          </span>
                        </span>
                      </td>
                      <td className="px-3 py-2.5"><span className="flex items-center gap-1.5"><Avatar nick={a.nick} hue={a.hue} size={24} /><span className="font-bold text-gray-700 whitespace-nowrap">{a.nick}</span></span></td>
                      <td className="px-1 py-2.5"><SocialIcon social={p.social} size={15} /></td>
                      <td className="px-3 py-2.5"><Badge tone={p.type === 'Stories' ? 'violet' : p.type === 'Reels' ? 'sky' : p.type === 'Видео' ? 'red' : 'indigo'}>{p.type}</Badge></td>
                      <td className="px-3 py-2.5 font-semibold text-gray-500 whitespace-nowrap tabular-nums">{p.date}</td>
                      <td className="px-3 py-2.5">
                        {p.eridOk
                          ? <Tip label="Маркер проверен вручную · соответствует 38-ФЗ"><Badge tone="green">✓ Проверен</Badge></Tip>
                          : <Tip label="Нарушение 38-ФЗ — создана задача менеджеру"><Badge tone="red">✗ Отсутствует</Badge></Tip>}
                      </td>
                      <td className="px-3 py-2.5 font-bold text-gray-900 tabular-nums whitespace-nowrap">{fmtNum(p.views)}</td>
                      <td className="px-3 py-2.5 whitespace-nowrap">
                        <span className="inline-flex items-center gap-2.5 font-semibold text-gray-500 tabular-nums">
                          <span className="inline-flex items-center gap-1"><Heart size={11} className="text-rose-400" />{fmtNum(p.likes)}</span>
                          <span className="inline-flex items-center gap-1"><MessageSquare size={11} className="text-sky-400" />{fmtNum(p.comments)}</span>
                          <span className="inline-flex items-center gap-1"><Repeat2 size={11} className="text-indigo-400" />{fmtNum(p.reposts)}</span>
                          <span className="inline-flex items-center gap-1"><Bookmark size={11} className="text-amber-400" />{fmtNum(p.saves)}</span>
                        </span>
                      </td>
                      <td className={`px-3 py-2.5 tabular-nums whitespace-nowrap ${erColor(er)}`}>{er.toFixed(1)}%</td>
                      <td className="px-3 py-2.5 font-semibold text-gray-600 tabular-nums whitespace-nowrap"><MousePointerClick size={10} className="inline mr-1 text-gray-300" />{fmtNum(p.clicks)}</td>
                      <td className="px-3 py-2.5 font-semibold text-gray-600 tabular-nums whitespace-nowrap"><ShoppingCart size={10} className="inline mr-1 text-gray-300" />{p.conv}</td>
                      <td className="px-3 py-2.5 font-bold text-gray-900 tabular-nums whitespace-nowrap">{fmtMoney(p.revenue)}</td>
                      <td className={`px-3 py-2.5 font-extrabold tabular-nums whitespace-nowrap ${roasColor(roas)}`}>{roas.toFixed(1)}x</td>
                      <td className="px-3 py-2.5 whitespace-nowrap" onClick={e => e.stopPropagation()}>
                        {p.sync === 'syncing'
                          ? <span className={`inline-flex items-center gap-1.5 font-bold ${syncUi[p.sync].cls}`}><RefreshCw size={12} className="animate-spin" />{syncUi[p.sync].label}</span>
                          : p.sync === 'error'
                            ? <Tip label="Токен GA4 истёк — исправьте подключение в Источниках"><span className={`inline-flex items-center gap-1 font-bold ${syncUi[p.sync].cls} cursor-help`}>✗ {syncUi[p.sync].label}</span></Tip>
                            : <span className={`inline-flex items-center gap-1 font-bold ${syncUi[p.sync].cls}`}>{p.sync === 'ok' ? '✓' : '⏱'} {syncUi[p.sync].label}</span>}
                      </td>
                      <td className="px-3 py-2.5 whitespace-nowrap" onClick={e => e.stopPropagation()}>
                        <span className="flex items-center gap-1">
                          <Tip label="Обновить метрики">
                            <button onClick={() => refresh(p)} disabled={refreshing === p.id}
                              className="w-7 h-7 rounded-lg text-gray-400 hover:bg-indigo-50 hover:text-indigo-600 flex items-center justify-center transition-colors disabled:opacity-50">
                              <RefreshCw size={13} className={refreshing === p.id ? 'animate-spin' : ''} />
                            </button>
                          </Tip>
                          <button onClick={() => setSel(p)} className="text-[11px] font-bold text-indigo-600 hover:text-indigo-700 px-1">Подробнее</button>
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {sel && <PublicationDetail sel={sel} onClose={() => setSel(null)} />}
    </div>
  );
}

function PublicationDetail({ sel, onClose }: { sel: Publication; onClose: () => void }) {
  const toast = useToast();
  const [mtab, setMtab] = useState<'eng' | 'conv' | 'cmp'>('eng');
  const a = authorById(sel.authorId);
  const er = (sel.likes + sel.comments) / sel.views * 100;
  const days = ['1д', '2д', '3д', '4д', '5д', '6д', '7д'];

  return (
    <SidePanel onClose={onClose} w="w-[720px]">
      {/* Шапка */}
      <div className="px-5 py-4 border-b border-gray-100 shrink-0 flex items-start gap-4">
        <div className="relative shrink-0 rounded-xl overflow-hidden shadow-lg" style={{ width: 132, height: 176, background: `linear-gradient(150deg, hsl(${sel.hue} 75% 80%), hsl(${sel.hue + 45} 65% 55%))` }}>
          <div className="absolute inset-0" style={{ background: 'radial-gradient(circle at 30% 20%, rgba(255,255,255,.55), transparent 55%)' }} />
          <div className="absolute bottom-0 inset-x-0 p-2.5 bg-gradient-to-t from-gray-900/60 to-transparent">
            <div className="flex items-center gap-1.5"><Avatar nick={a.nick} hue={a.hue} size={18} ring /><span className="text-[9.5px] font-bold text-white">{a.nick}</span></div>
          </div>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge tone="dark"><span className="inline-flex items-center gap-1"><SocialIcon social={sel.social} size={11} />{sel.social}</span></Badge>
            <Badge tone="violet">{sel.type}</Badge>
            <Badge tone="gray">{sel.date}</Badge>
          </div>
          <h2 className="text-[15px] font-bold text-gray-900 leading-snug mt-2">{sel.text}</h2>
          <div className="flex items-center gap-4 mt-2.5 text-[12px] font-bold text-gray-500">
            <span className="inline-flex items-center gap-1"><Eye size={13} />{fmtNum(sel.views)}</span>
            <span className="inline-flex items-center gap-1"><Heart size={13} className="text-rose-400" />{fmtNum(sel.likes)}</span>
            <span className="inline-flex items-center gap-1"><MessageSquare size={13} className="text-sky-400" />{fmtNum(sel.comments)}</span>
            <span className={erColor(er)}>ER {er.toFixed(1)}%</span>
          </div>
          <Btn variant="outline" size="sm" className="mt-3" onClick={() => toast('info', 'Открываем оригинал…')}><ExternalLink size={13} />Открыть оригинал</Btn>
        </div>
        <button onClick={onClose} className="text-gray-300 hover:text-gray-600 shrink-0"><X size={18} /></button>
      </div>

      {/* Табы метрик */}
      <div className="px-5 py-2.5 border-b border-gray-100 flex gap-1 shrink-0">
        {([['eng', 'Вовлечённость'], ['conv', 'Конверсии'], ['cmp', 'Сравнение']] as const).map(([id, l]) => (
          <button key={id} onClick={() => setMtab(id)} className={`px-3 py-1.5 rounded-lg text-[12.5px] font-bold transition-colors ${mtab === id ? 'bg-indigo-50 text-indigo-700' : 'text-gray-500 hover:bg-slate-50'}`}>{l}</button>
        ))}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto scroll-thin p-5 flex flex-col gap-4">
        {/* ERID compliance */}
        <Card className={`p-4 ${sel.eridOk ? '' : 'border-red-200'}`}>
          <div className="flex items-center justify-between mb-2.5">
            <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide">ERID Compliance</h3>
            {sel.eridOk ? <Badge tone="green" dot>Маркер найден в тексте</Badge> : <Badge tone="red" dot>Маркер отсутствует!</Badge>}
          </div>
          <div className="rounded-lg bg-slate-50 border border-gray-100 px-3.5 py-3 text-[12.5px] font-medium text-gray-700 leading-relaxed">
            {sel.text}. {sel.eridOk
              ? <>Реклама. <mark className="bg-yellow-200 px-1 rounded font-mono text-[11px] font-bold">ERID: 2Vtzqx{sel.id.toUpperCase()}Kp</mark> · {authorById(sel.authorId).nick}</>
              : <span className="text-red-500 font-bold">Токен ERID в тексте публикации не обнаружен.</span>}
          </div>
          {(sel.type === 'Stories' || sel.type === 'Reels') && (
            <div className="flex items-center gap-2 mt-2.5 text-[11.5px] font-bold text-amber-600">
              <AlertTriangle size={13} />ERID должен быть в первых 3 кадрах · {sel.eridOk ? 'проверено ✓' : 'требуется проверка'}
            </div>
          )}
          {!sel.eridOk && <Btn variant="danger" size="sm" className="mt-3" onClick={() => toast('ok', `Задача создана: ${a.nick} — добавить ERID-маркер`)}>Создать задачу автору</Btn>}
        </Card>

        {/* Метрики */}
        {mtab === 'eng' && (
          <div className="anim-in">
            <Card className="p-4 mb-4">
              <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-2">Динамика за 7 дней</h3>
              <LineChart h={180} labels={days} series={[
                { name: 'Просмотры', color: '#6366F1', data: [0.32, 0.55, 0.68, 0.79, 0.88, 0.95, 1].map(k => Math.round(sel.views * k)), fill: true },
                { name: 'Лайки', color: '#F43F5E', data: [0.3, 0.52, 0.65, 0.77, 0.87, 0.94, 1].map(k => Math.round(sel.likes * k)) },
                { name: 'Комментарии', color: '#0EA5E9', data: [0.25, 0.48, 0.63, 0.75, 0.85, 0.93, 1].map(k => Math.round(sel.comments * k)) },
              ]} />
            </Card>
            <div className="grid grid-cols-2 gap-4">
              <Card className="p-4">
                <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-2">Распределение вовлечённости</h3>
                <Donut size={116} thickness={16} centerTop={fmtNum(sel.likes + sel.comments + sel.reposts + sel.saves)} centerBottom="действий" parts={[
                  { label: 'Лайки', v: sel.likes, color: '#F43F5E' }, { label: 'Комментарии', v: sel.comments, color: '#0EA5E9' },
                  { label: 'Репосты', v: sel.reposts, color: '#6366F1' }, { label: 'Сохранения', v: sel.saves, color: '#F59E0B' },
                ]} />
              </Card>
              <Card className="p-4 flex flex-col items-center justify-center">
                <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-1 self-start">Engagement Rate</h3>
                <Gauge value={Math.round(er * 10) / 10} label={er > 5 ? 'выше среднего по нише' : 'в пределах нормы'} zone={er > 5 ? 'green' : er >= 2 ? 'green' : 'amber'} />
              </Card>
            </div>
          </div>
        )}
        {mtab === 'conv' && (
          <Card className="p-4 anim-in">
            <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-3">Воронка конверсий</h3>
            <Funnel steps={[
              { label: 'Просмотры', v: sel.views },
              { label: 'Клики (UTM)', v: sel.clicks },
              { label: 'Добавления в корзину', v: Math.round(sel.clicks * 0.22) },
              { label: 'Покупки', v: sel.conv },
            ]} />
            <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t border-gray-100 text-center">
              <div><div className="text-[16px] font-extrabold font-display text-gray-900">{fmtMoney(sel.revenue)}</div><div className="text-[10.5px] font-bold text-gray-400 uppercase">Выручка</div></div>
              <div><div className="text-[16px] font-extrabold font-display text-gray-900">{fmtMoney(sel.cost)}</div><div className="text-[10.5px] font-bold text-gray-400 uppercase">Стоимость</div></div>
              <div><div className={`text-[16px] font-extrabold font-display ${roasColor(sel.revenue / sel.cost)}`}>{(sel.revenue / sel.cost).toFixed(1)}x</div><div className="text-[10.5px] font-bold text-gray-400 uppercase">ROAS</div></div>
            </div>
          </Card>
        )}
        {mtab === 'cmp' && (
          <Card className="p-4 anim-in">
            <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-3">ER в сравнении, %</h3>
            <HBars suffix="%" items={[
              { label: 'Эта публикация', v: Math.round(er * 10) / 10, color: '#6366F1' },
              { label: `Средний по ${a.nick}`, v: a.er, color: '#A5B4FC' },
              { label: 'Средний по бренду', v: 3.4, color: '#C7D2FE' },
              { label: 'Средний по нише', v: 3.1, color: '#E0E7FF' },
            ]} />
          </Card>
        )}

        {/* UTM */}
        <Card className="p-4">
          <h3 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-2.5">UTM и трекинг</h3>
          <div className="rounded-lg bg-slate-50 border border-gray-100 px-3.5 py-2.5 font-mono text-[11px] font-semibold text-gray-700 break-all">
            https://{sel.link}?utm_source={sel.social.toLowerCase()}&utm_medium=influencer&utm_campaign=cf_q3&utm_content={sel.id}
          </div>
          <div className="flex items-center justify-between mt-2">
            <CopyBtn text={`https://${sel.link}?utm_source=${sel.social.toLowerCase()}&utm_medium=influencer&utm_campaign=cf_q3&utm_content=${sel.id}`} />
            <span className="text-[10.5px] font-bold text-gray-300">обновлено {sel.date}</span>
          </div>
          <div className="flex items-center gap-4 mt-4 pt-4 border-t border-gray-100">
            <div className="border border-gray-200 rounded-xl p-1.5 bg-white shrink-0"><FakeQR seed={sel.link} size={80} /></div>
            <div className="grid grid-cols-3 gap-4 flex-1">
              <div><div className="text-[15px] font-extrabold text-gray-900 font-display tabular-nums">{fmtNum(sel.clicks)}</div><div className="text-[10.5px] font-bold text-gray-400 uppercase">Клики</div></div>
              <div><div className="text-[15px] font-extrabold text-gray-900 font-display tabular-nums">{fmtNum(Math.round(sel.clicks * 0.82))}</div><div className="text-[10.5px] font-bold text-gray-400 uppercase">Уник. польз.</div></div>
              <div><div className="text-[15px] font-extrabold text-gray-900 font-display tabular-nums">38%</div><div className="text-[10.5px] font-bold text-gray-400 uppercase">Отказы</div></div>
            </div>
          </div>
        </Card>
      </div>
    </SidePanel>
  );
}
const selCls = inputCls + ' !w-44 !h-8.5 text-[12.5px] font-semibold';
