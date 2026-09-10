import { useEffect, useRef, useState, type ReactNode } from 'react';

/* Все графики — собственный лёгкий SVG с hover-тултипами */

function useWidth(initial: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const cw = entries[0]?.contentRect.width;
      if (cw) setW(cw);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return { ref, w };
}

const fmt = (v: number) => v >= 1_000_000 ? (v / 1_000_000).toFixed(1).replace('.', ',') + 'M' : v >= 1000 ? Math.round(v / 1000) + 'K' : String(v);

export interface Series { name: string; color: string; data: number[]; dash?: boolean; fill?: boolean; }

export function LineChart({ labels, series, h = 190, unit = '', money }: { labels: string[]; series: Series[]; h?: number; unit?: string; money?: boolean }) {
  const { ref, w } = useWidth(560);
  const [idx, setIdx] = useState<number | null>(null);
  const W = Math.round(w), H = h, PL = 44, PR = 14, PT = 12, PB = 26;
  const n = labels.length;
  const all = series.flatMap(s => s.data);
  const max = Math.max(...all) * 1.15, min = 0;
  const X = (i: number) => PL + (i / (n - 1)) * (W - PL - PR);
  const Y = (v: number) => PT + (1 - (v - min) / (max - min)) * (H - PT - PB);
  const onMove = (e: React.MouseEvent) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    const x = (e.clientX - r.left) / r.width;
    setIdx(Math.min(n - 1, Math.max(0, Math.round(x * (n - 1)))));
  };
  const path = (s: Series) => s.data.map((v, i) => `${i ? 'L' : 'M'}${X(i)},${Y(v)}`).join(' ');
  const val = (v: number) => money ? Math.round(v).toLocaleString('ru-RU') + ' ₽' : fmt(v) + unit;
  return (
    <div className="relative" ref={ref} onMouseMove={onMove} onMouseLeave={() => setIdx(null)} style={{ height: h }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
        {[0, .25, .5, .75, 1].map(t => {
          const y = PT + t * (H - PT - PB);
          return <g key={t}>
            <line x1={PL} x2={W - PR} y1={y} y2={y} stroke="#EEF2F7" strokeWidth="1" />
            <text x={PL - 7} y={y + 3.5} textAnchor="end" fontSize="9.5" fill="#9CA3AF" fontFamily="Inter">{fmt(max - t * max)}</text>
          </g>;
        })}
        {labels.map((l, i) => (i % Math.ceil(n / 7) === 0 || i === n - 1) && (
          <text key={i} x={X(i)} y={H - 8} textAnchor="middle" fontSize="9.5" fill="#9CA3AF" fontFamily="Inter">{l}</text>
        ))}
        {series.map((s, si) => (
          <g key={si}>
            {s.fill && <path d={`${path(s)} L${X(n - 1)},${H - PB} L${X(0)},${H - PB} Z`} fill={s.color} opacity=".09" />}
            <path d={path(s)} fill="none" stroke={s.color} strokeWidth="2.2" strokeLinecap="round" strokeDasharray={s.dash ? '5 5' : undefined} className={s.dash ? '' : 'line-draw'} />
          </g>
        ))}
        {idx !== null && <line x1={X(idx)} x2={X(idx)} y1={PT} y2={H - PB} stroke="#C7D2FE" strokeWidth="1" />}
        {idx !== null && series.map((s, si) => (
          <circle key={si} cx={X(idx)} cy={Y(s.data[idx])} r="4" fill="#fff" stroke={s.color} strokeWidth="2.2" />
        ))}
      </svg>
      {idx !== null && (
        <div className="absolute z-20 pointer-events-none bg-gray-900 text-white rounded-lg px-3 py-2 text-[11px] shadow-xl"
          style={{ left: `${(X(idx) / W) * 100}%`, top: 0, transform: idx > n / 2 ? 'translateX(-108%)' : 'translateX(8%)' }}>
          <div className="font-semibold text-gray-300 mb-1">{labels[idx]}</div>
          {series.map((s, si) => (
            <div key={si} className="flex items-center gap-1.5 whitespace-nowrap">
              <span className="w-2 h-2 rounded-full" style={{ background: s.color }} />
              <span className="text-gray-300">{s.name}:</span><b>{val(s.data[idx])}</b>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function HBars({ items, h, money, suffix = '' }: { items: { label: string; v: number; color?: string; sub?: ReactNode }[]; h?: number; money?: boolean; suffix?: string }) {
  const max = Math.max(...items.map(i => i.v));
  return (
    <div className="flex flex-col gap-2.5" style={{ minHeight: h }}>
      {items.map((it, i) => (
        <div key={i} className="group">
          <div className="flex items-center justify-between text-[12px] mb-1">
            <span className="font-semibold text-gray-700 flex items-center gap-2 min-w-0 truncate">{it.sub}{it.label}</span>
            <span className="font-bold text-gray-900 tabular-nums shrink-0">{money ? Math.round(it.v).toLocaleString('ru-RU') + ' ₽' : it.v.toLocaleString('ru-RU') + suffix}</span>
          </div>
          <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
            <div className="h-full rounded-full bar-grow group-hover:opacity-80 transition-opacity" style={{ width: `${(it.v / max) * 100}%`, background: it.color ?? '#6366F1', animationDelay: `${i * 60}ms` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export function Donut({ parts, size = 150, thickness = 22, centerTop, centerBottom }: {
  parts: { label: string; v: number; color: string }[]; size?: number; thickness?: number; centerTop?: string; centerBottom?: string;
}) {
  const [hov, setHov] = useState<number | null>(null);
  const total = parts.reduce((a, p) => a + p.v, 0);
  const R = (size - thickness) / 2, C = 2 * Math.PI * R;
  let acc = 0;
  return (
    <div className="flex items-center gap-5">
      <div className="relative shrink-0" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={R} fill="none" stroke="#F1F5F9" strokeWidth={thickness} />
          {parts.map((p, i) => {
            const frac = p.v / total, dash = frac * C, off = -acc * C; acc += frac;
            return <circle key={i} cx={size / 2} cy={size / 2} r={R} fill="none" stroke={p.color}
              strokeWidth={hov === i ? thickness + 5 : thickness} strokeDasharray={`${dash - 1.5} ${C - dash + 1.5}`} strokeDashoffset={off}
              strokeLinecap="butt" className="transition-all duration-200 cursor-pointer" onMouseEnter={() => setHov(i)} onMouseLeave={() => setHov(null)} />;
          })}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <div className="text-[19px] font-extrabold font-display text-gray-900 leading-none">{hov !== null ? Math.round(parts[hov].v / total * 100) + '%' : centerTop}</div>
          <div className="text-[10.5px] font-semibold text-gray-400 mt-1 max-w-[70%] text-center leading-tight">{hov !== null ? parts[hov].label : centerBottom}</div>
        </div>
      </div>
      <div className="flex flex-col gap-1.5 min-w-0">
        {parts.map((p, i) => (
          <div key={i} className={`flex items-center gap-2 text-[12px] cursor-pointer transition-opacity ${hov !== null && hov !== i ? 'opacity-40' : ''}`}
            onMouseEnter={() => setHov(i)} onMouseLeave={() => setHov(null)}>
            <span className="w-2.5 h-2.5 rounded-[4px] shrink-0" style={{ background: p.color }} />
            <span className="text-gray-600 font-medium truncate">{p.label}</span>
            <b className="text-gray-900 ml-auto tabular-nums pl-2">{Math.round(p.v / total * 100)}%</b>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Funnel({ steps, money }: { steps: { label: string; v: number }[]; money?: boolean }) {
  const [hov, setHov] = useState<number | null>(null);
  const max = Math.max(...steps.map(s => s.v));
  return (
    <div className="flex flex-col items-center gap-1 py-1">
      {steps.map((s, i) => {
        const w = 26 + (s.v / max) * 74;
        const conv = i ? (s.v / steps[i - 1].v) * 100 : 100;
        return (
          <div key={i} className="w-full flex items-center gap-3 group cursor-default" onMouseEnter={() => setHov(i)} onMouseLeave={() => setHov(null)}>
            <div className="flex-1 flex justify-center">
              <div className="h-8 rounded-lg flex items-center justify-center text-[11.5px] font-bold text-white transition-all duration-200"
                style={{ width: `${w}%`, background: hov === i ? '#4F46E5' : '#6366F1', opacity: 1 - i * 0.13, boxShadow: hov === i ? '0 6px 16px -6px rgba(79,70,229,.55)' : 'none' }}>
                {money ? Math.round(s.v).toLocaleString('ru-RU') + ' ₽' : s.v.toLocaleString('ru-RU')}
              </div>
            </div>
            <div className="w-36 shrink-0">
              <div className="text-[11.5px] font-semibold text-gray-700 leading-tight">{s.label}</div>
              {i > 0 && <div className="text-[10.5px] font-semibold text-indigo-500">↓ {conv.toFixed(conv < 10 ? 1 : 0)}% конверсия</div>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function Gauge({ value, size = 168, label, zone = 'green' }: { value: number; size?: number; label?: string; zone?: 'green' | 'amber' | 'red' }) {
  const [anim, setAnim] = useState(0);
  const R = size / 2 - 16, C = Math.PI * R;
  const color = zone === 'green' ? '#10B981' : zone === 'amber' ? '#F59E0B' : '#EF4444';
  if (typeof window !== 'undefined' && anim === 0) requestAnimationFrame(() => setTimeout(() => setAnim(value), 60));
  return (
    <div className="relative flex flex-col items-center" style={{ width: size, height: size / 2 + 26 }}>
      <svg width={size} height={size / 2 + 16} viewBox={`0 0 ${size} ${size / 2 + 16}`}>
        <path d={`M 16 ${size / 2 + 8} A ${R} ${R} 0 0 1 ${size - 16} ${size / 2 + 8}`} fill="none" stroke="#EEF2F7" strokeWidth="15" strokeLinecap="round" />
        <path d={`M 16 ${size / 2 + 8} A ${R} ${R} 0 0 1 ${size - 16} ${size / 2 + 8}`} fill="none" stroke={color} strokeWidth="15" strokeLinecap="round"
          strokeDasharray={C} strokeDashoffset={C - (anim / 100) * C} style={{ transition: 'stroke-dashoffset 1s cubic-bezier(.22,.7,.3,1)' }} />
      </svg>
      <div className="absolute inset-x-0 bottom-3 text-center">
        <div className="text-[24px] font-extrabold font-display text-gray-900 leading-none">{value}%</div>
        {label && <div className="text-[10.5px] font-semibold text-gray-400 mt-1">{label}</div>}
      </div>
    </div>
  );
}

export function VBars({ items, h = 150, money }: { items: { label: string; parts: { v: number; color: string }[] }[]; h?: number; money?: boolean }) {
  const [hov, setHov] = useState<number | null>(null);
  const max = Math.max(...items.map(i => i.parts.reduce((a, p) => a + p.v, 0)));
  return (
    <div className="flex items-end gap-3 px-1" style={{ height: h }} onMouseLeave={() => setHov(null)}>
      {items.map((it, i) => {
        const total = it.parts.reduce((a, p) => a + p.v, 0);
        return (
          <div key={i} className="flex-1 flex flex-col items-center gap-1.5 h-full justify-end cursor-pointer" onMouseEnter={() => setHov(i)}>
            <div className={`text-[10px] font-bold tabular-nums transition-opacity ${hov === i ? 'opacity-100 text-gray-900' : 'opacity-0'}`}>
              {money ? Math.round(total).toLocaleString('ru-RU') + ' ₽' : fmt(total)}
            </div>
            <div className="w-full max-w-9 flex flex-col-reverse rounded-t-md overflow-hidden gap-px" style={{ height: `${(total / max) * 78}%`, transition: 'opacity .15s', opacity: hov !== null && hov !== i ? .45 : 1 }}>
              {it.parts.map((p, pi) => <div key={pi} style={{ background: p.color, flexGrow: p.v }} className="min-h-[3px]" />)}
            </div>
            <div className="text-[10px] font-semibold text-gray-400">{it.label}</div>
          </div>
        );
      })}
    </div>
  );
}

export function Scatter({ points, xLabel, yLabel }: {
  points: { x: number; y: number; r: number; color: string; label: string }[]; xLabel: string; yLabel: string;
}) {
  const [hov, setHov] = useState<number | null>(null);
  const { ref, w } = useWidth(560);
  const W = Math.round(w), H = 230, PL = 40, PB = 30, PT = 14, PR = 14;
  const maxX = Math.max(...points.map(p => p.x)) * 1.2, maxY = Math.max(...points.map(p => p.y)) * 1.2;
  const X = (v: number) => PL + (v / maxX) * (W - PL - PR);
  const Y = (v: number) => PT + (1 - v / maxY) * (H - PT - PB);
  return (
    <div className="relative" ref={ref} style={{ height: 230 }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
        {[0, .5, 1].map(t => <line key={t} x1={PL} x2={W - PR} y1={PT + t * (H - PT - PB)} y2={PT + t * (H - PT - PB)} stroke="#EEF2F7" />)}
        {[0, .5, 1].map(t => <text key={t} x={PL + t * (W - PL - PR)} y={H - 12} textAnchor="middle" fontSize="9.5" fill="#9CA3AF" fontFamily="Inter">{Math.round(t * maxX)} ₽</text>)}
        <text x={12} y={PT + 6} fontSize="9.5" fill="#9CA3AF" fontFamily="Inter">{yLabel} →</text>
        <text x={W - PR} y={H - 12} textAnchor="end" fontSize="9.5" fill="#9CA3AF" fontFamily="Inter">{xLabel}</text>
        {points.map((p, i) => (
          <g key={i} onMouseEnter={() => setHov(i)} onMouseLeave={() => setHov(null)} className="cursor-pointer">
            <circle cx={X(p.x)} cy={Y(p.y)} r={p.r} fill={p.color} opacity={hov === null || hov === i ? .32 : .12} />
            <circle cx={X(p.x)} cy={Y(p.y)} r={hov === i ? 5 : 3.5} fill={p.color} />
            {hov === i && <text x={X(p.x)} y={Y(p.y) - p.r - 6} textAnchor="middle" fontSize="10.5" fontWeight="700" fill="#111827" fontFamily="Inter">{p.label}</text>}
          </g>
        ))}
      </svg>
    </div>
  );
}

/* Стилизованная карта России */
const RU_PATH = 'M58 208 L88 152 L138 132 L198 96 L258 104 L318 64 L414 74 L516 44 L636 62 L756 42 L876 70 L946 122 L902 178 L942 238 L884 298 L806 330 L712 338 L648 388 L566 378 L506 332 L436 350 L382 318 L306 330 L244 300 L184 310 L138 278 L92 258 Z';
export const RU_CITIES = [
  { name: 'Москва', x: 208, y: 252, v: 38 },
  { name: 'Санкт-Петербург', x: 172, y: 200, v: 24 },
  { name: 'Екатеринбург', x: 392, y: 254, v: 12 },
  { name: 'Новосибирск', x: 560, y: 272, v: 9 },
  { name: 'Казань', x: 300, y: 246, v: 8 },
  { name: 'Краснодар', x: 196, y: 302, v: 6 },
  { name: 'Владивосток', x: 896, y: 306, v: 3 },
];
export function RussiaMap() {
  const [hov, setHov] = useState<number | null>(null);
  const max = 38;
  return (
    <div className="relative">
      <svg viewBox="0 0 1000 420" className="w-full" style={{ height: 210 }} preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="rug" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#EEF2FF" /><stop offset="1" stopColor="#E0E7FF" />
          </linearGradient>
        </defs>
        <path d={RU_PATH} fill="url(#rug)" stroke="#C7D2FE" strokeWidth="1.5" strokeLinejoin="round" />
        {RU_CITIES.map((c, i) => (
          <g key={i} onMouseEnter={() => setHov(i)} onMouseLeave={() => setHov(null)} className="cursor-pointer">
            <circle cx={c.x} cy={c.y} r={6 + (c.v / max) * 16} fill="#6366F1" opacity={hov === i ? .34 : .18} className="transition-all" />
            <circle cx={c.x} cy={c.y} r="3.5" fill={hov === i ? '#4338CA' : '#6366F1'} className="transition-all" />
            {hov === i && (
              <g>
                <rect x={c.x - 58} y={c.y - 44} width="116" height="30" rx="7" fill="#111827" />
                <text x={c.x} y={c.y - 31} textAnchor="middle" fontSize="11" fontWeight="700" fill="#fff" fontFamily="Inter">{c.name}</text>
                <text x={c.x} y={c.y - 19} textAnchor="middle" fontSize="9.5" fill="#A5B4FC" fontFamily="Inter">{c.v}% аудитории</text>
              </g>
            )}
          </g>
        ))}
      </svg>
      <div className="flex items-center justify-center gap-2 text-[10.5px] font-semibold text-gray-400 -mt-1">
        <span>Доля аудитории</span>
        <span className="w-20 h-1.5 rounded-full" style={{ background: 'linear-gradient(90deg,#E0E7FF,#6366F1)' }} />
      </div>
    </div>
  );
}
