import { useId } from 'react';
import type { Social } from '../data';

/* Фирменные иконки соцсетей в стиле Lucide (тонкие, 24×24 viewBox) */
export function SocialIcon({ social, size = 14, className = '' }: { social: Social; size?: number; className?: string }) {
  const id = useId();
  const s = { width: size, height: size, className };
  switch (social) {
    case 'Instagram':
      return (
        <svg viewBox="0 0 24 24" {...s}>
          <defs>
            <linearGradient id={id} x1="0" y1="1" x2="1" y2="0">
              <stop offset="0" stopColor="#F9CE34" /><stop offset=".5" stopColor="#EE2A7B" /><stop offset="1" stopColor="#6228D7" />
            </linearGradient>
          </defs>
          <rect x="3" y="3" width="18" height="18" rx="5.5" fill="none" stroke={`url(#${id})`} strokeWidth="1.8" />
          <circle cx="12" cy="12" r="4.4" fill="none" stroke={`url(#${id})`} strokeWidth="1.8" />
          <circle cx="17.2" cy="6.8" r="1.25" fill="#6228D7" />
        </svg>
      );
    case 'VK':
      return (
        <svg viewBox="0 0 24 24" {...s}>
          <rect x="2.5" y="2.5" width="19" height="19" rx="5.5" fill="#0077FF" />
          <path d="M6.8 9h1.7c.2 2 .8 3.6 1.9 4.6V9h1.6v1.7c1-.1 2-1.1 2.4-1.7h-1.5V7.4h3.9c.4.6 1 1.5 1.5 2.6-.6 1.2-1.4 2.2-2.3 2.7.6.4 1.5 1.2 2 2.3h-2c-.5-.9-1.5-1.9-2.5-2.1V15h-.5c-2.4-.2-4.4-2.2-4.7-6z" fill="#fff" transform="scale(.92) translate(1 .5)" />
        </svg>
      );
    case 'Telegram':
      return (
        <svg viewBox="0 0 24 24" {...s}>
          <circle cx="12" cy="12" r="9.5" fill="#2AABEE" />
          <path d="M7 12.1l9.6-3.7c.5-.2.9.1.7.9l-1.6 7.6c-.1.6-.5.7-1 .4l-2.5-1.8-1.2 1.2c-.1.1-.3.3-.5.3l.2-2.4 4.4-4c.2-.2 0-.3-.3-.1L9.4 14l-2.3-.7c-.6-.2-.6-.8.9-1.2z" fill="#fff" />
        </svg>
      );
    case 'TikTok':
      return (
        <svg viewBox="0 0 24 24" {...s}>
          <path d="M14.5 3h2.2c.2 1.6 1.2 2.9 3.3 3.1v2.4c-1.3 0-2.4-.4-3.3-1v5.8c0 3.6-2.5 5.7-5.5 5.7-2.9 0-5.2-2.1-5.2-5 0-3 2.4-5.1 5.6-5v2.5c-.3-.1-.7-.2-1-.2-1.5 0-2.6 1.1-2.6 2.6 0 1.6 1.1 2.6 2.6 2.6 1.6 0 2.9-1.1 2.9-3.2V3z" fill="#111827" />
          <path d="M13.6 3v10.5c0 2.1-1.3 3.2-2.9 3.2-.5 0-1-.1-1.4-.4.6.6 1.4 1 2.4 1 1.6 0 2.9-1.1 2.9-3.2V3h-1z" fill="#25F4EE" opacity=".85" />
          <path d="M17.7 6.9v2.2c.7.3 1.5.4 2.3.4V7.3c-1 0-1.8-.2-2.3-.4z" fill="#FE2C55" opacity=".85" />
        </svg>
      );
    case 'YouTube':
      return (
        <svg viewBox="0 0 24 24" {...s}>
          <rect x="2.5" y="5.5" width="19" height="13.5" rx="4" fill="#FF0000" />
          <path d="M10.2 9.3l4.8 2.9-4.8 2.9z" fill="#fff" />
        </svg>
      );
    case 'Дзен':
      return (
        <svg viewBox="0 0 24 24" {...s}>
          <circle cx="12" cy="12" r="9.5" fill="#111827" />
          <path d="M12 5.5c.5 2.9 1.7 4.8 4.9 6.5-3.2 1.7-4.4 3.6-4.9 6.5-.5-2.9-1.7-4.8-4.9-6.5 3.2-1.7 4.4-3.6 4.9-6.5z" fill="#fff" />
        </svg>
      );
  }
}

export function socialLabel(s: Social) { return s; }

/* Детерминированный псевдо-QR */
export function FakeQR({ seed, size = 96, className = '' }: { seed: string; size?: number; className?: string }) {
  const n = 21;
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) { h ^= seed.charCodeAt(i); h = Math.imul(h, 16777619); }
  const rnd = () => { h ^= h << 13; h ^= h >>> 17; h ^= h << 5; return ((h >>> 0) / 4294967296); };
  const cells: boolean[] = [];
  for (let i = 0; i < n * n; i++) cells.push(rnd() > 0.52);
  const inFinder = (x: number, y: number) => (x < 8 && y < 8) || (x > n - 9 && y < 8) || (x < 8 && y > n - 9);
  const finder = (cx: number, cy: number, x: number, y: number) => {
    const dx = Math.abs(x - cx), dy = Math.abs(y - cy);
    return Math.max(dx, dy) <= 3 && !(Math.max(dx, dy) === 2) ? (Math.max(dx, dy) <= 1 || Math.max(dx, dy) === 3) : false;
  };
  const rects: { x: number; y: number }[] = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    if (inFinder(x, y)) {
      if (finder(3, 3, x, y) || finder(n - 4, 3, x, y) || finder(3, n - 4, x, y)) rects.push({ x, y });
    } else if (cells[y * n + x]) rects.push({ x, y });
  }
  return (
    <svg viewBox={`-1 -1 ${n + 2} ${n + 2}`} width={size} height={size} className={className} shapeRendering="crispEdges">
      <rect x="-1" y="-1" width={n + 2} height={n + 2} fill="#fff" />
      {rects.map((c, i) => <rect key={i} x={c.x} y={c.y} width="1" height="1" fill="#111827" />)}
    </svg>
  );
}

/* Миниатюра поста */
export function PostThumb({ hue, size = 48, label }: { hue: number; size?: number; label?: string }) {
  return (
    <div className="relative rounded-lg overflow-hidden shrink-0 shadow-sm" style={{ width: size, height: size, background: `linear-gradient(140deg, hsl(${hue} 75% 82%), hsl(${hue + 40} 65% 62%))` }}>
      <div className="absolute inset-0" style={{ background: `radial-gradient(circle at 30% 25%, rgba(255,255,255,.5), transparent 55%)` }} />
      <div className="absolute bottom-0 inset-x-0 h-1/3" style={{ background: 'linear-gradient(transparent, rgba(17,24,39,.28))' }} />
      {label && <span className="absolute bottom-0.5 left-1 text-[8px] font-bold text-white/95 uppercase tracking-wide">{label}</span>}
    </div>
  );
}
