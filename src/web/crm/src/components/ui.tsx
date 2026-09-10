import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import { CheckCircle2, AlertTriangle, Info, X, Copy, Check } from 'lucide-react';

/* ================= Toasts ================= */
interface Toast { id: number; type: 'ok' | 'err' | 'info'; msg: string; retry?: () => void; }
const ToastCtx = createContext<(type: Toast['type'], msg: string, retry?: () => void) => void>(() => {});
export const useToast = () => useContext(ToastCtx);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((type: Toast['type'], msg: string, retry?: () => void) => {
    const id = Date.now() + Math.random();
    setToasts(t => [...t, { id, type, msg, retry }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="fixed top-4 right-4 z-[120] flex flex-col gap-2 w-[340px]">
        {toasts.map(t => (
          <div key={t.id} className={`toast-in flex items-start gap-2.5 rounded-xl border bg-white px-3.5 py-3 shadow-lg shadow-slate-900/8 ${
            t.type === 'ok' ? 'border-emerald-200' : t.type === 'err' ? 'border-red-200' : 'border-blue-200'}`}>
            {t.type === 'ok' && <CheckCircle2 size={17} className="text-emerald-500 shrink-0 mt-0.5" />}
            {t.type === 'err' && <AlertTriangle size={17} className="text-red-500 shrink-0 mt-0.5" />}
            {t.type === 'info' && <Info size={17} className="text-blue-500 shrink-0 mt-0.5" />}
            <div className="text-[13px] font-medium text-gray-800 leading-snug flex-1">{t.msg}</div>
            {t.type === 'err' && t.retry && (
              <button onClick={() => { t.retry?.(); setToasts(x => x.filter(y => y.id !== t.id)); }}
                className="text-[12px] font-semibold text-red-600 hover:text-red-700 shrink-0">Повторить</button>
            )}
            <button onClick={() => setToasts(x => x.filter(y => y.id !== t.id))} className="text-gray-300 hover:text-gray-500 shrink-0">
              <X size={14} />
            </button>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

/* ================= Buttons ================= */
export function Btn({ variant = 'primary', size = 'md', className = '', children, onClick, disabled, type = 'button' }: {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline';
  size?: 'sm' | 'md' | 'xs'; className?: string; children: ReactNode;
  onClick?: () => void; disabled?: boolean; type?: 'button' | 'submit';
}) {
  const base = 'inline-flex items-center justify-center gap-1.5 font-semibold rounded-lg transition-all duration-150 active:scale-[.97] disabled:opacity-45 disabled:pointer-events-none whitespace-nowrap';
  const sizes = { xs: 'text-[12px] px-2.5 h-7', sm: 'text-[13px] px-3 h-8.5', md: 'text-[13.5px] px-4 h-9.5' };
  const variants = {
    primary: 'bg-indigo-500 text-white hover:bg-indigo-600 shadow-sm shadow-indigo-500/25',
    secondary: 'bg-white text-gray-700 border border-gray-200 hover:border-gray-300 hover:bg-gray-50 shadow-sm shadow-slate-900/4',
    outline: 'bg-transparent text-indigo-600 border border-indigo-200 hover:bg-indigo-50',
    ghost: 'bg-transparent text-gray-600 hover:bg-gray-100',
    danger: 'bg-red-50 text-red-600 border border-red-200 hover:bg-red-100',
  };
  return (
    <button type={type} disabled={disabled} onClick={onClick} className={`${base} ${sizes[size]} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
}

/* ================= Badge ================= */
const tones: Record<string, string> = {
  green: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  amber: 'bg-amber-50 text-amber-700 border-amber-200',
  orange: 'bg-orange-50 text-orange-700 border-orange-200',
  red: 'bg-red-50 text-red-600 border-red-200',
  blue: 'bg-blue-50 text-blue-700 border-blue-200',
  sky: 'bg-sky-50 text-sky-700 border-sky-200',
  gray: 'bg-gray-100 text-gray-600 border-gray-200',
  indigo: 'bg-indigo-50 text-indigo-600 border-indigo-200',
  violet: 'bg-violet-50 text-violet-700 border-violet-200',
  dark: 'bg-gray-900 text-white border-gray-900',
};
export function Badge({ tone = 'gray', className = '', children, dot }: { tone?: string; className?: string; children: ReactNode; dot?: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1.5 text-[11.5px] font-semibold px-2 py-0.5 rounded-md border ${tones[tone] ?? tones.gray} ${className}`}>
      {dot && <span className="w-1.5 h-1.5 rounded-full bg-current opacity-80" />}
      {children}
    </span>
  );
}

/* ================= Card ================= */
export function Card({ className = '', children, onClick }: { className?: string; children: ReactNode; onClick?: () => void }) {
  return (
    <div onClick={onClick} className={`bg-white rounded-xl border border-gray-200/80 shadow-[0_1px_2px_rgba(16,24,40,.05)] ${onClick ? 'cursor-pointer' : ''} ${className}`}>
      {children}
    </div>
  );
}

/* ================= Modal / SidePanel ================= */
export function Modal({ onClose, w = 'max-w-lg', children, dark }: { onClose: () => void; w?: string; children: ReactNode; dark?: boolean }) {
  return (
    <div className="fixed inset-0 z-[90] flex items-center justify-center p-6">
      <div className={`absolute inset-0 anim-fade ${dark ? 'bg-gray-900/60' : 'bg-slate-900/45'} backdrop-blur-[2px]`} onClick={onClose} />
      <div className={`relative modal-in bg-white rounded-2xl shadow-2xl w-full ${w} max-h-[92vh] flex flex-col overflow-hidden`}>
        {children}
      </div>
    </div>
  );
}
export function SidePanel({ onClose, w = 'w-[640px]', children }: { onClose: () => void; w?: string; children: ReactNode }) {
  return (
    <div className="fixed inset-0 z-[80]">
      <div className="absolute inset-0 bg-slate-900/35 anim-fade" onClick={onClose} />
      <div className={`absolute right-0 top-0 h-full ${w} max-w-[95vw] bg-white shadow-2xl panel-in flex flex-col`}>
        {children}
      </div>
    </div>
  );
}

/* ================= Tabs ================= */
export function Tabs({ items, active, onChange, className = '' }: {
  items: { id: string; label: string; icon?: ReactNode }[]; active: string; onChange: (id: string) => void; className?: string;
}) {
  return (
    <div className={`flex items-center gap-1 border-b border-gray-200 ${className}`}>
      {items.map(t => (
        <button key={t.id} onClick={() => onChange(t.id)}
          className={`relative flex items-center gap-1.5 px-3 py-2.5 text-[13px] font-semibold transition-colors -mb-px border-b-2 ${
            active === t.id ? 'text-indigo-600 border-indigo-500' : 'text-gray-500 border-transparent hover:text-gray-800'}`}>
          {t.icon}{t.label}
        </button>
      ))}
    </div>
  );
}

/* ================= Form ================= */
export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <div className="text-[12px] font-semibold text-gray-500 mb-1.5">{label}</div>
      {children}
      {hint && <div className="text-[11px] text-gray-400 mt-1">{hint}</div>}
    </label>
  );
}
export const inputCls = 'w-full h-9.5 px-3 rounded-lg border border-gray-200 bg-white text-[13.5px] text-gray-900 placeholder:text-gray-400 outline-none transition-shadow focus:border-indigo-400 focus:ring-4 focus:ring-indigo-100';

export function Toggle({ on, onChange, disabled }: { on: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button disabled={disabled} onClick={() => onChange(!on)}
      className={`relative w-9 h-5 rounded-full transition-colors duration-200 shrink-0 ${on ? 'bg-indigo-500' : 'bg-gray-300'} ${disabled ? 'opacity-50' : ''}`}>
      <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200 ${on ? 'left-[18px]' : 'left-0.5'}`} />
    </button>
  );
}

/* ================= Avatar ================= */
export function Avatar({ nick, hue, size = 32, ring }: { nick: string; hue: number; size?: number; ring?: boolean }) {
  const letter = nick.replace('@', '').slice(0, 2).toUpperCase();
  return (
    <div className={`rounded-full flex items-center justify-center font-bold text-white shrink-0 select-none ${ring ? 'ring-2 ring-white' : ''}`}
      style={{ width: size, height: size, fontSize: size * 0.36, background: `linear-gradient(135deg, hsl(${hue} 70% 55%), hsl(${hue + 30} 65% 45%))` }}>
      {letter}
    </div>
  );
}

/* ================= Tooltip ================= */
export function Tip({ label, children, className = '' }: { label: string; children: ReactNode; className?: string }) {
  return <span className={`tip inline-flex ${className}`}><span className="tip-body">{label}</span>{children}</span>;
}

/* ================= CopyBtn ================= */
export function CopyBtn({ text, className = '' }: { text: string; className?: string }) {
  const [ok, setOk] = useState(false);
  const toast = useToast();
  return (
    <button className={`inline-flex items-center gap-1 text-[12px] font-semibold text-indigo-600 hover:text-indigo-700 ${className}`}
      onClick={() => { navigator.clipboard?.writeText(text).catch(() => {}); setOk(true); toast('ok', 'Скопировано в буфер обмена'); setTimeout(() => setOk(false), 1500); }}>
      {ok ? <Check size={13} /> : <Copy size={13} />}{ok ? 'Скопировано' : 'Копировать'}
    </button>
  );
}

/* ================= Stat ================= */
export function Stat({ label, value, sub, icon, tone = 'indigo' }: { label: string; value: ReactNode; sub?: ReactNode; icon?: ReactNode; tone?: string }) {
  const t: Record<string, string> = { indigo: 'bg-indigo-50 text-indigo-600', green: 'bg-emerald-50 text-emerald-600', amber: 'bg-amber-50 text-amber-600', red: 'bg-red-50 text-red-500', blue: 'bg-blue-50 text-blue-500' };
  return (
    <Card className="px-4 py-3.5 flex items-start gap-3 hover:border-indigo-200 transition-colors group">
      {icon && <div className={`w-8.5 h-8.5 rounded-lg flex items-center justify-center shrink-0 ${t[tone]} group-hover:scale-105 transition-transform`}>{icon}</div>}
      <div className="min-w-0">
        <div className="text-[11.5px] font-semibold text-gray-400 uppercase tracking-wide">{label}</div>
        <div className="text-[19px] font-extrabold text-gray-900 font-display leading-tight mt-0.5 tabular-nums">{value}</div>
        {sub && <div className="text-[11.5px] text-gray-400 font-medium mt-0.5">{sub}</div>}
      </div>
    </Card>
  );
}
