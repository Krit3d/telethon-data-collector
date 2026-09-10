import { useState, type FormEvent } from 'react';
import { Zap, Mail, Lock, Eye, EyeOff, LogIn, UserPlus, AlertTriangle } from 'lucide-react';
import { api } from '../services/api';
import { Btn, Field, inputCls, Tabs } from './ui';

export default function LoginModal({ onLogin }: { onLogin: (user: Record<string, unknown>) => void }) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [show, setShow] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = email.trim();
    if (!trimmed || !password) {
      setError('Введите email и пароль');
      return;
    }
    if (mode === 'register' && password.length < 6) {
      setError('Пароль должен содержать не менее 6 символов');
      return;
    }
    setBusy(true);
    setError('');
    try {
      const data = mode === 'register'
        ? await api.register(trimmed, password, name.trim() || undefined)
        : await api.login(trimmed, password);
      onLogin(data.user);
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : 'Не удалось выполнить операцию. Проверьте подключение к серверу');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen flex bg-slate-50 text-gray-900 overflow-hidden">
      <div className="flex-1 min-w-0 flex flex-col items-center justify-center px-6">
        <div className="w-full max-w-[400px]">
          <div className="flex items-center gap-3 mb-8">
            <span className="w-11 h-11 rounded-2xl bg-indigo-500 text-white flex items-center justify-center shadow-lg shadow-indigo-500/30"><Zap size={22} /></span>
            <div>
              <div className="text-[19px] font-extrabold font-display text-gray-900 leading-none">CreatorFlow</div>
              <div className="text-[11px] font-bold text-gray-400 mt-1 tracking-wide uppercase">Influence CRM</div>
            </div>
          </div>

          <form onSubmit={submit} className="bg-white rounded-2xl border border-gray-200 shadow-xl p-6 flex flex-col gap-4">
            <Tabs
              items={[
                { id: 'login', label: 'Вход', icon: <LogIn size={14} /> },
                { id: 'register', label: 'Регистрация', icon: <UserPlus size={14} /> },
              ]}
              active={mode}
              onChange={id => { setMode(id === 'register' ? 'register' : 'login'); setError(''); }}
            />

            <div>
              <h1 className="text-[17px] font-bold text-gray-900">{mode === 'register' ? 'Создание аккаунта' : 'Вход в CRM'}</h1>
              <p className="text-[12.5px] font-medium text-gray-400 mt-1">{mode === 'register' ? 'Получите персональный кабинет CreatorFlow' : 'Используйте учётную запись Twenty CRM'}</p>
            </div>

            {mode === 'register' && (
              <Field label="Имя (необязательно)">
                <div className="relative">
                  <UserPlus size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300" />
                  <input className={inputCls + ' !pl-9'} type="text" autoComplete="name"
                    placeholder="Иван Иванов" value={name} onChange={e => { setName(e.target.value); setError(''); }} />
                </div>
              </Field>
            )}

            <Field label="Email">
              <div className="relative">
                <Mail size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300" />
                <input className={inputCls + ' !pl-9'} type="email" autoComplete="email" autoFocus
                  placeholder="you@company.com" value={email} onChange={e => { setEmail(e.target.value); setError(''); }} />
              </div>
            </Field>

            <Field label="Пароль">
              <div className="relative">
                <Lock size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300" />
                <input className={inputCls + ' !pl-9 !pr-9'} type={show ? 'text' : 'password'} autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
                  placeholder="••••••••" value={password} onChange={e => { setPassword(e.target.value); setError(''); }} />
                <button type="button" onClick={() => setShow(s => !s)} tabIndex={-1}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
                  {show ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </Field>

            {mode === 'register' && (
              <p className="text-[11px] font-medium text-gray-400">Пароль должен содержать не менее 6 символов</p>
            )}

            {error && (
              <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2.5 text-[12.5px] font-semibold text-red-600">
                <AlertTriangle size={15} className="shrink-0" />{error}
              </div>
            )}

            <Btn type="submit" disabled={busy} className="w-full justify-center">
              {busy ? <span className="w-4 h-4 rounded-full border-2 border-white/40 border-t-white animate-spin" /> : <LogIn size={15} />}
              {busy ? (mode === 'register' ? 'Регистрируем…' : 'Входим…') : (mode === 'register' ? 'Зарегистрироваться' : 'Войти')}
            </Btn>
          </form>

          <p className="text-[11.5px] font-medium text-gray-400 mt-5 text-center">{mode === 'register' ? 'После регистрации вы попадёте в свой персональный кабинет' : 'Доступ предоставляется администратором Twenty CRM'}</p>
        </div>
      </div>

      <div className="hidden lg:flex w-[420px] shrink-0 bg-gradient-to-br from-indigo-600 via-indigo-500 to-violet-600 relative overflow-hidden">
        <div className="absolute inset-0 opacity-15" style={{ background: 'radial-gradient(circle at 20% 20%, rgba(255,255,255,.6), transparent 45%), radial-gradient(circle at 85% 75%, rgba(255,255,255,.5), transparent 40%)' }} />
        <div className="relative p-8 flex flex-col gap-6">
          <div className="text-[13px] font-bold text-white/80 uppercase tracking-widest">CreatorFlow</div>
          <div className="text-[26px] font-display font-semibold text-white leading-snug">Шортлист авторов, сделки и аналитика в одном окне</div>
          <div className="flex flex-col gap-3 text-[13px] font-semibold text-white/90">
            <span className="flex items-center gap-2.5"><span className="w-6 h-6 rounded-lg bg-white/15 flex items-center justify-center text-[12px]">1</span>Реальные авторы из Twenty CRM</span>
            <span className="flex items-center gap-2.5"><span className="w-6 h-6 rounded-lg bg-white/15 flex items-center justify-center text-[12px]">2</span>Статусы и охваты в реальном времени</span>
            <span className="flex items-center gap-2.5"><span className="w-6 h-6 rounded-lg bg-white/15 flex items-center justify-center text-[12px]">3</span>Воронка сделок и ERID-реестр</span>
          </div>
        </div>
      </div>
    </div>
  );
}