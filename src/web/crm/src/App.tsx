import { useEffect, useMemo, useState } from 'react';
import { KanbanSquare, Briefcase, Users, MessageCircle, Tag, BookOpen, BarChart3, Bell, Plus, Search, Zap, X, Clapperboard, BellRing, ShieldAlert, PenTool, LogOut } from 'lucide-react';
import { DEALS_INIT, BRANDS, AUTHORS, STAGES, authorById, brandById, fmtMoney, type Deal } from './data';
import { ToastProvider, useToast, Badge, Avatar, Btn, Modal, Field, inputCls } from './components/ui';
import LoginModal from './components/LoginModal';
import { api, userName, userHandle } from './services/api';
import Kanban from './screens/Kanban';
import DealPanel from './screens/DealPanel';
import KnowledgeBase from './screens/KnowledgeBase';
import Authors from './screens/Authors';
import AuthorProfile from './screens/AuthorProfile';
import ContractWizard from './screens/ContractWizard';
import Erid from './screens/Erid';
import Analytics from './screens/Analytics';
import Publications from './screens/Publications';
import { DealsList, CommsList } from './screens/Lists';

type Screen = 'kanban' | 'deals' | 'authors' | 'author' | 'comms' | 'erid' | 'kb' | 'analytics' | 'pubs';

const MENU: { id: Screen; label: string; icon: React.ReactNode }[] = [
  { id: 'kanban', label: 'Воронка сделок', icon: <KanbanSquare size={16} /> },
  { id: 'deals', label: 'Сделки', icon: <Briefcase size={16} /> },
  { id: 'authors', label: 'Авторы', icon: <Users size={16} /> },
  { id: 'comms', label: 'Коммуникации', icon: <MessageCircle size={16} /> },
  { id: 'erid', label: 'ERID-реестр', icon: <Tag size={16} /> },
  { id: 'kb', label: 'База знаний бренда', icon: <BookOpen size={16} /> },
  { id: 'pubs', label: 'Публикации', icon: <Clapperboard size={16} /> },
  { id: 'analytics', label: 'Аналитика', icon: <BarChart3 size={16} /> },
];

const NOTIFS = [
  { icon: <ShieldAlert size={14} />, tone: 'text-red-500 bg-red-50', text: '3 публикации без ERID-маркера', time: '10 мин назад' },
  { icon: <PenTool size={14} />, tone: 'text-indigo-500 bg-indigo-50', text: '@game_zone подписал договор по Xiaomi', time: '1 ч назад' },
  { icon: <BellRing size={14} />, tone: 'text-amber-500 bg-amber-50', text: 'Напоминание: follow-up @travel_diary завтра', time: '3 ч назад' },
];

function Shell({ user, onLogout }: { user: Record<string, unknown> | null; onLogout: () => void }) {
  const toast = useToast();
  const [screen, setScreen] = useState<Screen>('kanban');
  const [deals, setDeals] = useState<Deal[]>(DEALS_INIT);
  const [dealId, setDealId] = useState<string | null>(null);
  const [dealTab, setDealTab] = useState<string | undefined>(undefined);
  const [wizardDeal, setWizardDeal] = useState<Deal | null>(null);
  const [newDeal, setNewDeal] = useState<null | { authorId?: string }>(null);
  const [notifOpen, setNotifOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [kbBrand, setKbBrand] = useState<string | undefined>(undefined);
  const [profileId, setProfileId] = useState<string>('a1');

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setSearchOpen(o => !o); }
      if (e.key === 'Escape') { setSearchOpen(false); setNotifOpen(false); }
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, []);

  const openDeal = (id: string, tab?: string) => { setDealTab(tab); setDealId(id); };
  const deal = dealId ? deals.find(d => d.id === dealId) ?? null : null;
  const openKB = (brandId?: string) => { setKbBrand(brandId); setScreen('kb'); setDealId(null); };

  return (
    <div className="h-screen flex bg-slate-50 text-gray-900 overflow-hidden">
      {/* ===== Сайдбар ===== */}
      <aside className="w-[240px] shrink-0 bg-white border-r border-gray-200 flex flex-col">
        <div className="px-5 pt-5 pb-4 flex items-center gap-2.5">
          <span className="w-9 h-9 rounded-xl bg-indigo-500 text-white flex items-center justify-center shadow-lg shadow-indigo-500/30"><Zap size={18} /></span>
          <div>
            <div className="text-[15px] font-extrabold font-display text-gray-900 leading-none">CreatorFlow</div>
            <div className="text-[10px] font-bold text-gray-400 mt-1 tracking-wide uppercase">Influence CRM</div>
          </div>
        </div>
        <nav className="flex-1 min-h-0 overflow-y-auto scroll-thin px-3 flex flex-col gap-0.5">
          {MENU.map(m => (
            <button key={m.id} onClick={() => { setScreen(m.id); if (m.id !== 'kb') setKbBrand(undefined); if (m.id !== 'author') setProfileId(p => p); }}
              className={`flex items-center gap-2.5 px-3 h-9.5 rounded-lg text-[13px] font-semibold transition-all ${
                screen === m.id ? 'bg-indigo-50 text-indigo-700 shadow-sm shadow-indigo-100' : 'text-gray-500 hover:bg-slate-50 hover:text-gray-800'}`}>
              <span className={screen === m.id ? 'text-indigo-500' : 'text-gray-400'}>{m.icon}</span>{m.label}
              {m.id === 'erid' && <span className="ml-auto text-[10px] font-extrabold text-amber-600 bg-amber-50 border border-amber-200 rounded-md px-1.5 py-0.5">38-ФЗ</span>}
            </button>
          ))}
        </nav>
        <div className="p-3 border-t border-gray-100">
          <div className="flex items-center gap-2.5 px-2 py-2 rounded-xl">
            <Avatar nick={userHandle(user)} hue={255} size={34} />
            <div className="min-w-0">
              <div className="text-[13px] font-bold text-gray-900 truncate">{userName(user)}</div>
              <div className="text-[10.5px] font-semibold text-gray-400">Менеджер · онлайн</div>
            </div>
            <span className="ml-auto w-2 h-2 rounded-full bg-emerald-500 shrink-0" />
          </div>
          <button onClick={onLogout}
            className="mt-1 w-full flex items-center justify-center gap-1.5 h-8 rounded-lg text-[12px] font-semibold text-gray-500 hover:text-red-600 hover:bg-red-50 transition-colors">
            <LogOut size={14} />Выйти
          </button>
        </div>
      </aside>

      {/* ===== Основная область ===== */}
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-[58px] shrink-0 bg-white border-b border-gray-200 flex items-center gap-3 px-5">
          <button onClick={() => setSearchOpen(true)}
            className="flex items-center gap-2.5 h-9 px-3.5 rounded-lg border border-gray-200 bg-slate-50/60 text-gray-400 hover:border-indigo-300 hover:bg-white transition-all w-[340px]">
            <Search size={14} /><span className="text-[12.5px] font-medium">Поиск: сделка, автор, бренд…</span>
            <span className="ml-auto flex items-center gap-1"><span className="kbd">⌘</span><span className="kbd">K</span></span>
          </button>
          <div className="ml-auto flex items-center gap-2">
            <div className="relative">
              <button onClick={() => setNotifOpen(o => !o)} className="relative w-9 h-9 rounded-lg border border-gray-200 bg-white text-gray-500 hover:text-indigo-600 hover:border-indigo-300 flex items-center justify-center transition-colors">
                <Bell size={16} />
                <span className="absolute -top-1 -right-1 w-4.5 h-4.5 min-w-[18px] h-[18px] rounded-full bg-red-500 text-white text-[9.5px] font-extrabold flex items-center justify-center border-2 border-white">3</span>
              </button>
              {notifOpen && (
                <>
                  <div className="fixed inset-0 z-30" onClick={() => setNotifOpen(false)} />
                  <div className="absolute right-0 top-11 w-[330px] bg-white rounded-xl border border-gray-200 shadow-xl z-40 pop-in overflow-hidden">
                    <div className="px-4 py-2.5 border-b border-gray-100 flex items-center justify-between">
                      <span className="text-[12.5px] font-bold text-gray-900">Уведомления</span>
                      <button className="text-[11px] font-bold text-indigo-600 hover:text-indigo-700" onClick={() => { setNotifOpen(false); toast('ok', 'Все уведомления прочитаны'); }}>Прочитать все</button>
                    </div>
                    {NOTIFS.map((n, i) => (
                      <button key={i} onClick={() => { setNotifOpen(false); toast('info', n.text); }} className="w-full flex items-start gap-2.5 px-4 py-3 hover:bg-slate-50 text-left transition-colors">
                        <span className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 ${n.tone}`}>{n.icon}</span>
                        <span className="min-w-0">
                          <span className="block text-[12.5px] font-semibold text-gray-800 leading-snug">{n.text}</span>
                          <span className="block text-[10.5px] font-semibold text-gray-300 mt-0.5">{n.time}</span>
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
            <Btn onClick={() => setNewDeal({})}><Plus size={15} />Новая сделка</Btn>
          </div>
        </header>

        <main className="flex-1 min-h-0 relative">
          {screen === 'kanban' && <Kanban deals={deals} setDeals={setDeals} onOpenDeal={openDeal} onNewDeal={() => setNewDeal({})} onOpenKB={openKB} />}
          {screen === 'deals' && <DealsList deals={deals} onOpenDeal={openDeal} />}
          {screen === 'authors' && <Authors onOpenProfile={id => { setProfileId(id); setScreen('author'); }} onNewDeal={id => setNewDeal({ authorId: id })} />}
          {screen === 'author' && <AuthorProfile authorId={profileId} deals={deals} onBack={() => setScreen('authors')} onNewDeal={id => setNewDeal({ authorId: id })} onOpenDeal={openDeal} />}
          {screen === 'comms' && <CommsList deals={deals} />}
          {screen === 'erid' && <Erid deals={deals} onOpenDeal={openDeal} />}
          {screen === 'kb' && <KnowledgeBase initialBrandId={kbBrand} key={kbBrand ?? 'kb'} />}
          {screen === 'pubs' && <Publications onOpenDeal={openDeal} />}
          {screen === 'analytics' && <Analytics deals={deals} />}
        </main>
      </div>

      {/* ===== Оверлеи ===== */}
      {deal && (
        <DealPanel deal={deal} deals={deals} initialTab={dealTab} key={deal.id + (dealTab ?? '')} onClose={() => setDealId(null)}
          onUpdate={patch => setDeals(prev => prev.map(d => d.id === deal.id ? { ...d, ...patch } : d))}
          onOpenKB={openKB}
          onGenerateContract={d => { setDealId(null); setWizardDeal(d); }} />
      )}
      {wizardDeal && <ContractWizard deal={wizardDeal} onClose={() => setWizardDeal(null)} onDone={() => setWizardDeal(null)} />}
      {newDeal && <NewDealModal initialAuthor={newDeal.authorId} onClose={() => setNewDeal(null)} onCreate={d => {
        setDeals(prev => [d, ...prev]);
        setNewDeal(null);
        setScreen('kanban');
        setDealId(d.id);
        toast('ok', `Сделка «${d.title}» создана в воронке`);
      }} />}
      {searchOpen && <SearchOverlay onClose={() => setSearchOpen(false)} deals={deals}
        onPick={(kind, id) => {
          setSearchOpen(false);
          if (kind === 'deal') { setScreen('kanban'); setDealId(id); }
          if (kind === 'author') { setProfileId(id); setScreen('author'); }
          if (kind === 'brand') openKB(id);
        }} />}
    </div>
  );
}

/* ===== Модальное окно «Новая сделка» ===== */
function NewDealModal({ initialAuthor, onClose, onCreate }: { initialAuthor?: string; onClose: () => void; onCreate: (d: Deal) => void }) {
  const [title, setTitle] = useState('');
  const [authorId, setAuthorId] = useState(initialAuthor ?? AUTHORS[0].id);
  const [brandId, setBrandId] = useState(BRANDS[0].id);
  const [budget, setBudget] = useState('50000');
  const [type, setType] = useState<Deal['type']>('Stories');
  const [pubDate, setPubDate] = useState('2024-10-15');
  const a = authorById(authorId), b = brandById(brandId);

  const create = () => {
    onCreate({
      id: 'd' + Date.now(),
      title: title.trim() || `Интеграция с ${a.nick} — ${b.name}`,
      brandId, authorId, budget: Number(budget) || 50000, type,
      stage: 1, date: new Date().toLocaleDateString('ru-RU'), pubDate: new Date(pubDate).toLocaleDateString('ru-RU'),
      msgs: [], desc: 'Описание и ТЗ будут добавлены на этапе согласования.', terms: `Фиксированная оплата ${fmtMoney(Number(budget) || 50000)}`,
      exclusive: false, edits: 2,
    });
  };

  return (
    <Modal onClose={onClose} w="max-w-[520px]">
      <div className="px-5 py-4 border-b border-gray-100 flex items-center">
        <h2 className="text-[15.5px] font-bold text-gray-900">Новая сделка</h2>
        <button onClick={onClose} className="ml-auto text-gray-300 hover:text-gray-600"><X size={18} /></button>
      </div>
      <div className="p-5 flex flex-col gap-4">
        <Field label="Название"><input className={inputCls} autoFocus placeholder={`Интеграция с ${a.nick}…`} value={title} onChange={e => setTitle(e.target.value)} /></Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Автор">
            <select className={inputCls} value={authorId} onChange={e => setAuthorId(e.target.value)}>
              {AUTHORS.slice(0, 20).map(x => <option key={x.id} value={x.id}>{x.nick} · {x.social}</option>)}
            </select>
          </Field>
          <Field label="Бренд">
            <select className={inputCls} value={brandId} onChange={e => setBrandId(e.target.value)}>
              {BRANDS.map(x => <option key={x.id} value={x.id}>{x.name}</option>)}
            </select>
          </Field>
          <Field label="Бюджет, ₽"><input className={inputCls + ' tabular-nums'} type="number" value={budget} onChange={e => setBudget(e.target.value)} /></Field>
          <Field label="Тип контента">
            <select className={inputCls} value={type} onChange={e => setType(e.target.value as Deal['type'])}>
              {(['Stories', 'Reels', 'Пост', 'Видео'] as const).map(t => <option key={t}>{t}</option>)}
            </select>
          </Field>
        </div>
        <Field label="Дата публикации"><input className={inputCls} type="date" value={pubDate} onChange={e => setPubDate(e.target.value)} /></Field>
        <div className="rounded-xl border border-indigo-100 bg-indigo-50/50 px-3.5 py-2.5 text-[11.5px] font-semibold text-indigo-700">
          Подсказка из базы знаний {b.name}: по умолчанию действует условие «{b.payTypes.find(p => p.def)?.label}»
        </div>
      </div>
      <div className="px-5 py-4 border-t border-gray-100 flex justify-end gap-2 bg-slate-50/50">
        <Btn variant="ghost" onClick={onClose}>Отмена</Btn>
        <Btn onClick={create}><Plus size={14} />Создать сделку</Btn>
      </div>
    </Modal>
  );
}

/* ===== Глобальный поиск ⌘K ===== */
function SearchOverlay({ onClose, deals, onPick }: {
  onClose: () => void; deals: Deal[];
  onPick: (kind: 'deal' | 'author' | 'brand', id: string) => void;
}) {
  const [q, setQ] = useState('');
  const res = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return { deals: deals.slice(0, 3), authors: AUTHORS.slice(0, 3), brands: BRANDS.slice(0, 3) };
    return {
      deals: deals.filter(d => d.title.toLowerCase().includes(s)).slice(0, 4),
      authors: AUTHORS.filter(a => a.nick.toLowerCase().includes(s)).slice(0, 4),
      brands: BRANDS.filter(b => b.name.toLowerCase().includes(s)).slice(0, 4),
    };
  }, [q, deals]);
  const total = res.deals.length + res.authors.length + res.brands.length;

  return (
    <Modal onClose={onClose} w="max-w-[560px]" dark>
      <div className="flex items-center gap-2.5 px-4 border-b border-gray-100">
        <Search size={16} className="text-gray-300" />
        <input autoFocus value={q} onChange={e => setQ(e.target.value)} placeholder="Сделка, автор или бренд…"
          className="flex-1 h-12 text-[14px] font-medium outline-none bg-transparent placeholder:text-gray-300" />
        <span className="kbd">esc</span>
      </div>
      <div className="p-2.5 max-h-[420px] overflow-y-auto scroll-thin">
        {total === 0 && <div className="py-10 text-center text-[13px] font-semibold text-gray-400">Ничего не найдено по запросу «{q}»</div>}
        {res.deals.length > 0 && <GroupTitle>Сделки</GroupTitle>}
        {res.deals.map(d => (
          <Row key={d.id} onClick={() => onPick('deal', d.id)}
            left={<Briefcase size={14} className="text-indigo-500" />}
            main={d.title} sub={`${authorById(d.authorId).nick} · ${fmtMoney(d.budget)} · ${STAGES.find(s => s.id === d.stage)?.name}`} />
        ))}
        {res.authors.length > 0 && <GroupTitle>Авторы</GroupTitle>}
        {res.authors.map(a => (
          <Row key={a.id} onClick={() => onPick('author', a.id)} left={<Avatar nick={a.nick} hue={a.hue} size={22} />}
            main={a.nick} sub={`${a.social} · ${a.niche} · ER ${a.er}%`} />
        ))}
        {res.brands.length > 0 && <GroupTitle>Бренды · база знаний</GroupTitle>}
        {res.brands.map(b => (
          <Row key={b.id} onClick={() => onPick('brand', b.id)}
            left={<span className="w-[22px] h-[22px] rounded-md text-[9px] font-extrabold text-white flex items-center justify-center" style={{ background: `hsl(${b.hue} 70% 50%)` }}>{b.letter}</span>}
            main={b.name} sub={`${b.category} · ${b.status}`} />
        ))}
      </div>
    </Modal>
  );
}
function GroupTitle({ children }: { children: React.ReactNode }) {
  return <div className="px-3 pt-2.5 pb-1 text-[10.5px] font-extrabold text-gray-400 uppercase tracking-wide">{children}</div>;
}
function Row({ left, main, sub, onClick }: { left: React.ReactNode; main: string; sub: string; onClick: () => void }) {
  return (
    <button onClick={onClick} className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-indigo-50 text-left transition-colors">
      <span className="shrink-0 flex items-center">{left}</span>
      <span className="min-w-0">
        <span className="block text-[13px] font-bold text-gray-900 truncate">{main}</span>
        <span className="block text-[11px] font-semibold text-gray-400 truncate">{sub}</span>
      </span>
    </button>
  );
}

export default function App() {
  const [user, setUser] = useState<Record<string, unknown> | null>(() => api.getCurrentUser());
  const [authed, setAuthed] = useState<boolean>(() => api.isAuthenticated());

  const handleLogin = (loggedIn: Record<string, unknown>) => {
    setUser(loggedIn);
    setAuthed(true);
  };

  const handleLogout = () => {
    api.logout();
    setUser(null);
    setAuthed(false);
  };

  return (
    <ToastProvider>
      {authed ? <Shell user={user} onLogout={handleLogout} /> : <LoginModal onLogin={handleLogin} />}
    </ToastProvider>
  );
}
