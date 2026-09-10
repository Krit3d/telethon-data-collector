import { useState } from 'react';
import { Plus, Save, Upload, Check, X, Trash2 } from 'lucide-react';
import { BRANDS, type Brand } from '../data';
import { Badge, Btn, Card, Field, inputCls, useToast, PostThumbWrap } from '../components/kb-helpers';

export default function KnowledgeBase({ initialBrandId }: { initialBrandId?: string }) {
  const toast = useToast();
  const [brands, setBrands] = useState<Brand[]>(BRANDS);
  const [selId, setSelId] = useState(initialBrandId ?? BRANDS[0].id);
  const [tab, setTab] = useState('main');
  const sel = brands.find(b => b.id === selId)!;

  const patch = (p: Partial<Brand>) => setBrands(bs => bs.map(b => b.id === selId ? { ...b, ...p } : b));
  const patchArr = (key: keyof Brand, v: string[]) => patch({ [key]: v } as Partial<Brand>);

  const addBrand = () => {
    const nb: Brand = {
      id: 'b' + Date.now(), name: 'Новый бренд', letter: 'Н', hue: 260, category: 'Другое', site: '',
      desc: '', shortDesc: '', status: 'Черновик',
      products: [], usp: [], colors: ['#6366F1'], fonts: '', logoRules: '',
      tovTraits: [], tovPhrases: [], tovForbidden: [], allowed: [], forbidden: [], mentions: [],
      payTypes: [
        { id: 'fix', label: 'Фиксированная стоимость публикации', on: true, rate: '', def: true },
        { id: 'barter', label: 'За товар (бартер)', on: false, rate: '' },
      ],
    };
    setBrands(bs => [...bs, nb]); setSelId(nb.id); setTab('main');
    toast('ok', 'Бренд добавлен — заполните основную информацию');
  };

  const TABS = [
    ['main', 'Основная информация'], ['product', 'Продукт и УТП'],
    ['rules', 'Правила для авторов'], ['terms', 'Условия сотрудничества'],
  ];

  return (
    <div className="h-full flex">
      {/* Левая панель */}
      <aside className="w-[280px] shrink-0 border-r border-gray-200 bg-white flex flex-col">
        <div className="px-4 pt-5 pb-3">
          <h2 className="text-[15px] font-display font-semibold text-gray-900">База знаний бренда</h2>
          <p className="text-[11.5px] font-medium text-gray-400 mt-1">Гайдлайны и условия для авторов</p>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto scroll-thin px-2.5 flex flex-col gap-1">
          {brands.map(b => (
            <button key={b.id} onClick={() => { setSelId(b.id); setTab('main'); }}
              className={`flex items-center gap-2.5 px-2.5 py-2.5 rounded-xl text-left transition-all ${selId === b.id ? 'bg-indigo-50 border border-indigo-200' : 'hover:bg-slate-50 border border-transparent'}`}>
              <span className="w-8 h-8 rounded-lg flex items-center justify-center text-[11px] font-extrabold text-white shrink-0"
                style={{ background: `linear-gradient(135deg, hsl(${b.hue} 70% 52%), hsl(${b.hue + 25} 65% 42%))` }}>{b.letter}</span>
              <span className="min-w-0 flex-1">
                <span className={`block text-[13px] font-bold truncate ${selId === b.id ? 'text-indigo-700' : 'text-gray-800'}`}>{b.name}</span>
                <span className="block text-[10.5px] font-semibold text-gray-400">{b.category}</span>
              </span>
              <span className={`w-2 h-2 rounded-full shrink-0 ${b.status === 'Активен' ? 'bg-emerald-500' : 'bg-gray-300'}`} />
            </button>
          ))}
        </div>
        <div className="p-3">
          <Btn variant="outline" className="w-full" onClick={addBrand}><Plus size={14} />Добавить бренд</Btn>
        </div>
      </aside>

      {/* Редактор */}
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="px-6 py-4 bg-white border-b border-gray-200 shrink-0">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="w-11 h-11 rounded-xl flex items-center justify-center text-[15px] font-extrabold text-white shrink-0"
              style={{ background: `linear-gradient(135deg, hsl(${sel.hue} 70% 52%), hsl(${b_hue(sel)} 65% 42%))` }}>{sel.letter}</span>
            <input
              value={sel.name}
              onChange={e => patch({ name: e.target.value })}
              className="text-[20px] font-display font-semibold text-gray-900 bg-transparent outline-none border-b-2 border-transparent focus:border-indigo-400 transition-colors min-w-0 flex-1" />
            <Btn variant="secondary" size="sm" onClick={() => { patch({ hue: (sel.hue + 70) % 360 }); toast('ok', 'Логотип обновлён'); }}><Upload size={13} />Логотип</Btn>
            <button onClick={() => { patch({ status: sel.status === 'Активен' ? 'Черновик' : 'Активен' }); toast('info', `Статус: ${sel.status === 'Активен' ? 'Черновик' : 'Активен'}`); }}>
              <Badge tone={sel.status === 'Активен' ? 'green' : 'gray'} dot>{sel.status}</Badge>
            </button>
            <Btn onClick={() => toast('ok', `База знаний «${sel.name}» сохранена`)} className="ml-auto"><Save size={14} />Сохранить</Btn>
          </div>
          <div className="flex items-center gap-1 mt-3 -mb-4">
            {TABS.map(([id, l]) => (
              <button key={id} onClick={() => setTab(id)}
                className={`px-3 py-2 text-[12.5px] font-bold rounded-t-lg border-b-2 transition-colors ${tab === id ? 'text-indigo-600 border-indigo-500' : 'text-gray-500 border-transparent hover:text-gray-800'}`}>{l}</button>
            ))}
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto scroll-thin p-6">
          <div className="max-w-[860px] mx-auto">
            {tab === 'main' && (
              <Card className="p-5 anim-in flex flex-col gap-4">
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Название бренда"><input className={inputCls} value={sel.name} onChange={e => patch({ name: e.target.value })} /></Field>
                  <Field label="Категория">
                    <select className={inputCls} value={sel.category} onChange={e => patch({ category: e.target.value })}>
                      {['Красота и уход', 'Электроника', 'Спорт и одежда', 'E-commerce', 'Финансы', 'Еда и напитки', 'Другое'].map(c => <option key={c}>{c}</option>)}
                    </select>
                  </Field>
                </div>
                <Field label="Описание бренда">
                  <textarea rows={4} className={inputCls + ' !h-auto py-2.5 resize-none leading-relaxed'} value={sel.desc} onChange={e => patch({ desc: e.target.value })} />
                </Field>
                <Field label="Краткое описание для авторов" hint="Одна строка — увидит автор в карточке сделки">
                  <input className={inputCls} value={sel.shortDesc} onChange={e => patch({ shortDesc: e.target.value })} />
                </Field>
              </Card>
            )}

            {tab === 'product' && (
              <div className="flex flex-col gap-4 anim-in">
                <div className="flex items-center justify-between">
                  <h3 className="text-[13.5px] font-bold text-gray-900">Карточки продуктов</h3>
                  <Btn variant="outline" size="sm" onClick={() => { patch({ products: [...sel.products, { name: 'Новый продукт', desc: '', tags: [], hue: Math.floor(Math.random() * 360) }] }); toast('ok', 'Продукт добавлен'); }}>
                    <Plus size={13} />Добавить продукт
                  </Btn>
                </div>
                {sel.products.map((p, i) => (
                  <Card key={i} className="p-4 flex gap-4">
                    <PostThumbWrap hue={p.hue} size={92} />
                    <div className="flex-1 min-w-0 flex flex-col gap-2">
                      <div className="flex items-center gap-2">
                        <input className={inputCls + ' !font-bold'} value={p.name}
                          onChange={e => patch({ products: sel.products.map((x, xi) => xi === i ? { ...x, name: e.target.value } : x) })} />
                        <button className="w-8 h-8 rounded-lg text-gray-300 hover:text-red-500 hover:bg-red-50 flex items-center justify-center shrink-0 transition-colors"
                          onClick={() => { patch({ products: sel.products.filter((_, xi) => xi !== i) }); toast('info', 'Продукт удалён'); }}>
                          <Trash2 size={15} />
                        </button>
                      </div>
                      <textarea rows={2} className={inputCls + ' !h-auto py-2 resize-none'} value={p.desc}
                        onChange={e => patch({ products: sel.products.map((x, xi) => xi === i ? { ...x, desc: e.target.value } : x) })} />
                      <div className="flex flex-wrap gap-1.5">
                        {p.tags.map(t => <Badge key={t} tone="indigo">{t}</Badge>)}
                        <button className="text-[11px] font-bold text-indigo-500 hover:text-indigo-700 px-1"
                          onClick={() => {
                            const t = prompt('Ключевое преимущество:');
                            if (t) patch({ products: sel.products.map((x, xi) => xi === i ? { ...x, tags: [...x.tags, t] } : x) });
                          }}>+ тег</button>
                      </div>
                    </div>
                  </Card>
                ))}
                {sel.products.length === 0 && <Card className="p-8 text-center text-[12.5px] font-semibold text-gray-400">Продуктов пока нет — добавьте первый</Card>}
                <Card className="p-4">
                  <div className="flex items-center justify-between mb-2.5">
                    <h4 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide">Уникальные торговые предложения (УТП)</h4>
                    <Btn variant="ghost" size="xs" onClick={() => patchArr('usp', [...sel.usp, 'Новое УТП'])}><Plus size={12} />Добавить</Btn>
                  </div>
                  <EditableList items={sel.usp} onChange={v => patchArr('usp', v)} tone="indigo" />
                </Card>
              </div>
            )}

            {tab === 'rules' && (
              <div className="flex flex-col gap-4 anim-in">
                <div className="grid grid-cols-2 gap-4">
                  <Card className="p-4 border-emerald-200">
                    <h4 className="text-[12px] font-bold text-emerald-700 uppercase tracking-wide mb-2.5 flex items-center gap-1.5"><Check size={13} />Что можно говорить</h4>
                    <EditableList items={sel.allowed} onChange={v => patchArr('allowed', v)} tone="green" addLabel="Добавить пункт" />
                  </Card>
                  <Card className="p-4 border-red-200">
                    <h4 className="text-[12px] font-bold text-red-600 uppercase tracking-wide mb-2.5 flex items-center gap-1.5"><X size={13} />Что запрещено</h4>
                    <EditableList items={sel.forbidden} onChange={v => patchArr('forbidden', v)} tone="red" addLabel="Добавить запрет" />
                  </Card>
                </div>
                <Card className="p-4">
                  <h4 className="text-[12px] font-bold text-gray-400 uppercase tracking-wide mb-2.5">Обязательные упоминания</h4>
                  <EditableList items={sel.mentions} onChange={v => patchArr('mentions', v)} tone="indigo" addLabel="Добавить упоминание" />
                  <p className="text-[11px] font-semibold text-gray-400 mt-2.5">Включая требования 38-ФЗ «О рекламе» и Роскомнадзора</p>
                </Card>
              </div>
            )}

            {tab === 'terms' && (
              <div className="flex flex-col gap-3 anim-in">
                {sel.payTypes.map(pt => (
                  <Card key={pt.id} className={`p-4 flex items-center gap-4 transition-colors ${pt.on ? 'border-indigo-200 bg-indigo-50/30' : 'opacity-70'}`}>
                    <ToggleLocal on={pt.on} onChange={v => patch({ payTypes: sel.payTypes.map(x => x.id === pt.id ? { ...x, on: v } : x) })} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-[13px] font-bold text-gray-900">{pt.label}</span>
                        {pt.def && pt.on && <Badge tone="indigo">по умолчанию</Badge>}
                      </div>
                      {pt.on && (
                        <input className={inputCls + ' !h-8 !text-[12px] mt-2 !w-72'} value={pt.rate} placeholder="Ставка / параметры"
                          onChange={e => patch({ payTypes: sel.payTypes.map(x => x.id === pt.id ? { ...x, rate: e.target.value } : x) })} />
                      )}
                    </div>
                    {pt.on && !pt.def && (
                      <Btn variant="ghost" size="xs" onClick={() => patch({ payTypes: sel.payTypes.map(x => ({ ...x, def: x.id === pt.id })) })}>По умолчанию</Btn>
                    )}
                  </Card>
                ))}
                <Btn variant="outline" size="sm" className="self-start" onClick={() => { patch({ payTypes: [...sel.payTypes, { id: 'p' + Date.now(), label: 'Новый тип оплаты', on: true, rate: '' }] }); toast('ok', 'Тип оплаты добавлен'); }}>
                  <Plus size={13} />Добавить тип оплаты
                </Btn>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function b_hue(b: Brand) { return b.hue + 25; }

function EditableList({ items, onChange, tone, addLabel = 'Добавить' }: {
  items: string[]; onChange: (v: string[]) => void; tone: 'green' | 'red' | 'indigo'; addLabel?: string;
}) {
  const dot = tone === 'green' ? 'text-emerald-500' : tone === 'red' ? 'text-red-500' : 'text-indigo-500';
  return (
    <div className="flex flex-col gap-1.5">
      {items.map((it, i) => (
        <div key={i} className="group flex items-center gap-2">
          <span className={`shrink-0 ${dot}`}>{tone === 'red' ? <X size={13} /> : <Check size={13} />}</span>
          <input value={it} onChange={e => onChange(items.map((x, xi) => xi === i ? e.target.value : x))}
            className="flex-1 text-[12.5px] font-medium text-gray-700 bg-transparent outline-none border-b border-transparent hover:border-gray-200 focus:border-indigo-400 transition-colors py-1" />
          <button className="opacity-0 group-hover:opacity-100 text-gray-300 hover:text-red-500 transition-all shrink-0"
            onClick={() => onChange(items.filter((_, xi) => xi !== i))}><Trash2 size={13} /></button>
        </div>
      ))}
      {items.length === 0 && <div className="text-[11.5px] font-semibold text-gray-300 py-1">Пусто — добавьте первый пункт</div>}
      <button className={`self-start text-[11.5px] font-bold mt-1 ${dot} hover:opacity-70`} onClick={() => onChange([...items, ''])}>+ {addLabel}</button>
    </div>
  );
}

function ToggleLocal({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <button onClick={() => onChange(!on)} className={`relative w-9 h-5 rounded-full transition-colors duration-200 shrink-0 ${on ? 'bg-indigo-500' : 'bg-gray-300'}`}>
      <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200 ${on ? 'left-[18px]' : 'left-0.5'}`} />
    </button>
  );
}
