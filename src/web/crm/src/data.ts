/* Типы и мок-данные CreatorFlow CRM (РФ-рынок) */

export type Social = 'Instagram' | 'VK' | 'Telegram' | 'TikTok' | 'YouTube' | 'Дзен';

export interface Author {
  id: string; nick: string; social: Social; followers: number; niche: string;
  reach: number; cpm: number; er: number; roas: number;
  status: 'Свободен' | 'В сделке' | 'На паузе'; hue: number; deals: number;
}

export interface Product { name: string; desc: string; tags: string[]; hue: number }
export interface PayType { id: string; label: string; on: boolean; rate: string; def?: boolean }

export interface Brand {
  id: string; name: string; letter: string; hue: number; category: string; site: string;
  desc: string; shortDesc: string; status: 'Активен' | 'Черновик';
  products: Product[]; usp: string[]; colors: string[]; fonts: string; logoRules: string;
  tovTraits: string[]; tovPhrases: string[]; tovForbidden: string[];
  allowed: string[]; forbidden: string[]; mentions: string[]; payTypes: PayType[];
}

export interface DealMsg { id: string; kind: 'in' | 'out' | 'sys'; text: string; time: string }

export interface Deal {
  id: string; title: string; brandId: string; authorId: string; budget: number;
  type: 'Stories' | 'Reels' | 'Пост' | 'Видео';
  stage: number; date: string; pubDate: string;
  msgs: DealMsg[]; file?: string; erid?: string;
  desc: string; terms: string; exclusive: boolean; edits: number;
}

export interface EridRec {
  id: string; code: string; dealId: string; authorId: string; brandId: string; date: string;
  status: 'Активен' | 'Ожидает публикации' | 'Опубликован' | 'Архивирован';
  link?: string;
  checks: { date: string; result: string; by: string }[];
}

export interface Publication {
  id: string; authorId: string; social: Social; type: 'Stories' | 'Reels' | 'Пост' | 'Видео';
  date: string; text: string; hue: number;
  views: number; likes: number; comments: number; reposts: number; saves: number;
  clicks: number; conv: number; revenue: number; cost: number;
  eridOk: boolean; link: string; sync: 'syncing' | 'waiting' | 'ok' | 'error';
}

export interface SourceItem {
  id: string; name: string; desc: string; status: 'active' | 'error' | 'off';
  metrics: string; hue: number; letter: string;
}

export interface Note { icon: string; text: string; by: string; tone: 'green' | 'amber' | 'red' | 'indigo' }
export interface Template { id: string; name: string; desc: string; preview: string[]; badge?: string }

export const STAGES = [
  { id: 1, name: 'Первичный контакт', color: '#64748B', soft: '#F1F5F9' },
  { id: 2, name: 'Переговоры', color: '#F97316', soft: '#FFF7ED' },
  { id: 3, name: 'Согласование ТЗ', color: '#F59E0B', soft: '#FFFBEB' },
  { id: 4, name: 'Подписание договора', color: '#6366F1', soft: '#EEF2FF' },
  { id: 5, name: 'Контент создан', color: '#10B981', soft: '#ECFDF5' },
];

export const fmtMoney = (n: number) => `${Math.round(n).toLocaleString('ru-RU')} ₽`;
export const fmtNum = (n: number) =>
  n >= 1_000_000 ? (n / 1_000_000).toFixed(1).replace('.0', '').replace('.', ',') + 'M'
    : n >= 1000 ? (n / 1000).toFixed(1).replace('.0', '').replace('.', ',') + 'K'
      : String(n);

/* ================= АВТОРЫ ================= */
const BASE_AUTHORS: Author[] = [
  { id: 'a1', nick: '@beauty_blog', social: 'Instagram', followers: 245000, niche: 'Красота', reach: 128400, cpm: 180, er: 3.2, roas: 2.1, status: 'В сделке', hue: 330, deals: 12 },
  { id: 'a2', nick: '@tech_guru', social: 'YouTube', followers: 890000, niche: 'Технологии', reach: 402000, cpm: 240, er: 4.8, roas: 4.2, status: 'В сделке', hue: 210, deals: 18 },
  { id: 'a3', nick: '@gadget_review', social: 'YouTube', followers: 1240000, niche: 'Технологии', reach: 615000, cpm: 265, er: 4.1, roas: 3.6, status: 'Свободен', hue: 200, deals: 22 },
  { id: 'a4', nick: '@travel_diary', social: 'Instagram', followers: 412000, niche: 'Путешествия', reach: 189000, cpm: 195, er: 3.8, roas: 2.4, status: 'Свободен', hue: 175, deals: 9 },
  { id: 'a5', nick: '@mama_blog', social: 'VK', followers: 156000, niche: 'Семья', reach: 74000, cpm: 120, er: 5.1, roas: 1.8, status: 'Свободен', hue: 45, deals: 7 },
  { id: 'a6', nick: '@foodie_moscow', social: 'Telegram', followers: 98000, niche: 'Еда', reach: 52000, cpm: 150, er: 6.2, roas: 2.0, status: 'На паузе', hue: 15, deals: 11 },
  { id: 'a7', nick: '@vegan_food', social: 'Telegram', followers: 64000, niche: 'Еда', reach: 31000, cpm: 130, er: 5.6, roas: 1.6, status: 'Свободен', hue: 95, deals: 5 },
  { id: 'a8', nick: '@crypto_talk', social: 'Telegram', followers: 210000, niche: 'Финансы', reach: 98000, cpm: 310, er: 2.9, roas: 1.2, status: 'На паузе', hue: 260, deals: 8 },
  { id: 'a9', nick: '@fitness_pro', social: 'Instagram', followers: 530000, niche: 'Фитнес', reach: 240000, cpm: 175, er: 3.5, roas: 2.2, status: 'В сделке', hue: 130, deals: 14 },
  { id: 'a10', nick: '@fashion_week', social: 'Instagram', followers: 720000, niche: 'Мода', reach: 310000, cpm: 220, er: 3.0, roas: 2.6, status: 'Свободен', hue: 290, deals: 16 },
  { id: 'a11', nick: '@auto_drive', social: 'YouTube', followers: 356000, niche: 'Авто', reach: 150000, cpm: 205, er: 3.3, roas: 1.9, status: 'В сделке', hue: 225, deals: 10 },
  { id: 'a12', nick: '@game_zone', social: 'TikTok', followers: 980000, niche: 'Игры', reach: 470000, cpm: 90, er: 7.4, roas: 1.5, status: 'В сделке', hue: 350, deals: 13 },
];

const GEN_NICHES = ['Красота', 'Технологии', 'Еда', 'Путешествия', 'Фитнес', 'Мода', 'Финансы', 'Семья'];
const GEN_SOCIALS: Social[] = ['Instagram', 'VK', 'Telegram', 'TikTok', 'YouTube', 'Дзен'];
const PREF = ['pro', 'the', 'daily', 'real', 'top', 'max', 'neo', 'go', 'my', 'just'];
const ROOT = ['style', 'life', 'food', 'tech', 'beauty', 'travel', 'fit', 'media', 'blog', 'gram', 'space', 'lab'];
const GEN_STATUS: Author['status'][] = ['Свободен', 'Свободен', 'В сделке', 'На паузе'];

function mulberry(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const AUTHORS: Author[] = (() => {
  const out = [...BASE_AUTHORS];
  const rnd = mulberry(42);
  let i = 0;
  while (out.length < 57) {
    const followers = Math.round((20 + rnd() * 900) * 1000);
    out.push({
      id: 'g' + i,
      nick: '@' + PREF[i % PREF.length] + '_' + ROOT[i % ROOT.length] + (i % 4 === 0 ? String(2 + (i % 9)) : ''),
      social: GEN_SOCIALS[i % GEN_SOCIALS.length],
      followers,
      niche: GEN_NICHES[i % GEN_NICHES.length],
      reach: Math.round(followers * (0.18 + rnd() * 0.35)),
      cpm: Math.round(90 + rnd() * 260),
      er: Math.round((1.4 + rnd() * 6) * 10) / 10,
      roas: Math.round((1 + rnd() * 3.4) * 10) / 10,
      status: GEN_STATUS[Math.floor(rnd() * GEN_STATUS.length)],
      hue: Math.floor(rnd() * 360),
      deals: Math.floor(rnd() * 15),
    });
    i++;
  }
  return out;
})();

/* ================= БРЕНДЫ + БАЗА ЗНАНИЙ ================= */
const stdPay = (rate: string): PayType[] => [
  { id: 'cpm', label: 'За просмотры (CPM)', on: true, rate: '250 ₽ за 1000 просмотров' },
  { id: 'barter', label: 'За товар (бартер)', on: true, rate: 'Продукция до 15 000 ₽' },
  { id: 'cpc', label: 'За клики (CPC)', on: false, rate: '12 ₽ за клик' },
  { id: 'cpa', label: 'За покупки (CPA)', on: false, rate: '8% от заказа' },
  { id: 'fix', label: 'Фиксированная стоимость публикации', on: true, rate, def: true },
  { id: 'hybrid', label: 'Фикс + за просмотры', on: true, rate: 'Фикс + 2 ₽/просмотр свыше 100K' },
];

export const BRANDS: Brand[] = [
  {
    id: 'b1', name: "L'Oréal Paris", letter: 'L', hue: 48, category: 'Красота и уход', site: 'loreal-paris.ru',
    status: 'Активен',
    desc: 'Мировой лидер в области красоты: уход за кожей, волосами и макияж. В России представлены линейки Revitalift, Elseve, Colorista и другие.',
    shortDesc: 'Премиальный уход за кожей по доступной цене. Наука + красота.',
    products: [
      { name: 'Revitalift Филлер', desc: 'Антивозрастной крем-филлер с 5% гиалуроновой кислотой. Заполняет морщины, возвращает объём кожи за 4 недели.', tags: ['Гиалуроновая кислота', '−26% морщин за 4 недели', 'Все типы кожи'], hue: 265 },
      { name: 'Elseve Гиалурон Наполнитель', desc: 'Линия ухода за волосами с гиалуроновой кислотой: увлажнение и плотность без утяжеления.', tags: ['Увлажнение 72 ч', 'Без утяжеления'], hue: 195 },
    ],
    usp: [
      '№1 в мире по уходу за кожей (Euromonitor, 2023)',
      "Формулы разработаны в лабораториях L'Oréal Research",
      'Клинически доказанная эффективность за 4 недели',
      'Доступная цена в сегменте премиум-ухода',
    ],
    colors: ['#241B2F', '#C9A227', '#F5E6D3', '#7A5FAF'],
    fonts: "L'Oréal Sans (основной), Unbounded (акцидентный)",
    logoRules: 'Логотип размещается на светлом фоне, охранное поле — не менее высоты буквы «L». Запрещены искажения, перекрашивание и эффекты тени.',
    tovTraits: ['Экспертный', 'Дружелюбный', 'Вдохновляющий'],
    tovPhrases: [
      '«Ты этого достойна» — ключевой месседж бренда',
      'Говорим о науке формул простым языком',
      'Фокус на результате и ощущениях, а не на составе ради состава',
    ],
    tovForbidden: ['«лучший в мире»', '«гарантированный результат»', 'сравнения с конкурентами по имени', '«лечит», «излечивает»'],
    allowed: [
      'Демонстрация текстуры и процесса нанесения',
      'Личный опыт «до/после» с оговоркой об индивидуальном результате',
      'Упоминание активных компонентов состава',
      'Совместное использование с другими средствами бренда',
    ],
    forbidden: [
      'Медицинские обещания («лечит акне», «излечивает»)',
      'Прямые сравнения «дешевле/лучше» с конкурентами',
      'Ретушь кожи в кадрах с демонстрацией результата',
      'Упоминание инъекционной косметологии в одном кадре',
    ],
    mentions: [
      '«Реклама. ERID: {маркер}» в первых 3 кадрах Stories/Reels',
      'Ссылка на официальный сайт бренда в описании',
      'Возрастная маркировка 18+ для антивозрастных линеек',
    ],
    payTypes: stdPay('от 40 000 ₽'),
  },
  {
    id: 'b2', name: 'Nike', letter: 'N', hue: 20, category: 'Спорт и одежда', site: 'nike.com/ru',
    status: 'Активен',
    desc: 'Мировой производитель спортивной одежды и обуви. Месседж — движение, дисциплина и личный прогресс.',
    shortDesc: 'Just Do It. Спорт для каждого — от новичка до про.',
    products: [
      { name: 'Air Max Dn', desc: 'Кроссовки с динамической воздушной подушкой нового поколения.', tags: ['Динамическая амортизация', 'Лимитированный дроп'], hue: 15 },
    ],
    usp: ['Технологии, проверенные профессиональными атлетами', 'Лимитированные дропы — высокий виральный потенциал', 'Сильное комьюнити Nike Run Club'],
    colors: ['#111111', '#FA5400', '#FFFFFF'],
    fonts: 'Helvetica Now (основной), Futura Condensed (акцидентный)',
    logoRules: 'Свуш не менее 24px по ширине, контраст к фону ≥ 4.5:1. Запрещён наклон и обводка.',
    tovTraits: ['Дерзкий', 'Вдохновляющий', 'Экспертный'],
    tovPhrases: ['Мотивация через действие, а не обещания', 'Английские слоганы не переводим'],
    tovForbidden: ['«дешевле, чем у Adidas»', 'обещания результата без тренировок'],
    allowed: ['Личные спортивные истории', 'Демонстрация в реальных тренировках'],
    forbidden: ['Сравнения с другими брендами', 'Политические высказывания в интеграциях'],
    mentions: ['«Реклама. ERID: {маркер}»', 'Отметка официального аккаунта @nike'],
    payTypes: stdPay('от 80 000 ₽'),
  },
  {
    id: 'b3', name: 'Samsung', letter: 'S', hue: 215, category: 'Электроника', site: 'samsung.com/ru',
    status: 'Активен',
    desc: 'Флагманская линейка Galaxy: смартфоны, планшеты, часы и экосистема умного дома.',
    shortDesc: 'Galaxy AI уже здесь. Технологии, которые работают за вас.',
    products: [
      { name: 'Galaxy S24 Ultra', desc: 'Флагман с Galaxy AI, камерой 200 МП и титановой рамкой.', tags: ['Galaxy AI', 'Камера 200 МП', 'Титан'], hue: 220 },
    ],
    usp: ['Galaxy AI — перевод звонков и генеративное редактирование фото', '7 лет обновлений ОС', 'Лучший дисплей на рынке (DisplayMate A+)'],
    colors: ['#1428A0', '#000000', '#E6E8EB'],
    fonts: 'Samsung One (фирменный)',
    logoRules: 'Эллипс Samsung — только фирменный синий #1428A0, минимальная высота 16px.',
    tovTraits: ['Экспертный', 'Профессиональный'],
    tovPhrases: ['Характеристики через сценарии использования', 'Сравнения только с прошлыми поколениями Galaxy'],
    tovForbidden: ['«убийца iPhone»', 'прямые сравнения с Apple'],
    allowed: ['Распаковки и первые впечатления', 'Сравнение с предыдущими моделями Galaxy', 'Реальные тесты камеры'],
    forbidden: ['Сравнения с Apple и другими брендами по имени', 'Спойлеры до официального релиза'],
    mentions: ['«Реклама. ERID: {маркер}»', 'Ссылка на samsung.com/ru с UTM'],
    payTypes: stdPay('от 150 000 ₽'),
  },
  {
    id: 'b4', name: 'Garnier', letter: 'G', hue: 130, category: 'Красота и уход', site: 'garnier.ru',
    status: 'Активен',
    desc: 'Натуральная косметика масс-маркет сегмента. Фокус на экологичность и доступность.',
    shortDesc: 'Сила природы для вашей кожи. Эко-ответственный бренд.',
    products: [
      { name: 'Botanic Therapy', desc: 'Линия ухода с натуральными маслами и 98% биоразлагаемой формулой.', tags: ['98% натуральные компоненты', 'Эко-упаковка'], hue: 120 },
    ],
    usp: ['Формулы на 98% из натуральных компонентов', 'Перерабатываемая упаковка', 'Цена масс-маркета при качестве ухода'],
    colors: ['#00754A', '#A4D65E', '#FFFFFF'],
    fonts: 'Garnier Rounded (фирменный)',
    logoRules: 'Логотип только на белом или фирменном зелёном фоне.',
    tovTraits: ['Дружелюбный', 'Заботливый'],
    tovPhrases: ['Простые слова о составах', 'Акцент на экологичность без гринвошинга'],
    tovForbidden: ['«полностью натуральное» (юридически рискованно)', '«эко на 100%»'],
    allowed: ['Семейные сценарии использования', 'Разбор составов'],
    forbidden: ['Гринвошинг и недоказуемые эко-заявления'],
    mentions: ['«Реклама. ERID: {маркер}»'],
    payTypes: stdPay('от 25 000 ₽'),
  },
  {
    id: 'b5', name: 'Xiaomi', letter: 'X', hue: 25, category: 'Электроника', site: 'mi.com/ru',
    status: 'Черновик',
    desc: 'Смартфоны и экосистема умных устройств. Лидер по соотношению цена/качество.',
    shortDesc: 'Инновации для всех. Топ-железо без переплаты.',
    products: [
      { name: 'Xiaomi 14T Pro', desc: 'Флагманская камера Leica и чип Dimensity 9300+ по цене середняка.', tags: ['Камера Leica', 'Dimensity 9300+'], hue: 25 },
    ],
    usp: ['Камеры Leica в среднем сегменте', 'Гиперзарядка 120 Вт — 100% за 19 минут', 'Экосистема из 200+ устройств'],
    colors: ['#FF6900', '#000000', '#F5F5F5'],
    fonts: 'Mi Sans (фирменный)',
    logoRules: 'Оранжевый логотип MI — только фирменный #FF6900.',
    tovTraits: ['Юмористический', 'Дружелюбный'],
    tovPhrases: ['Ирония над переплатой за бренды', 'Цифры и бенчмарки'],
    tovForbidden: ['«лучше iPhone за эти деньги»'],
    allowed: ['Бенчмарки и сравнения по цифрам без имён конкурентов', 'Юмористические форматы'],
    forbidden: ['Прямые сравнения с Apple по имени'],
    mentions: ['«Реклама. ERID: {маркер}»'],
    payTypes: stdPay('от 60 000 ₽'),
  },
  {
    id: 'b6', name: 'Ozon', letter: 'O', hue: 225, category: 'E-commerce', site: 'ozon.ru',
    status: 'Активен',
    desc: 'Маркетплейс №2 в России. Кампании: быстрая доставка, подборки, кэшбек баллами.',
    shortDesc: 'Всё нужное — за 15 минут. Доставка, которая удивляет.',
    products: [
      { name: 'Ozon Fresh', desc: 'Доставка продуктов и готовой еды за 15 минут в крупных городах.', tags: ['Доставка за 15 мин', 'Кэшбек баллами'], hue: 225 },
    ],
    usp: ['Доставка за 15 минут (Ozon Fresh)', 'Кэшбек баллами до 25%', 'Пункты выдачи у дома'],
    colors: ['#005BFF', '#F91D47', '#FFFFFF'],
    fonts: 'Ozon Sans (фирменный)',
    logoRules: 'Синий #005BFF основной, розовый акцент не более 30% макета.',
    tovTraits: ['Дружелюбный', 'Юмористический'],
    tovPhrases: ['Живые сценарии «заказал — уже курьер»', 'Акцент на экономию времени'],
    tovForbidden: ['сравнения сроков доставки с Wildberries по имени'],
    allowed: ['Скринкасты процесса заказа', 'Распаковки и подборки товаров'],
    forbidden: ['Прямые сравнения с конкурентами', 'Обещание точного времени доставки («всегда за 15 минут»)'],
    mentions: ['«Реклама. ERID: {маркер}»', 'Промокод автора в описании'],
    payTypes: stdPay('от 30 000 ₽'),
  },
];

/* ================= СДЕЛКИ ================= */
const M = (id: string, kind: DealMsg['kind'], text: string, time: string): DealMsg => ({ id, kind, text, time });

export const DEALS_INIT: Deal[] = [
  {
    id: 'd1', title: 'Промо кроссовок Air Max — Reels', brandId: 'b2', authorId: 'a4', budget: 120000,
    type: 'Reels', stage: 1, date: '19.09.2024', pubDate: '10.10.2024',
    msgs: [M('m1', 'in', 'Здравствуйте! Какие сроки по интеграции Air Max?', '12:04')],
    desc: 'Reels 30–45 секунд с демонстрацией новой расцветки Air Max Dn. Акцент на комфорт в путешествиях, съёмка в аэропорту и городе.',
    terms: 'Фиксированная оплата 120 000 ₽', exclusive: false, edits: 2,
  },
  {
    id: 'd2', title: 'Детская линия шампуней — подборка', brandId: 'b4', authorId: 'a5', budget: 35000,
    type: 'Пост', stage: 1, date: '18.09.2024', pubDate: '05.10.2024',
    msgs: [M('m1', 'in', 'Добрый день! Интересует формат поста с подборкой для мам', '09:12')],
    desc: 'Пост-подборка «5 средств для купания малышей» с органичным включением Botanic Therapy Kids.',
    terms: 'Фиксированная оплата 35 000 ₽', exclusive: false, edits: 1,
  },
  {
    id: 'd3', title: 'Galaxy Watch 7 — обзор для бегунов', brandId: 'b3', authorId: 'a9', budget: 90000,
    type: 'Reels', stage: 1, date: '17.09.2024', pubDate: '01.10.2024',
    msgs: [],
    desc: 'Reels о трекинге тренировок с Galaxy Watch 7: пульс, VO2 max, рекомендации по восстановлению.',
    terms: 'Фикс 60 000 ₽ + 30 000 ₽ при охвате 300K+', exclusive: false, edits: 2,
  },
  {
    id: 'd4', title: 'Подборка ПП-продуктов в Stories', brandId: 'b6', authorId: 'a7', budget: 28000,
    type: 'Stories', stage: 1, date: '16.09.2024', pubDate: '28.09.2024',
    msgs: [M('m1', 'out', 'Добрый день! Отправили бриф по подборке продуктов, посмотрите 🙌', '15:40')],
    desc: 'Серия из 5 Stories: заказ ПП-продуктов в Ozon Fresh, распаковка, приготовление.',
    terms: 'Фиксированная оплата 28 000 ₽ + промокод VEGAN10', exclusive: false, edits: 1,
  },
  {
    id: 'd5', title: 'Рекламная интеграция — крем для лица', brandId: 'b1', authorId: 'a1', budget: 45000,
    type: 'Stories', stage: 2, date: '15.09.2024', pubDate: '30.09.2024', file: 'Договор_2024.docx',
    msgs: [
      M('m1', 'in', 'Добрый день! Готова обсудить условия', '10:30'),
      M('m2', 'out', 'Отлично, высылаю ТЗ и бюджет', '10:45'),
      M('m3', 'sys', 'Менеджер прикрепил файл договора', '10:46'),
    ],
    desc: 'Серия из 4 Stories: проблема → знакомство с Revitalift Филлер → нанесение → результат через 4 недели (архивные кадры). ERID в первых 3 кадрах.',
    terms: 'Фиксированная оплата 45 000 ₽ + 2 ₽ за просмотр свыше 100K', exclusive: false, edits: 2,
  },
  {
    id: 'd6', title: 'Galaxy S24 Ultra — большой видеообзор', brandId: 'b3', authorId: 'a2', budget: 240000,
    type: 'Видео', stage: 2, date: '14.09.2024', pubDate: '12.10.2024',
    msgs: [
      M('m1', 'in', 'Готов взять обзор, но нужна техника до 25.09', '18:22'),
      M('m2', 'out', 'Доставим курьером 23.09, курьер согласует время', '18:51'),
    ],
    desc: 'Ролик 12–15 минут: камера (сравнение с S23 Ultra), Galaxy AI в реальных задачах, автономность. Интеграция 60–90 секунд в начале.',
    terms: 'Фиксированная оплата 240 000 ₽, выплата 50/50', exclusive: true, edits: 2,
  },
  {
    id: 'd7', title: 'Stories-серия: тренировки в Nike Zoom 2', brandId: 'b2', authorId: 'a9', budget: 60000,
    type: 'Stories', stage: 2, date: '13.09.2024', pubDate: '07.10.2024',
    msgs: [M('m1', 'sys', 'Автор запросил 42 размер для съёмки', '11:02')],
    desc: '6 Stories: утренний забег, ощущения от амортизации, замеры темпа через Nike Run Club.',
    terms: 'Фиксированная оплата 60 000 ₽ + пара кроссовок (бартер)', exclusive: false, edits: 1,
  },
  {
    id: 'd8', title: 'Reels: макияж Paris Fashion Week', brandId: 'b1', authorId: 'a10', budget: 180000,
    type: 'Reels', stage: 3, date: '11.09.2024', pubDate: '20.10.2024', file: 'ТЗ_PFW.pdf',
    msgs: [M('m1', 'in', 'ТЗ посмотрела, есть вопрос по хэштегам — можно свои?', '14:17')],
    desc: "Reels 30 секунд: 3 образа с показа на базе продуктов L'Oréal Paris. Титры с названиями средств, ERID в описании и на 1 кадре.",
    terms: 'Фиксированная оплата 180 000 ₽', exclusive: false, edits: 2,
  },
  {
    id: 'd9', title: 'Распаковка Xiaomi 14T Pro', brandId: 'b5', authorId: 'a3', budget: 150000,
    type: 'Видео', stage: 3, date: '10.09.2024', pubDate: '15.10.2024', file: 'ТЗ_14TPro.docx',
    msgs: [],
    desc: 'Видео 8–10 минут: распаковка, первые впечатления, тест камеры Leica при ночной съёмке.',
    terms: 'Фикс 100 000 ₽ + 50 000 ₽ при 500K просмотров', exclusive: false, edits: 2,
  },
  {
    id: 'd10', title: 'Интеграция: Galaxy Tab в дороге', brandId: 'b3', authorId: 'a11', budget: 110000,
    type: 'Видео', stage: 4, date: '08.09.2024', pubDate: '05.10.2024', file: 'Договор_Samsung_08.09.pdf',
    erid: 'ERID-1694683200-A7B3C9D2E1',
    msgs: [M('m1', 'sys', 'Договор подписан обеими сторонами', '16:30')],
    desc: 'Интеграция 45 секунд в ролик об автопутешествии: планшет как навигатор и развлечение для пассажиров.',
    terms: 'Фиксированная оплата 110 000 ₽', exclusive: false, edits: 1,
  },
  {
    id: 'd11', title: 'Стрим + пост: Poco X6 для геймеров', brandId: 'b5', authorId: 'a12', budget: 75000,
    type: 'Пост', stage: 4, date: '07.09.2024', pubDate: '02.10.2024', file: 'Договор_Xiaomi_07.09.pdf',
    erid: 'ERID-1694596800-C5D1E9F3A4',
    msgs: [M('m1', 'sys', 'Договор на подписи у автора', '10:15')],
    desc: 'Часовой стрим с геймплеем на Poco X6 + закреп с результатами FPS-тестов.',
    terms: 'Фикс 50 000 ₽ + 25 000 ₽ при 200K просмотров стрима', exclusive: false, edits: 1,
  },
  {
    id: 'd12', title: 'Пост: доставка Ozon Fresh за 15 минут', brandId: 'b6', authorId: 'a6', budget: 40000,
    type: 'Пост', stage: 5, date: '01.09.2024', pubDate: '12.09.2024', file: 'Акт_FOODIE.pdf',
    erid: 'ERID-1694509800-B8C4D0E3F2',
    msgs: [M('m1', 'sys', 'Контент опубликован, метрики синхронизированы', '13:00')],
    desc: 'Пост с таймлапсом заказа: корзина → курьер → стол накрыт. Промокод FOODIE15.',
    terms: 'Фиксированная оплата 40 000 ₽', exclusive: false, edits: 1,
  },
];

/* ================= ERID-РЕЕСТР ================= */
export const ERIDS: EridRec[] = [
  { id: 'e1', code: 'ERID-1694683200-A7B3C9D2E1', dealId: 'd10', authorId: 'a11', brandId: 'b3', date: '14.09.2024', status: 'Активен', checks: [{ date: '14.09.2024', result: 'Сгенерирован вручную', by: 'Анна Соколова' }] },
  { id: 'e2', code: 'ERID-1694596800-C5D1E9F3A4', dealId: 'd11', authorId: 'a12', brandId: 'b5', date: '13.09.2024', status: 'Ожидает публикации', checks: [{ date: '13.09.2024', result: 'Сгенерирован вручную', by: 'Игорь Волков' }] },
  { id: 'e3', code: 'ERID-1694509800-B8C4D0E3F2', dealId: 'd12', authorId: 'a6', brandId: 'b6', date: '12.09.2024', status: 'Опубликован', link: 't.me/foodie_moscow/1284', checks: [{ date: '12.09.2024', result: 'Сгенерирован вручную', by: 'Игорь Волков' }, { date: '13.09.2024', result: 'Маркер найден, размещение корректно', by: 'Анна Соколова' }] },
  { id: 'e4', code: 'ERID-1693998700-D9E2F4A6B8', dealId: 'd6', authorId: 'a2', brandId: 'b3', date: '10.09.2024', status: 'Ожидает публикации', checks: [{ date: '10.09.2024', result: 'Сгенерирован вручную', by: 'Анна Соколова' }] },
  { id: 'e5', code: 'ERID-1693912300-E1F3A5B7C9', dealId: 'd8', authorId: 'a10', brandId: 'b1', date: '09.09.2024', status: 'Ожидает публикации', checks: [{ date: '09.09.2024', result: 'Сгенерирован вручную', by: 'Мария Ким' }] },
  { id: 'e6', code: 'ERID-1692703200-F2A4B6C8D0', dealId: 'd3', authorId: 'a9', brandId: 'b2', date: '22.08.2024', status: 'Опубликован', link: 'instagram.com/p/Cx94kdL', checks: [{ date: '22.08.2024', result: 'Сгенерирован вручную', by: 'Мария Ким' }, { date: '25.08.2024', result: 'Маркер найден в первых 3 кадрах', by: 'Игорь Волков' }] },
  { id: 'e7', code: 'ERID-1691494100-A3B5C7D9E1', dealId: 'd7', authorId: 'a9', brandId: 'b2', date: '08.08.2024', status: 'Опубликован', link: 'instagram.com/stories/fit', checks: [{ date: '08.08.2024', result: 'Сгенерирован вручную', by: 'Анна Соколова' }, { date: '11.08.2024', result: 'Размещение корректно', by: 'Анна Соколова' }] },
  { id: 'e8', code: 'ERID-1689075900-B4C6D8E0F2', dealId: 'd9', authorId: 'a3', brandId: 'b5', date: '12.07.2024', status: 'Архивирован', checks: [{ date: '12.07.2024', result: 'Сгенерирован вручную', by: 'Игорь Волков' }, { date: '30.07.2024', result: 'Кампания завершена, архив', by: 'система' }] },
  { id: 'e9', code: 'ERID-1688122500-C5D7E9F1A3', dealId: 'd1', authorId: 'a4', brandId: 'b2', date: '01.07.2024', status: 'Архивирован', checks: [{ date: '01.07.2024', result: 'Сгенерирован вручную', by: 'Мария Ким' }, { date: '20.07.2024', result: 'Кампания завершена, архив', by: 'система' }] },
];

/* ================= ПУБЛИКАЦИИ ================= */
const P = (id: string, authorId: string, social: Publication['social'], type: Publication['type'], date: string, text: string, hue: number,
  views: number, likes: number, comments: number, reposts: number, saves: number, clicks: number, conv: number,
  revenue: number, cost: number, eridOk: boolean, link: string, sync: Publication['sync']): Publication =>
  ({ id, authorId, social, type, date, text, hue, views, likes, comments, reposts, saves, clicks, conv, revenue, cost, eridOk, link, sync });

export const PUBLICATIONS: Publication[] = [
  P('p1', 'a2', 'YouTube', 'Видео', '18.09.2024', 'Galaxy S24 Ultra: честный обзор после месяца использования', 215, 486000, 31200, 2140, 980, 4300, 12400, 640, 918000, 240000, true, 'youtu.be/gx24rev', 'ok'),
  P('p2', 'a1', 'Instagram', 'Stories', '16.09.2024', 'Мой утренний ритуал: 4 шага к сияющей коже + промокод', 330, 125400, 8900, 640, 210, 1800, 4100, 205, 268000, 45000, true, 'instagram.com/stories/beauty', 'ok'),
  P('p3', 'a12', 'TikTok', 'Пост', '15.09.2024', 'Собрал ПК-сетап мечты за 60 секунд (часть 2)', 350, 1240000, 98000, 5400, 12800, 21000, 18600, 720, 655000, 150000, true, 'tiktok.com/@game_zone/v/21', 'syncing'),
  P('p4', 'a10', 'Instagram', 'Reels', '14.09.2024', '3 образа с показа, которые повторит каждая', 290, 342000, 21400, 1230, 3400, 5600, 6800, 310, 402000, 180000, true, 'instagram.com/reel/Cw82ka', 'ok'),
  P('p5', 'a6', 'Telegram', 'Пост', '12.09.2024', 'Заказал продукты — курьер приехал за 14 минут. Проверка!', 15, 52300, 2100, 380, 190, 940, 3200, 260, 214000, 40000, true, 't.me/foodie_moscow/1284', 'ok'),
  P('p6', 'a3', 'YouTube', 'Видео', '10.09.2024', 'Xiaomi 14T Pro — распаковка и первые тесты камеры Leica', 200, 615000, 38900, 3100, 1500, 6100, 15800, 830, 1120000, 240000, true, 'youtu.be/mi14t', 'waiting'),
  P('p7', 'a9', 'Instagram', 'Stories', '08.09.2024', 'Утренняя пробежка с Galaxy Watch 7 — замеры VO2 max', 130, 240000, 14200, 890, 540, 2900, 5400, 190, 187000, 90000, true, 'instagram.com/stories/fitpro', 'ok'),
  P('p8', 'a4', 'Instagram', 'Reels', '05.09.2024', 'Аэропорт → город за 24 часа: что в рюкзаке', 175, 189000, 11800, 720, 1600, 3200, 4600, 240, 296000, 120000, true, 'instagram.com/reel/Cv41xk', 'ok'),
  P('p9', 'a5', 'VK', 'Пост', '03.09.2024', '5 средств для купания малышей: честный разбор составов', 45, 74000, 3900, 510, 280, 1400, 1900, 150, 98000, 35000, false, 'vk.com/mama_blog?w=wall8821', 'error'),
  P('p10', 'a7', 'Telegram', 'Пост', '01.09.2024', 'ПП-подборка недели: 7 продуктов до 300 ₽', 95, 31000, 1400, 260, 150, 620, 1100, 95, 76000, 28000, false, 't.me/vegan_food/812', 'ok'),
  P('p11', 'a11', 'YouTube', 'Видео', '28.08.2024', 'Москва → Сочи на авто: Galaxy Tab как штурман', 225, 150000, 8200, 640, 310, 1700, 3600, 140, 154000, 110000, true, 'youtu.be/tabroad', 'ok'),
  P('p12', 'a8', 'Telegram', 'Пост', '25.08.2024', 'Разбор: стоит ли заходить в стейкинг сейчас', 260, 98000, 3100, 890, 420, 1100, 2400, 120, 132000, 70000, false, 't.me/crypto_talk/1024', 'waiting'),
  P('p13', 'a2', 'YouTube', 'Видео', '20.08.2024', 'Собрал умный дом за 100 000 ₽ — полный гайд', 210, 402000, 26400, 1980, 840, 3900, 9800, 510, 705000, 200000, true, 'youtu.be/smarthome100k', 'ok'),
  P('p14', 'a12', 'TikTok', 'Reels', '18.08.2024', 'Тест Poco X6 в Genshin на максималках', 350, 980000, 76000, 4200, 9800, 15600, 14200, 560, 486000, 75000, true, 'tiktok.com/@game_zone/v/18', 'ok'),
  P('p15', 'a10', 'Instagram', 'Пост', '15.08.2024', 'Базовый гардероб осени: 12 вещей', 290, 310000, 19800, 1050, 2800, 4800, 5200, 230, 342000, 140000, true, 'instagram.com/p/Cu91bk', 'syncing'),
  P('p16', 'a1', 'Instagram', 'Reels', '12.08.2024', 'До/после: 4 недели с Revitalift Филлер', 330, 128400, 9100, 580, 1200, 2400, 3800, 175, 208000, 45000, true, 'instagram.com/reel/Ct55zx', 'ok'),
  P('p17', 'a6', 'Telegram', 'Пост', '08.08.2024', 'Готовые ужины за 250 ₽: тестирую 5 сервисов', 15, 52000, 2300, 410, 210, 890, 2800, 210, 168000, 40000, true, 't.me/foodie_moscow/1198', 'ok'),
  P('p18', 'a3', 'YouTube', 'Видео', '02.08.2024', 'ТОП-7 гаджетов июля, которые реально удивили', 200, 512000, 33600, 2800, 1400, 5200, 13400, 690, 890000, 180000, true, 'youtu.be/top7july', 'ok'),
];

/* ================= ЗАМЕТКИ (ЭКРАН 11) ================= */
export const ANALYTICS_NOTES: Note[] = [
  { icon: '📈', text: '@tech_guru показал лучший ROAS (4.2x) в категории «Технологии». Рекомендую увеличить бюджет на 30%', by: 'Анна Соколова', tone: 'green' },
  { icon: '⚠️', text: '3 публикации без ERID-маркера обнаружены. Созданы задачи менеджерам', by: 'система', tone: 'red' },
  { icon: '💡', text: 'В Stories конверсия в 2,3 раза выше, чем в постах. Скорректируйте стратегию', by: 'Игорь Волков', tone: 'indigo' },
  { icon: '🎯', text: 'CPA по Telegram на 15% ниже, чем по Instagram', by: 'данные', tone: 'amber' },
];

/* ================= ИСТОЧНИКИ АНАЛИТИКИ ================= */
export const SOURCES_INIT: SourceItem[] = [
  { id: 's1', name: 'AppsFlyer', desc: 'Мобильная атрибуция', status: 'active', metrics: '1.2M событий · синхр. 5 мин назад', hue: 340, letter: 'AF' },
  { id: 's2', name: 'Яндекс Метрика', desc: 'Веб-аналитика', status: 'active', metrics: '860K сессий · синхр. 12 мин назад', hue: 45, letter: 'Я' },
  { id: 's3', name: 'Google Analytics 4', desc: 'Событийная аналитика', status: 'error', metrics: 'Токен истёк 12.09.2024', hue: 25, letter: 'GA' },
  { id: 's4', name: 'VK Реклама', desc: 'Статистика посевов', status: 'off', metrics: 'Не подключён', hue: 215, letter: 'VK' },
  { id: 's5', name: 'MyTracker', desc: 'Атрибуция Mail.ru', status: 'off', metrics: 'Не подключён', hue: 265, letter: 'MT' },
];

/* ================= ШАБЛОНЫ ДОГОВОРОВ ================= */
export const TEMPLATES: Template[] = [
  {
    id: 't1', name: 'Договор оказания услуг', badge: 'Рекомендуем',
    desc: 'Универсальный договор на создание рекламного контента с обязательной маркировкой по 38-ФЗ',
    preview: ['г. Москва', 'Исполнитель обязуется оказать Заказчику услуги по созданию и размещению рекламных материалов…', 'Оплата производится в течение 10 рабочих дней после подписания Акта оказанных услуг'],
  },
  {
    id: 't2', name: 'Авторский договор',
    desc: 'Передача исключительных прав на созданный рекламный контент',
    preview: ['Автор передаёт Заказчику исключительные права на материалы в полном объёме…', 'Вознаграждение составляет фиксированную сумму, указанную в п. 3.1'],
  },
  {
    id: 't3', name: 'Лицензионный договор',
    desc: 'Предоставление права использования контента на срок кампании',
    preview: ['Лицензиар предоставляет Лицензиату право использования на 6 месяцев…', 'Территория использования: Российская Федерация'],
  },
  {
    id: 't4', name: 'Договор с рекламным агентством',
    desc: 'Рамочный договор для работы через агентство-посредника',
    preview: ['Агентство действует от имени и по поручению Рекламодателя…', 'Комиссия Агентства составляет 10% от бюджета Кампании'],
  },
];

/* ================= УТИЛИТЫ ================= */
export const authorById = (id: string) => AUTHORS.find(a => a.id === id) ?? AUTHORS[0];
export const brandById = (id: string) => BRANDS.find(b => b.id === id) ?? BRANDS[0];
export const dealById = (deals: Deal[], id: string) => deals.find(d => d.id === id);
