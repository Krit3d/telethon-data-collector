import asyncio
import base64
import logging
import math
import struct

import httpx

from src.config.config import Settings

logger = logging.getLogger(__name__)

BUSINESS_ANCHORS: tuple[str, ...] = (
    "Ритейл и электронная коммерция: интернет-магазин, розничный магазин, бутик, каталог товаров, "
    "продажа одежды, обуви, техники, смартфонов. Цены, розница, опт, дропшиппинг, Kaspi Red, рассрочка, "
    "обмен, Trade-in, доставка по стране, накладний платіж, передоплата, відправка Новою Поштою, "
    "обмін та повернення, наявність. Retail shop, store, order online, shipping.",
    "Локальные услуги и заведения: салон красоты, лазерная эпиляция, косметология, барбершоп, "
    "детейлинг студия, автосервис, клининг, ремонт, ресторан, отель, база отдыха. Адрес заведения, "
    "филиалы, рынок, график работы, прайс-лист, бронирование, запись на процедуру, вакансии. "
    "Local service, salon, detailing, restaurant, booking.",
    "Производство и B2B: собственное производство, фабрика, изготовление под заказ, мебель, "
    "стройматериалы, декор, сантехника, двери, полифасад, власне виробництво, виготовлення меблів, "
    "співпраця, оптові поставки, працюємо офіційно, ФОП. Manufacturer, factory, custom production, "
    "wholesale B2B.",
)

CREATOR_ANCHORS: tuple[str, ...] = (
    "Лайфстайл и мысли: личный блог, персональная страница, авторский контент, лайфстайл блогер, "
    "инфлюенсер. Мои мысли, рассуждения, личный опыт, истории из жизни, семья, дети, путешествия, "
    "досуг, развлечения, вдохновение, юмор, подкаст. Особистий блог, моє життя, подорожі, думки, "
    "лайфстайл. Personal blog, lifestyle creator, thoughts, daily life.",
    "Практикующий специалист и творец: экспертный блог практикующего специалиста: врач, стоматолог, "
    "орнитолог, психолог, психотерапевт, преподаватель, репетитор, тренер, художник, арт-директор, "
    "фотограф, писатель. Личная практика, персональные консультации, приём пациентов, авторские "
    "методики, обучение, амбассадор проектов. Лікар, стоматолог, спеціаліст, консультації, приватна "
    "практика, мистецтво. Independent specialist, doctor, artist, expert, educator, consultant.",
)

COMMUNITY_ANCHORS: tuple[str, ...] = (
    "Тематическое сообщество, паблик, медиа, новостной портал, агрегатор новостей, городской паблик, "
    "афиша событий. Публикация новостей, дайджесты событий, статьи разных авторов, мемы, подборки "
    "материалов, полезные посты от редакции, информационный ресурс. Thematic community, public page, "
    "media outlet, news portal, event digest, news aggregator, public feed.",
    "Профессиональное комьюнити, бизнес-клуб, конференция, саммит, форум, хаб, комьюнити про "
    "искусственный интеллект, IT-сообщество, нетворкинг платформа. Сообщество единомышленников, клуб "
    "предпринимателей, площадка для общения, некоммерческая ассоциация, организаторы мероприятий, "
    "платформа развития. Community hub, business club, summit conference, networking platform, forum, "
    "tech community, association.",
)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class BusinessSemanticDetector:

    def __init__(self, settings: Settings, margin: float = 0.02) -> None:
        self.settings = settings
        self.margin = margin
        self._creator_vectors: list[list[float]] | None = None
        self._community_vectors: list[list[float]] | None = None
        self._business_vectors: list[list[float]] | None = None
        self._lock = asyncio.Lock()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.settings.cloud_ru_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.cloud_ru_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.settings.cloud_ru_embedding_model, "input": texts}
        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        vectors: list[list[float]] = []
        for item in data["data"]:
            embedding = item["embedding"]
            if isinstance(embedding, str):
                decoded = base64.b64decode(embedding)
                embedding = list(struct.unpack(f"{len(decoded) // 4}f", decoded))
            vectors.append(embedding)
        return vectors

    async def _ensure_anchors(self) -> None:
        if (
            self._business_vectors is not None
            and self._community_vectors is not None
            and self._creator_vectors is not None
        ):
            return
        async with self._lock:
            if (
                self._business_vectors is not None
                and self._community_vectors is not None
                and self._creator_vectors is not None
            ):
                return
            vectors = await self._fetch_embeddings(
                list(BUSINESS_ANCHORS) + list(COMMUNITY_ANCHORS) + list(CREATOR_ANCHORS)
            )
            biz_count = len(BUSINESS_ANCHORS)
            comm_count = len(COMMUNITY_ANCHORS)
            self._business_vectors = vectors[:biz_count]
            self._community_vectors = vectors[biz_count : biz_count + comm_count]
            self._creator_vectors = vectors[biz_count + comm_count :]

    async def classify(self, text: str) -> tuple[str, float, float, float]:
        if not text or not text.strip():
            return "parsed", 0.0, 0.0, 0.0
        text = text[:1500]
        await self._ensure_anchors()
        business_vectors = self._business_vectors
        community_vectors = self._community_vectors
        creator_vectors = self._creator_vectors
        assert (
            business_vectors is not None
            and community_vectors is not None
            and creator_vectors is not None
        )
        text_vector = (await self._fetch_embeddings([text]))[0]
        sim_biz = max(cosine_similarity(text_vector, bv) for bv in business_vectors)
        sim_comm = max(cosine_similarity(text_vector, mv) for mv in community_vectors)
        sim_creator = max(cosine_similarity(text_vector, cv) for cv in creator_vectors)
        if (sim_biz - sim_creator) > self.margin and sim_biz >= sim_comm:
            status = "business"
        elif (sim_comm - sim_creator) > self.margin and sim_comm > sim_biz:
            status = "community"
        else:
            status = "parsed"
        logger.debug(
            "Classification: status=%s sim_biz=%.4f sim_comm=%.4f sim_creator=%.4f",
            status,
            sim_biz,
            sim_comm,
            sim_creator,
        )
        return status, sim_biz, sim_comm, sim_creator

    async def evaluate(self, text: str) -> tuple[bool, float, float, float]:
        status, sim_biz, _, sim_creator = await self.classify(text)
        return status == "business", sim_biz, sim_creator, sim_biz - sim_creator

    async def is_business(self, text: str) -> bool:
        status, _, _, _ = await self.classify(text)
        return status == "business"

    async def close(self) -> None:
        await self.client.aclose()
