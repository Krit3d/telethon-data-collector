from typing import Any

PLATFORM_PROFILE_LINKS: dict[str, str] = {
    "INSTAGRAM": "https://instagram.com/{username}",
    "TIKTOK": "https://www.tiktok.com/@{username}",
    "YOUTUBE": "https://youtube.com/@{username}",
    "THREADS": "https://threads.net/@{username}",
}

LINK_IN_BIO_DOMAINS: frozenset[str] = frozenset({
    "linktr.ee", "taplink.cc", "taplink.ws", "beacons.ai", "msha.ke",
    "solo.to", "lu.ma", "lnk.bio", "campsite.bio", "carrd.co",
    "linkin.bio", "bio.link", "linkbio.co", "my.link", "hipolink.net",
})

SOCIAL_MEDIA_DOMAINS: frozenset[str] = frozenset({
    "instagram.com", "tiktok.com", "youtube.com", "youtu.be",
    "threads.net", "threads.com", "facebook.com", "twitter.com", "x.com",
    "linkedin.com", "t.me", "telegram.me", "telegram.dog", "wa.me",
    "whatsapp.com", "vk.com", "vk.ru", "vkontakte.ru", "dzen.ru",
    "zen.yandex.ru", "zen.yandex.com", "rutube.ru", "ok.ru",
    "odnoklassniki.ru",
})

GENERIC_CATEGORIES: frozenset[str] = frozenset({
    "personal blog", "blogger", "public figure", "none",
    "community", "creator", "digital creator",
})

DEFAULT_CONTEXT_WINDOW: int = 60
AT_MARKER_WINDOW: int = 100

BARE_TELEGRAM_TRIGGERS: frozenset[str] = frozenset((
    "тг", "тгк", "телега", "телеграм", "telegram", "tg",
))

TELEGRAM_SERVICE_TOKENS: frozenset[str] = frozenset((
    "канал", "channel",
))

TELEGRAM_PLATFORM_MARKERS: frozenset[str] = frozenset((
    "тгк", "тг", "телега", "телеграм", "telegram", "канал", "channel", "tg",
    "сотрудничество", "реклама", "рекламе", "pr", "ads", "manager",
    "менеджер", "связь", "контакты", "пишите", "collabs", "cooperation",
))

TELEGRAM_BLACKLISTED_HANDLES: frozenset[str] = frozenset((
    "direct", "instagram", "false", "true", "status", "active", "username",
    "comment", "channel", "admin", "today", "share", "profile", "media",
    "posts", "reels", "stories", "highlights",
))

COMMERCIAL_EXACT: frozenset[str] = frozenset((
    "pr", "ad", "ads", "dm", "order", "sales", "booking", "direct", "директ", "заказ",
))

COMMERCIAL_PREFIXES: frozenset[str] = frozenset((
    "реклам", "сотруднич", "менеджер", "коммерч", "бронир", "интеграц",
    "маркетинг", "marketing", "продвижен", "букинг", "commercial",
    "collab", "cooperat", "manager", "promot", "sponsor",
))

PLATFORM_HANDLE_RULES: dict[str, dict[str, Any]] = {
    "youtube": {
        "ignore_hosts": ("youtu.be",),
        "system_sections": frozenset(("watch", "shorts", "live", "playlist", "feed")),
        "prefix_handlers": {"c": 1, "channel": 1, "user": 1},
        "allow_at_prefix": True,
    },
    "vk": {
        "system_sections": frozenset((
            "wall", "video", "clip", "story", "album", "topic", "photo", "feed",
            "im", "messages", "friends", "groups", "photos", "videos", "audios",
            "docs", "settings", "apps", "bookmarks", "support",
        )),
        "content_prefix_pattern": r"^(?:wall|video|clip|story|album|topic|photo)-?\d+",
    },
    "ok": {
        "system_sections": frozenset(),
        "prefix_handlers": {"profile": 1, "group": 1, "messages": 1},
    },
    "threads": {
        "system_sections": frozenset(("t", "post")),
    },
    "dzen": {
        "system_sections": frozenset(("a", "video", "media", "suite")),
        "prefix_handlers": {"id": 1},
    },
    "tiktok": {
        "ignore_hosts": ("vm.tiktok.com",),
        "system_sections": frozenset(("t", "video", "tag", "live", "foryou", "discover")),
    },
    "rutube": {
        "system_sections": frozenset(("video", "play", "pl")),
        "prefix_handlers": {"channel": 1, "u": 1},
    },
}

COMMERCIAL_PHRASES: frozenset[str] = frozenset((
    "пишите сюда", "пишите на почту", "писать по рекламе", "писать сюда",
    "почта для", "по вопросам", "for business", "work with me",
))

PERSONAL_EXACT: frozenset[str] = frozenset((
    "me", "мой", "моя", "автор", "life", "лс", "личка", "пишите", "связь",
))

PERSONAL_PREFIXES: frozenset[str] = frozenset((
    "личн", "жизн", "дневник", "официальн", "personal", "author", "official",
    "личк", "напиши",
))

PERSONAL_PHRASES: frozenset[str] = frozenset((
    "обо мне", "моя жизнь", "мой лайф", "about me",
    "писать сюда", "писать в лс", "для связи", "обращаться в лс",
))

CHANNEL_EXACT: frozenset[str] = frozenset((
    "тгк", "канал", "телега", "блог", "blog", "news",
))

CHANNEL_PREFIXES: frozenset[str] = frozenset((
    "паблик", "сообществ", "новост", "подпис", "channel", "community",
))

CHANNEL_PHRASES: frozenset[str] = frozenset((
    "мой канал", "в канале", "наш канал", "подписывайся", "подпишись",
))

PHONE_TRIGGER_WORDS: frozenset[str] = frozenset((
    "whatsapp", "ватсап", "вотсап", "wa", "телефон", "тел", "номер",
    "звонить", "phone", "call", "contact",
))

WHATSAPP_TRIGGER_WORDS: frozenset[str] = frozenset((
    "whatsapp", "ватсап", "вотсап", "wa",
))

__all__ = [
    "PLATFORM_PROFILE_LINKS", "LINK_IN_BIO_DOMAINS", "SOCIAL_MEDIA_DOMAINS",
    "GENERIC_CATEGORIES", "DEFAULT_CONTEXT_WINDOW", "AT_MARKER_WINDOW",
    "BARE_TELEGRAM_TRIGGERS", "TELEGRAM_SERVICE_TOKENS",
    "TELEGRAM_PLATFORM_MARKERS", "TELEGRAM_BLACKLISTED_HANDLES",
    "COMMERCIAL_EXACT", "COMMERCIAL_PREFIXES", "COMMERCIAL_PHRASES",
    "PERSONAL_EXACT", "PERSONAL_PREFIXES", "PERSONAL_PHRASES",
    "CHANNEL_EXACT", "CHANNEL_PREFIXES", "CHANNEL_PHRASES",
    "PHONE_TRIGGER_WORDS", "WHATSAPP_TRIGGER_WORDS",
    "PLATFORM_HANDLE_RULES",
]