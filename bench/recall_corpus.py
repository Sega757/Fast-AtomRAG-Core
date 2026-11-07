"""Deterministic stress corpus for multi-hop recall (RU/EN engineering incidents).

15 chains (9 two-hop, 6 three-hop). In every chain the shared entity is written in DIFFERENT
inflected forms in adjacent atoms, and the question names the start entity in yet another form.
The target (last hop) shares no content-word stems with the question (checked in tests).
30 distractors (2 per chain) are packed with question words; one per chain is also linked
to the start entity, so it competes in the graph channel too.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chain:
    name: str
    question: str
    start_entity: str  # canonical form as written in the first hop
    hops: tuple[tuple[str, tuple[str, ...]], ...]  # (text, entities) per hop; last one is the target
    distractors: tuple[tuple[str, tuple[str, ...]], ...]


CHAINS: tuple[Chain, ...] = (
    Chain(
        "ru_balancer_2hop",
        "Как в итоге починили причину отказа балансировщика?",
        "отказ балансировщика",
        (
            (
                "Отказ балансировщика вызвала утечка дескрипторов в модуле Ingress.",
                ("отказ балансировщика", "утечки дескрипторов"),
            ),
            ("Утечку дескрипторов устранили введением лимита пула соединений.", ("утечка дескрипторов", "лимит пула")),
        ),
        (
            (
                "Причину отказа балансировщика обсуждали на встрече: как в итоге починили, так и не решили.",
                ("отказы балансировщика",),
            ),
            ("Как починили балансировщик в итоге? Причина отказа балансировщика осталась загадкой.", ()),
        ),
    ),
    Chain(
        "ru_replication_3hop",
        "Что сделали с корнем проблемы, вызванной задержкой репликации?",
        "задержка репликации",
        (
            (
                "Задержка репликации привела к устаревшим данным в кэше каталога.",
                ("задержка репликации", "устаревшие данные кэша"),
            ),
            (
                "Устаревшие данные кэша появлялись из-за слишком длинного TTL записей.",
                ("устаревших данных кэша", "TTL записей"),
            ),
            ("TTL записи сократили до тридцати секунд в конфигурации сервиса.", ("TTL записи",)),
        ),
        (
            ("Проблема с задержкой репликации: что сделали с корнем проблемы, пока неясно.", ("задержки репликации",)),
            ("Корень проблемы задержки репликации вызванной нагрузкой сделали темой доклада.", ()),
        ),
    ),
    Chain(
        "ru_disk_2hop",
        "Чем закончилась история с переполнением диска?",
        "переполнение диска",
        (
            (
                "Переполнение диска на узле базы вызвали бесконтрольные журналы аудита.",
                ("переполнение диска", "журналы аудита"),
            ),
            ("Для журналов аудита включили ротацию по размеру и сжатие.", ("журналов аудита", "ротация")),
        ),
        (
            (
                "История с переполнением диска закончилась долгим разбором, чем всё закончилось — неясно.",
                ("переполнения диска",),
            ),
            ("Чем опасно переполнение диска и чем закончилась прошлая история — в вики.", ()),
        ),
    ),
    Chain(
        "ru_memory_3hop",
        "Что исправили в итоге по проблеме утечки памяти воркера?",
        "утечка памяти воркера",
        (
            ("Утечку памяти воркера вызывал незакрытый поток событий.", ("утечки памяти воркера", "поток событий")),
            (
                "Поток событий оставался открытым из-за ошибки в обработчике отмены.",
                ("потока событий", "обработчик отмены"),
            ),
            ("Обработчик отмены переписали с гарантированным освобождением ресурсов.", ("обработчика отмены",)),
        ),
        (
            ("Проблема утечки памяти воркера: что исправили в итоге, выясним на ретро.", ("утечкой памяти воркера",)),
            ("В итоге проблему утечки памяти у воркера отложили, ничего не исправили.", ()),
        ),
    ),
    Chain(
        "ru_certificate_2hop",
        "Как вернули работу после истории с просроченным сертификатом?",
        "просроченный сертификат",
        (
            (
                "Просроченный сертификат шлюза оборвал TLS-сессии партнёров.",
                ("просроченный сертификат", "TLS-сессии партнёров"),
            ),
            ("TLS-сессий партнёров восстановили через перевыпуск ключей в HSM.", ("TLS-сессий партнёров", "HSM")),
        ),
        (
            (
                "История с просроченным сертификатом: как вернули работу, расскажет дежурный.",
                ("просроченного сертификата",),
            ),
            ("Как вернули работу после истории с сертификатом, который был просрочен, — вопрос открыт.", ()),
        ),
    ),
    Chain(
        "ru_queue_3hop",
        "Чем лечили рост очереди заказов?",
        "рост очереди заказов",
        (
            (
                "Рост очереди заказов спровоцировал медленный консьюмер биллинга.",
                ("рост очереди заказов", "консьюмер биллинга"),
            ),
            (
                "Консьюмер биллинга тормозил на синхронных запросах к налоговому API.",
                ("консьюмера биллинга", "налоговое API"),
            ),
            ("Налоговое API начали вызывать асинхронно с локальным кэшем ставок.", ("налогового API", "кэш ставок")),
        ),
        (
            (
                "Рост очереди заказов лечили перезапуском, но чем лечили по-настоящему — неизвестно.",
                ("роста очереди заказов",),
            ),
            ("Чем лечили рост очереди заказов в прошлом квартале — смотри отчёт.", ()),
        ),
    ),
    Chain(
        "ru_dns_2hop",
        "Что предприняли после сбоя DNS?",
        "сбой DNS",
        (
            (
                "Сбой DNS в зоне eu-west возник из-за истёкшего делегирования домена.",
                ("сбой DNS", "делегирование домена"),
            ),
            (
                "Делегирования домена продлили и поставили мониторинг регистратора.",
                ("делегирования домена", "мониторинг регистратора"),
            ),
        ),
        (
            ("Что предприняли после сбоя DNS, обсудим на планёрке после релиза.", ("сбоя DNS",)),
            ("После сбоя DNS ничего не предприняли, сбой DNS повторился.", ()),
        ),
    ),
    Chain(
        "ru_build_3hop",
        "Как разобрались с падением сборки?",
        "падение сборки",
        (
            (
                "Падение сборки фронтенда вызвало обновление транзитивной зависимости.",
                ("падение сборки", "транзитивная зависимость"),
            ),
            (
                "Транзитивную зависимость подтянул незакреплённый диапазон версий в манифесте.",
                ("транзитивной зависимости", "диапазон версий"),
            ),
            (
                "Диапазона версий больше нет: стоит точный пин, lock-файл хранится в репозитории.",
                ("диапазона версий", "lock-файл"),
            ),
        ),
        (
            ("С падением сборки разобрались частично, как разобрались — описано кратко.", ("падением сборки",)),
            ("Как разобрались с падением сборки бэкенда, никто не помнит.", ()),
        ),
    ),
    Chain(
        "ru_timeouts_2hop",
        "Что помогло против таймаутов платёжного шлюза?",
        "таймауты платёжного шлюза",
        (
            (
                "Таймауты платёжного шлюза росли из-за исчерпания пула потоков.",
                ("таймауты платёжного шлюза", "исчерпание пула потоков"),
            ),
            (
                "Исчерпание пула потоков убрали, вынеся блокирующий ввод-вывод в отдельный экзекьютор.",
                ("исчерпания пула потоков", "отдельный экзекьютор"),
            ),
        ),
        (
            (
                "Против таймаутов платёжного шлюза помогло бы что-то, но что помогло — не записали.",
                ("таймаутов платёжного шлюза",),
            ),
            ("Что помогло против таймаутов шлюза платежей у соседней команды — неизвестно.", ()),
        ),
    ),
    Chain(
        "ru_search_2hop",
        "Как исправили деградацию поиска?",
        "деградация поиска",
        (
            (
                "Деградация поиска по каталогу совпала с ростом числа сегментов индекса.",
                ("деградация поиска", "сегменты индекса"),
            ),
            ("Сегментов индекса стало меньше после ночного задания force-merge.", ("сегментов индекса", "force-merge")),
        ),
        (
            ("Деградацию поиска исправили? Как исправили деградацию поиска — спросить у DBA.", ("деградации поиска",)),
            ("Как исправили деградацию поиска в мобильном приложении — отдельная история.", ()),
        ),
    ),
    Chain(
        "en_stampede_2hop",
        "How did the team finally fix the cache stampedes?",
        "cache stampede",
        (
            (
                "The cache stampede was triggered by synchronized key expirations.",
                ("cache stampede", "synchronized key expirations"),
            ),
            (
                "Synchronized key expiration was eliminated by adding random jitter to TTLs.",
                ("synchronized key expiration", "random jitter"),
            ),
        ),
        (
            ("The team discussed how to finally fix cache stampedes but did not fix them.", ("cache stampedes",)),
            ("How did the team fix the cache? The team finally fixed the stampede dashboard.", ()),
        ),
    ),
    Chain(
        "en_login_3hop",
        "What resolved the login outage in the end?",
        "login outage",
        (
            ("The login outage started when the session store ran out of memory.", ("login outage", "session store")),
            ("Session stores filled up because tokens never expired after logout.", ("session stores", "token expiry")),
            (
                "Token expiry now revokes refresh tokens on logout and caps lifetime at one day.",
                ("token expiries",),
            ),
        ),
        (
            ("What resolved the login outages in the end is still being investigated.", ("login outages",)),
            ("In the end the login outage was resolved... or was it? What resolved it remains unclear.", ()),
        ),
    ),
    Chain(
        "en_checkout_2hop",
        "How was the slow checkout eventually fixed?",
        "slow checkout",
        (
            ("Slow checkout pages were caused by an N+1 query in the pricing service.", ("slow checkout", "N+1 query")),
            ("The N+1 queries were replaced with a single batched join.", ("N+1 queries", "batched join")),
        ),
        (
            ("The slow checkout was eventually fixed, but how it was fixed is undocumented.", ("slow checkouts",)),
            ("How the slow checkout got eventually fixed is a question for the payments team.", ()),
        ),
    ),
    Chain(
        "en_pipeline_3hop",
        "What fixed the data pipeline delays?",
        "data pipeline delay",
        (
            (
                "The data pipeline delay came from a stuck Kafka consumer group.",
                ("data pipeline delay", "Kafka consumer group"),
            ),
            (
                "The Kafka consumer groups stalled on a poison message with a malformed schema.",
                ("Kafka consumer groups", "poison message"),
            ),
            (
                "Poison messages are now routed to a dead letter queue after three retries.",
                ("poison messages", "dead letter queue"),
            ),
        ),
        (
            (
                "What fixed the data pipeline delays? Nothing fixed the delays in the data pipeline yet.",
                ("data pipeline delays",),
            ),
            ("Data pipeline delays fixed: what fixed them is listed in the pipeline wiki.", ()),
        ),
    ),
    Chain(
        "en_rotation_2hop",
        "How did we stop the certificate rotation failures?",
        "certificate rotation failure",
        (
            (
                "Certificate rotation failures were caused by an expired service account token.",
                ("certificate rotation failure", "service account token"),
            ),
            (
                "The service account tokens are now issued by a workload identity provider.",
                ("service account tokens", "workload identity"),
            ),
        ),
        (
            (
                "How did we stop certificate rotation failures? We did not stop the failures.",
                ("certificate rotation failures",),
            ),
            ("We stopped the rotation of certificates; the failures did stop, how is unclear.", ()),
        ),
    ),
)

STOPWORDS = frozenset(
    {
        "как",
        "что",
        "чем",
        "для",
        "был",
        "была",
        "было",
        "были",
        "это",
        "или",
        "над",
        "под",
        "при",
        "про",
        "так",
        "все",
        "его",
        "она",
        "они",
        "мне",
        "нас",
        "вас",
        "the",
        "and",
        "was",
        "were",
        "how",
        "what",
        "why",
        "did",
        "does",
        "with",
        "from",
        "into",
        "that",
        "this",
        "then",
        "than",
        "for",
        "are",
        "not",
        "now",
    }
)


def corpus() -> list[tuple[int, str, tuple[str, ...]]]:
    """(atom_id, text, entities). Chain c, hop h -> 1000*c + h; distractors -> 1000*c + 500 + d."""
    rows: list[tuple[int, str, tuple[str, ...]]] = []
    for c, chain in enumerate(CHAINS, start=1):
        for h, (text, ents) in enumerate(chain.hops, start=1):
            rows.append((1000 * c + h, text, ents))
        for d, (text, ents) in enumerate(chain.distractors, start=1):
            rows.append((1000 * c + 500 + d, text, ents))
    return rows


def target_id(chain_index: int, chain: Chain) -> int:
    return 1000 * chain_index + len(chain.hops)


def chain_ids(chain_index: int, chain: Chain) -> list[int]:
    return [1000 * chain_index + h for h in range(1, len(chain.hops) + 1)]
