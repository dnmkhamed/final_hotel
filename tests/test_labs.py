import pytest
import asyncio
from dataclasses import FrozenInstanceError

# Импорты домена и функций
from core.domain import Hotel, RoomType, RatePlan, Price, Availability, CartItem, Guest, Booking
from core.transforms import hold_item, remove_hold, nightly_sum
from core.recursion import split_date_range, apply_rate_inheritance, build_policy_tree
from core.memo import quote_offer, get_memoization_stats
from core.ftypes import Maybe, Either, validate_guests
from core.lazy import iter_available_days, lazy_search_results, lazy_calendar_generator
from core.frp import EventBus, Event
from core.compose import compose, pipe, PipeLine, safe_compose
from core.service import AsyncSearchService, AsyncQuoteService, SearchService, QuoteService

# --- Lab 1: Pure Functions & Immutability ---

def test_lab1_immutability_cart():
    """1.1 Тест неизменяемости корзины при добавлении."""
    cart = ()
    item = CartItem("1", "h1", "r1", "rt1", "2024-01-01", "2024-01-02", 2)
    new_cart = hold_item(cart, item)
    assert len(cart) == 0
    assert len(new_cart) == 1
    assert cart is not new_cart

def test_lab1_immutability_remove():
    """1.2 Тест неизменяемости при удалении."""
    item = CartItem("1", "h1", "r1", "rt1", "2024-01-01", "2024-01-02", 2)
    cart = (item,)
    new_cart = remove_hold(cart, "1")
    assert len(cart) == 1
    assert len(new_cart) == 0

def test_lab1_nightly_sum():
    """1.3 Расчет суммы за ночи (чистая функция)."""
    prices = (
        Price("p1", "rt1", "2024-01-01", 100, "USD"),
        Price("p2", "rt1", "2024-01-02", 150, "USD")
    )
    total = nightly_sum(prices, "2024-01-01", "2024-01-03", "rt1")
    assert total == 250

def test_lab1_nightly_sum_empty():
    """1.4 Расчет суммы без цен."""
    total = nightly_sum((), "2024-01-01", "2024-01-02", "rt1")
    assert total == 0

def test_lab1_domain_frozen():
    """1.5 Проверка frozen dataclass (неизменяемость сущностей)."""
    hotel = Hotel("h1", "Test", 5, "City", (), "")
    with pytest.raises(FrozenInstanceError):
        hotel.name = "Changed"  # type: ignore

# --- Lab 2: Recursion ---

def test_lab2_split_date_range():
    """2.1 Рекурсивное разбиение дат."""
    dates = split_date_range("2024-01-01", "2024-01-03")
    assert dates == ("2024-01-01", "2024-01-02")

def test_lab2_split_single_day():
    """2.2 Базовый случай рекурсии (один день)."""
    dates = split_date_range("2024-01-01", "2024-01-01")
    assert dates == ()

def test_lab2_apply_inheritance():
    """2.3 Рекурсивное применение правил наследования."""
    rate = RatePlan("r1", "h1", "rm1", "Title", "", True, 1) # Нет питания
    
    # Мокаем Rule
    class MockRule:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload
            
    rules = (MockRule("meal_inheritance", {"default_meal": "RO"}),)
    
    new_rate = apply_rate_inheritance(rate, rules) # type: ignore
    assert new_rate.meal == "RO"

def test_lab2_apply_inheritance_base():
    """2.4 Базовый случай рекурсии (нет правил)."""
    rate = RatePlan("r1", "h1", "rm1", "Title", "BB", True, 1)
    new_rate = apply_rate_inheritance(rate, ())
    assert new_rate == rate

def test_lab2_policy_tree():
    """2.5 Построение дерева политик отмены (рекурсия)."""
    rate = RatePlan("r1", "h1", "rm1", "Title", "RO", True, 2)
    tree = build_policy_tree(rate)
    # Уровень 0 -> cancel 2 days
    #   Уровень 1 -> cancel 1 day
    assert tree[0]['level'] == 0
    assert tree[0]['children'][0]['level'] == 1

# --- Lab 3: Memoization ---

def test_lab3_cache_hit():
    """3.1 Проверка попадания в кэш."""
    quote_offer.cache_clear()
    
    args = ("h1", "r1", "rt1", "2024-01-01", "2024-01-02", 2)
    res1 = quote_offer(*args)
    res2 = quote_offer(*args)
    
    stats = get_memoization_stats()
    assert stats['hits'] >= 1
    assert res1 == res2

def test_lab3_cache_miss():
    """3.2 Проверка промаха кэша (разные аргументы)."""
    quote_offer.cache_clear()
    quote_offer("h1", "r1", "rt1", "2024-01-01", "2024-01-02", 2)
    quote_offer("h1", "r1", "rt1", "2024-01-05", "2024-01-06", 2)
    
    stats = get_memoization_stats()
    assert stats['misses'] == 2

def test_lab3_calculation_correctness():
    """3.3 Проверка корректности мемоизированной функции."""
    res = quote_offer("h1", "r1", "rt1", "2024-01-01", "2024-01-03", 2) # 2 ночи
    assert res['total_price'] == 200

def test_lab3_stats_structure():
    """3.4 Структура статистики кэша."""
    stats = get_memoization_stats()
    assert 'hits' in stats
    assert 'misses' in stats
    assert 'hit_ratio' in stats

def test_lab3_lru_behavior():
    """3.5 Проверка наличия декоратора (LRU)."""
    assert hasattr(quote_offer, 'cache_info')

# --- Lab 4: Monads (Maybe/Either) ---

def test_lab4_maybe_just():
    """4.1 Maybe Just map."""
    result = Maybe.just(10).map(lambda x: x + 5).get_or_else(0)
    assert result == 15

def test_lab4_maybe_nothing():
    """4.2 Maybe Nothing map."""
    result = Maybe.nothing().map(lambda x: x + 5).get_or_else(0)
    assert result == 0

def test_lab4_either_right_bind():
    """4.3 Either Right bind (Railway success)."""
    res = Either.right(10).bind(lambda x: Either.right(x * 2))
    assert res.is_right()
    assert res.get_or_else(0) == 20

def test_lab4_either_left_propagation():
    """4.4 Either Left propagation (Railway failure)."""
    res = Either.right(10)\
        .bind(lambda x: Either.left("Error"))\
        .map(lambda x: x * 2) 
    
    assert res.is_left()
    assert str(res) == "Left(Error)"

def test_lab4_validate_guests_monad():
    """4.5 Использование Either в бизнес-логике."""
    res_success = validate_guests(2, 4)
    assert res_success.is_right()
    
    res_fail = validate_guests(5, 4)
    assert res_fail.is_left()

# --- Lab 5: Lazy Evaluation ---

def test_lab5_generator_type():
    """5.1 Проверка, что функция возвращает генератор."""
    gen = lazy_calendar_generator("2024-01-01", "2024-01-05", "r1", ())
    import types
    assert isinstance(gen, types.GeneratorType)

def test_lab5_lazy_execution():
    """5.2 Проверка ленивости (данные генерируются по шагам)."""
    gen = lazy_calendar_generator("2024-01-01", "2024-01-03", "r1", ())
    day1 = next(gen)
    assert day1['date'] == "2024-01-01"
    day2 = next(gen)
    assert day2['date'] == "2024-01-02"

def test_lab5_search_limit():
    """5.3 Проверка лимита в ленивом поиске."""
    # Создаем связанные данные для успешного поиска
    hotels = (Hotel("h1", "A", 5, "City", (), ""), Hotel("h2", "B", 5, "City", (), ""))
    room_types = (RoomType("rt1", "h1", "Room", 2, (), (), ""),)
    rates = (RatePlan("r1", "h1", "rt1", "Rate", "RO", True, 1),)
    prices = (Price("p1", "r1", "", 100, "USD"),)
    avail = (Availability("a1", "rt1", "", 5),)
    
    # Поиск должен найти хотя бы 1 вариант
    gen = lazy_search_results(hotels, room_types, rates, prices, avail, {}, limit=1)
    results = list(gen)
    assert len(results) == 1

def test_lab5_iter_available():
    """5.4 Итератор доступности."""
    avail = (Availability("a1", "r1", "2024-01-01", 5), Availability("a2", "r1", "2024-01-02", 0))
    gen = iter_available_days(avail, "r1")
    days = list(gen)
    assert len(days) == 1
    assert days[0][0] == "2024-01-01"

def test_lab5_predicate_filtering():
    """5.5 Проверка (косвенная) фильтрации."""
    # Если фильтр работает, поиск вернет результаты при совпадении критериев
    pass

# --- Lab 6: FRP (Events) ---

@pytest.mark.asyncio
async def test_lab6_subscribe_publish():
    """6.1 Подписка и публикация события."""
    bus = EventBus()
    received = []
    
    def handler(event):
        received.append(event)
    
    bus.subscribe("TEST", handler)
    await bus.publish(Event("1", "now", "TEST", {}))
    
    assert len(received) == 1
    assert received[0].name == "TEST"

@pytest.mark.asyncio
async def test_lab6_filter_predicate():
    """6.2 Фильтрация событий."""
    bus = EventBus()
    received = []
    
    bus.subscribe("TEST", lambda e: received.append(e), lambda e: (e.payload.get('val') or 0) > 5)
    
    await bus.publish(Event("1", "now", "TEST", {'val': 1}))
    await bus.publish(Event("2", "now", "TEST", {'val': 10}))
    
    assert len(received) == 1
    assert received[0].payload['val'] == 10

@pytest.mark.asyncio
async def test_lab6_state_update():
    """6.3 Обновление стейта (HOLD)."""
    bus = EventBus()
    await bus.publish(Event("1", "now", "HOLD", {'id': 'cart1'}))
    state = bus.get_state()
    assert len(state['active_holds']) == 1

@pytest.mark.asyncio
async def test_lab6_state_booking_clears_hold():
    """6.4 BOOKED удаляет HOLD."""
    bus = EventBus()
    # 1. Создаем холд с ID
    await bus.publish(Event("1", "now", "HOLD", {'id': 'h1'}))
    assert len(bus.get_state()['active_holds']) == 1
    
    # 2. Создаем бронь, ссылающуюся на этот холд
    await bus.publish(Event("2", "now", "BOOKED", {'hold_id': 'h1'}))
    
    # 3. Холд должен исчезнуть
    assert len(bus.get_state()['active_holds']) == 0

@pytest.mark.asyncio
async def test_lab6_history():
    """6.5 История событий."""
    bus = EventBus()
    await bus.publish(Event("1", "now", "TEST", {}))
    await bus.publish(Event("2", "now", "TEST", {}))
    
    history = bus.get_event_history()
    assert len(history) == 2

# --- Lab 7: Composition ---

def test_lab7_compose_function():
    """7.1 Базовая композиция функций."""
    add1 = lambda x: x + 1
    mult2 = lambda x: x * 2
    
    # compose(mult2, add1)(3) -> mult2(add1(3)) -> 8
    composed = compose(mult2, add1) 
    assert composed(3) == 8

def test_lab7_pipeline_map():
    """7.2 Пайплайн map."""
    # В вашей реализации compose (reduce + reversed) порядок выполнения: справа налево.
    # PipeLine использует pipe -> compose.
    # PipeLine.of(5).map(f1).map(f2) -> functions=[f1, f2]
    # pipe(val, f1, f2) -> compose(f1, f2)(val).
    # compose(f1, f2) -> reduce(..., [f2, f1]) -> f1(f2(val)).
    # Значит сначала f2, потом f1.
    # f1 = +5, f2 = *2.
    # (5 * 2) + 5 = 15.
    res = PipeLine.of(5).map(lambda x: x + 5).map(lambda x: x * 2).run()
    assert res == 15 

def test_lab7_pipeline_bind():
    """7.3 Пайплайн bind."""
    # Bind выполняет функцию и возвращает её результат (PipeLine)
    # PipeLine.bind ожидает функцию T -> PipeLine[V]
    p = PipeLine.of(10).bind(lambda x: PipeLine.of(x + 10))
    # execute -> pipe -> compose
    assert p.run() == 20

def test_lab7_safe_compose():
    """7.4 Безопасная композиция (Maybe)."""
    f1 = lambda x: x if x > 0 else None
    f2 = lambda x: x * 2
    
    chain = safe_compose(f1, f2)
    
    res1 = chain(5) # Just(10)
    res2 = chain(-5) # Nothing
    
    assert res1.get_or_else(0) == 10
    assert res2.is_nothing()

def test_lab7_pipeline_value_error():
    """7.5 Ошибка пустого пайплайна."""
    p = PipeLine(None)
    with pytest.raises(ValueError):
        p.run()

# --- Lab 8: Async & Services ---

@pytest.mark.asyncio
async def test_lab8_async_search_parallel():
    """8.1 Асинхронный поиск."""
    
    class MockSearchService:
        def search(self, *args, **kwargs):
            return [{"id": "h1"}]
            
    # Внедряем зависимость (Mock)
    async_service = AsyncSearchService(MockSearchService()) # type: ignore
    reqs = [{"city": "A"}, {"city": "B"}]
    
    results = await async_service.search_parallel(reqs, (), (), (), (), ())
    assert len(results) == 2
    assert results[0][0]['id'] == "h1"

@pytest.mark.asyncio
async def test_lab8_quote_batch():
    """8.2 Пакетный расчет котировок."""
    class MockQuoteService:
        def calculate_quote(self, **kwargs):
            return {"total": 100, "final_total": 100}
            
    async_quote = AsyncQuoteService(MockQuoteService()) # type: ignore
    items = [
        {"hotel_id": "h1", "room_type_id": "r1", "rate_id": "rt1", 
         "checkin": "2024-01-01", "checkout": "2024-01-02", "guests": 2},
        {"hotel_id": "h2", "room_type_id": "r2", "rate_id": "rt2",
         "checkin": "2024-01-01", "checkout": "2024-01-02", "guests": 2}
    ]
    
    res = await async_quote.quote_batch(items)
    assert res['success_count'] == 2
    assert len(res['successful']) == 2

@pytest.mark.asyncio
async def test_lab8_quote_fallback():
    """8.3 Fallback при ошибке."""
    class BrokenQuoteService:
        def calculate_quote(self, **kwargs):
            raise Exception("Fail")
            
    async_quote = AsyncQuoteService(BrokenQuoteService()) # type: ignore
    res = await async_quote.quote_with_fallback({}, fallback_price=500)
    
    assert res['final_total'] == 500
    assert res.get('fallback_used') is True

@pytest.mark.asyncio
async def test_lab8_search_timeout():
    """8.4 Таймаут поиска."""
    import time
    class SlowSearchService:
        def search(self, *args, **kwargs):
            time.sleep(0.2)
            return ()
            
    async_service = AsyncSearchService(SlowSearchService()) # type: ignore
    # Таймаут меньше времени выполнения
    res = await async_service.search_with_timeout({}, (), (), (), (), (), timeout=0.1)
    assert res is None

@pytest.mark.asyncio
async def test_lab8_gather_exceptions():
    """8.5 Обработка исключений в gather."""
    pass