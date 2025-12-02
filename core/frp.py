from typing import Callable, Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from .domain import Event

@dataclass(frozen=True)
class Subscription:
    id: str
    event_type: str
    callback: Callable[[Event], Any]
    filter_predicate: Callable[[Event], bool] = None

class EventBus:
    """Улучшенная шина событий с поддержкой FRP"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Subscription]] = {}
        self._event_history: List[Event] = []
        self._state: Dict[str, Any] = {
            'active_holds': [],
            'recent_bookings': [],
            'price_changes': [],
            'search_queries': [],
            'cancellations': []
        }
    
    def subscribe(
        self, 
        event_type: str, 
        callback: Callable[[Event], Any],
        filter_predicate: Callable[[Event], bool] = None
    ) -> str:
        """Подписаться на тип событий с опциональным фильтром"""
        from uuid import uuid4
        sub_id = str(uuid4())
        
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        subscription = Subscription(sub_id, event_type, callback, filter_predicate)
        self._subscribers[event_type].append(subscription)
        return sub_id
    
    def unsubscribe(self, sub_id: str) -> bool:
        """Отписаться от событий"""
        for event_type, subscribers in self._subscribers.items():
            for i, sub in enumerate(subscribers):
                if sub.id == sub_id:
                    subscribers.pop(i)
                    return True
        return False
    
    async def publish(self, event: Event) -> None:
        """Опубликовать событие"""
        self._event_history.append(event)
        
        # Обновляем состояние на основе события
        await self._update_state(event)
        
        # Уведомляем подписчиков
        if event.name in self._subscribers:
            for subscriber in self._subscribers[event.name]:
                try:
                    # Применяем фильтр если есть
                    if subscriber.filter_predicate and not subscriber.filter_predicate(event):
                        continue
                    
                    if asyncio.iscoroutinefunction(subscriber.callback):
                        await subscriber.callback(event)
                    else:
                        subscriber.callback(event)
                except Exception as e:
                    print(f"Error in event subscriber {subscriber.id}: {e}")
    
    async def _update_state(self, event: Event) -> None:
        """Обновить состояние на основе события (чистая функция)"""
        if event.name == "HOLD":
            self._state['active_holds'].append({
                **event.payload,
                'event_id': event.id,
                'timestamp': event.ts
            })
        elif event.name == "BOOKED":
            self._state['recent_bookings'].append({
                **event.payload,
                'event_id': event.id,
                'timestamp': event.ts
            })
            # Удаляем соответствующий холд
            self._state['active_holds'] = [
                hold for hold in self._state['active_holds']
                if hold.get('id') != event.payload.get('hold_id')
            ]
        elif event.name == "CANCELLED":
            self._state['cancellations'].append({
                **event.payload,
                'event_id': event.id,
                'timestamp': event.ts
            })
        elif event.name == "PRICE_CHANGED":
            self._state['price_changes'].append({
                **event.payload,
                'event_id': event.id,
                'timestamp': event.ts
            })
        elif event.name == "SEARCH":
            self._state['search_queries'].append({
                **event.payload,
                'event_id': event.id,
                'timestamp': event.ts
            })
    
    def get_state(self) -> Dict[str, Any]:
        """Получить текущее состояние"""
        return self._state.copy()
    
    def get_event_history(self, limit: int = 100, event_type: str = None) -> Tuple[Event, ...]:
        """Получить историю событий"""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.name == event_type]
        return tuple(events[-limit:])
    
    def get_subscriber_count(self, event_type: str = None) -> int:
        """Получить количество подписчиков"""
        if event_type:
            return len(self._subscribers.get(event_type, []))
        return sum(len(subs) for subs in self._subscribers.values())

# Глобальный экземпляр шины событий
event_bus = EventBus()

# Реактивные обработчики событий
async def update_search_analytics(event: Event) -> None:
    """Обновить аналитику поиска"""
    print(f"Search analytics updated: {event.payload}")

async def update_booking_dashboard(event: Event) -> None:
    """Обновить дашборд бронирований"""
    print(f"Booking dashboard updated: {event.payload}")

async def notify_price_alerts(event: Event) -> None:
    """Уведомить о изменениях цен"""
    if event.name == "PRICE_CHANGED":
        print(f"Price alert: {event.payload}")

async def update_availability_cache(event: Event) -> None:
    """Обновить кэш доступности"""
    if event.name in ["BOOKED", "CANCELLED"]:
        print(f"Availability cache updated for: {event.payload}")

# Регистрируем обработчики
event_bus.subscribe("SEARCH", update_search_analytics)
event_bus.subscribe("HOLD", update_booking_dashboard)
event_bus.subscribe("BOOKED", update_booking_dashboard)
event_bus.subscribe("CANCELLED", update_booking_dashboard)
event_bus.subscribe("PRICE_CHANGED", notify_price_alerts)
event_bus.subscribe("BOOKED", update_availability_cache)
event_bus.subscribe("CANCELLED", update_availability_cache)

# Функции для создания событий
async def create_search_event(filters: Dict[str, Any]) -> None:
    """Создать событие поиска"""
    from uuid import uuid4
    event = Event(
        id=str(uuid4()),
        ts=datetime.now().isoformat(),
        name="SEARCH",
        payload=filters
    )
    await event_bus.publish(event)

async def create_hold_event(item: Dict[str, Any]) -> None:
    """Создать событие холда"""
    from uuid import uuid4
    event = Event(
        id=str(uuid4()),
        ts=datetime.now().isoformat(),
        name="HOLD",
        payload=item
    )
    await event_bus.publish(event)

async def create_booking_event(booking: Dict[str, Any]) -> None:
    """Создать событие бронирования"""
    from uuid import uuid4
    event = Event(
        id=str(uuid4()),
        ts=datetime.now().isoformat(),
        name="BOOKED",
        payload=booking
    )
    await event_bus.publish(event)

async def create_cancellation_event(booking_id: str, reason: str = "") -> None:
    """Создать событие отмены"""
    from uuid import uuid4
    event = Event(
        id=str(uuid4()),
        ts=datetime.now().isoformat(),
        name="CANCELLED",
        payload={"booking_id": booking_id, "reason": reason}
    )
    await event_bus.publish(event)

async def create_price_change_event(rate_id: str, old_price: int, new_price: int) -> None:
    """Создать событие изменения цены"""
    from uuid import uuid4
    event = Event(
        id=str(uuid4()),
        ts=datetime.now().isoformat(),
        name="PRICE_CHANGED",
        payload={
            "rate_id": rate_id,
            "old_price": old_price,
            "new_price": new_price,
            "change_percent": ((new_price - old_price) / old_price) * 100
        }
    )
    await event_bus.publish(event)








publish_search_event = create_search_event
publish_hold_event = create_hold_event  
publish_booked_event = create_booking_event
publish_cancelled_event = create_cancellation_event
publish_price_changed_event = create_price_change_event