from typing import Iterator, Callable, Tuple, Any, Optional
from .domain import Hotel, RoomType, RatePlan, Price, Availability
from .recursion import split_date_range
from .memo import quote_offer
from .ftypes import Maybe, Either

def iter_available_days(avail: Tuple[Availability, ...], room_type_id: str) -> Iterator[Tuple[str, int]]:
    """Ленивый генератор доступных дней для типа номера"""
    for availability in avail:
        if availability.room_type_id == room_type_id and availability.available > 0:
            yield (availability.date, availability.available)

def lazy_offers(
    hotels: Tuple[Hotel, ...],
    room_types: Tuple[RoomType, ...],
    rates: Tuple[RatePlan, ...],
    prices: Tuple[Price, ...],
    availability: Tuple[Availability, ...],
    predicate: Callable[[Hotel, RoomType, RatePlan, int], bool]
) -> Iterator[Tuple[Hotel, RoomType, RatePlan, int, bool]]:
    """
    Ленивый генератор предложений отелей.
    Возвращает (отель, тип номера, тариф, цена за ночь, доступен)
    """
    
    # Создаем индексы для быстрого поиска
    room_types_by_hotel = {}
    for room_type in room_types:
        if room_type.hotel_id not in room_types_by_hotel:
            room_types_by_hotel[room_type.hotel_id] = []
        room_types_by_hotel[room_type.hotel_id].append(room_type)
    
    rates_by_room_type = {}
    for rate in rates:
        if rate.room_type_id not in rates_by_room_type:
            rates_by_room_type[rate.room_type_id] = []
        rates_by_room_type[rate.room_type_id].append(rate)
    
    prices_by_rate = {}
    for price in prices:
        if price.rate_id not in prices_by_rate:
            prices_by_rate[price.rate_id] = []
        prices_by_rate[price.rate_id].append(price)
    
    availability_by_room_type = {}
    for avail in availability:
        if avail.room_type_id not in availability_by_room_type:
            availability_by_room_type[avail.room_type_id] = []
        availability_by_room_type[avail.room_type_id].append(avail)
    
    # Лениво генерируем предложения
    for hotel in hotels:
        hotel_rooms = room_types_by_hotel.get(hotel.id, [])
        
        for room_type in hotel_rooms:
            room_rates = rates_by_room_type.get(room_type.id, [])
            
            for rate in room_rates:
                # Получаем цену (берем первую доступную)
                rate_prices = prices_by_rate.get(rate.id, [])
                price_amount = rate_prices[0].amount if rate_prices else 100  # цена по умолчанию
                
                # Проверяем доступность
                room_availability = availability_by_room_type.get(room_type.id, [])
                is_available = any(avail.available > 0 for avail in room_availability)
                
                # Применяем предикат
                if predicate(hotel, room_type, rate, price_amount):
                    yield (hotel, room_type, rate, price_amount, is_available)

# core/lazy.py

def lazy_search_results(
    hotels: Tuple[Hotel, ...],
    room_types: Tuple[RoomType, ...],
    rates: Tuple[RatePlan, ...],
    prices: Tuple[Price, ...],
    availability: Tuple[Availability, ...],
    filters: dict,
    limit: Optional[int] = None
) -> Iterator[dict]:
    """Ленивый генератор результатов поиска с фильтрами"""
    
    def search_predicate(hotel: Hotel, room_type: RoomType, rate: RatePlan, price: int) -> bool:
        """Предикат для фильтрации предложений"""
        # Фильтр по городу (УЛУЧШЕНО: частичное совпадение и удаление пробелов)
        if filters.get('city'):
            search_query = filters['city'].strip().lower()
            hotel_city = hotel.city.lower()
            # Ищем вхождение строки (например, "new" найдет "New York")
            if search_query not in hotel_city:
                return False
        
        # Фильтр по звездам
        if filters.get('min_stars') and hotel.stars < filters['min_stars']:
            return False
        
        # Фильтр по вместимости
        if filters.get('guests') and room_type.capacity < filters['guests']:
            return False
        
        # Фильтр по цене
        if filters.get('max_price') and price > filters['max_price']:
            return False
        
        # Фильтр по features
        if filters.get('features'):
            required_features = set(filters['features'])
            hotel_features = set(hotel.features)
            room_features = set(room_type.features)
            if not required_features.issubset(hotel_features.union(room_features)):
                return False
        
        return True
    
    # Запускаем ленивый генератор предложений
    offers = lazy_offers(hotels, room_types, rates, prices, availability, search_predicate)
    
    count = 0
    for hotel, room_type, rate, price, available in offers:
        if limit and count >= limit:
            return
        
        yield {
            'hotel': hotel,
            'room_type': room_type,
            'rate': rate,
            'price_per_night': price,
            'available': available,
            'total_price': price * 3  # упрощенный расчет
        }
        count += 1

def lazy_calendar_generator(
    start_date: str,
    end_date: str,
    room_type_id: str,
    availability: Tuple[Availability, ...]
) -> Iterator[dict]:
    """Ленивый генератор календаря доступности"""
    dates = split_date_range(start_date, end_date)
    
    for date in dates:
        available_count = 0
        for avail in availability:
            if avail.room_type_id == room_type_id and avail.date == date:
                available_count = avail.available
                break
        
        yield {
            'date': date,
            'available': available_count,
            'room_type_id': room_type_id
        }