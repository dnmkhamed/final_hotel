from typing import Tuple, Callable, Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from .compose import compose, pipe, either_compose, PipeLine
from .domain import Hotel, RoomType, RatePlan, CartItem, Booking, SearchFilters
from .ftypes import Maybe, Either, validate_booking, validate_cart_item
from .lazy import lazy_search_results
from .memo import quote_offer
from .recursion import filter_hotels, by_city, by_capacity, by_features, by_price_range
from .frp import event_bus, create_booking_event, create_hold_event


@dataclass(frozen=True)
class SearchService:
    """Сервис поиска отелей - композиция чистых функций"""
    
    filters: Tuple[Callable, ...]
    scorers: Tuple[Callable, ...]
    
    def search(self, 
               hotels: Tuple[Hotel, ...],
               room_types: Tuple[RoomType, ...],
               rates: Tuple[RatePlan, ...],
               prices: Tuple[Any, ...],
               availability: Tuple[Any, ...],
               search_request: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
        """Поиск отелей - только композиция чистых функций"""
        
        # Применяем фильтры через композицию
        def apply_filters(result: Dict[str, Any]) -> bool:
            return all(filter_func(result) for filter_func in self.filters)
        
        # Ленивый поиск с применением фильтров
        results = list(lazy_search_results(
            hotels, room_types, rates, prices, availability,
            search_request,
            limit=search_request.get('limit', 50)
        ))
        
        # Фильтруем результаты
        filtered_results = [r for r in results if apply_filters(r)]
        
        # Применяем скореры для сортировки
        if self.scorers:
            for scorer in self.scorers:
                filtered_results.sort(key=scorer, reverse=True)
        
        return tuple(filtered_results)
    
    def search_with_filters(self,
                          hotels: Tuple[Hotel, ...],
                          search_filters: SearchFilters) -> Tuple[Hotel, ...]:
        """Поиск отелей с использованием функциональных фильтров"""
        return filter_hotels(hotels, search_filters)


@dataclass(frozen=True)
class QuoteService:
    """Сервис расчета стоимости - только композиции"""
    
    price_calculator: Callable
    tax_calculator: Callable
    discount_calculator: Callable
    
    def calculate_quote(self, 
                       hotel_id: str,
                       room_type_id: str,
                       rate_id: str,
                       checkin: str,
                       checkout: str,
                       guests: int) -> Dict[str, Any]:
        """Расчет стоимости через композицию функций"""
        
        # Используем мемоизированную функцию для базовой цены
        base_quote = quote_offer(
            hotel_id, room_type_id, rate_id, checkin, checkout, guests
        )
        
        # Применяем калькуляторы через пайплайн
        final_quote = pipe(
            base_quote,
            self.price_calculator,
            self.tax_calculator,
            self.discount_calculator
        )
        
        return final_quote
    
    def calculate_batch_quotes(self, quotes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Пакетный расчет котировок через map"""
        return list(map(lambda quote: self.calculate_quote(**quote), quotes))
    
    def safe_calculate_quote(self,
                           hotel_id: str,
                           room_type_id: str,
                           rate_id: str,
                           checkin: str,
                           checkout: str,
                           guests: int) -> Either[str, Dict[str, Any]]:
        """Безопасный расчет стоимости с использованием Either"""
        
        def validate_inputs() -> Either[str, Dict[str, Any]]:
            """Валидация входных данных"""
            if guests <= 0:
                return Either.left("Number of guests must be positive")
            if checkout <= checkin:
                return Either.left("Checkout date must be after checkin")
            
            return Either.right({
                'hotel_id': hotel_id,
                'room_type_id': room_type_id,
                'rate_id': rate_id,
                'checkin': checkin,
                'checkout': checkout,
                'guests': guests
            })
        
        def calculate_quote(params: Dict[str, Any]) -> Either[str, Dict[str, Any]]:
            """Расчет котировки"""
            try:
                quote = self.calculate_quote(**params)
                return Either.right(quote)
            except Exception as e:
                return Either.left(f"Calculation failed: {str(e)}")
        
        # Композиция операций с Either
        return validate_inputs().bind(calculate_quote)


@dataclass(frozen=True)
class BookingService:
    """Сервис бронирования - композиции валидаторов и квотеров"""
    
    validators: Tuple[Callable, ...]
    quoter: Callable
    finalizer: Callable
    
    def create_booking(self, 
                      guest: Dict[str, Any],
                      cart_items: Tuple[CartItem, ...],
                      prices: Tuple[Any, ...],
                      availability: Tuple[Any, ...],
                      rules: Tuple[Any, ...]) -> Either[str, Booking]:
        """Создание бронирования через функциональный пайплайн"""
        
        # Создаем пайплайн валидации и подтверждения
        booking_pipeline = (
            PipeLine.of({
                'guest': guest,
                'items': cart_items,
                'prices': prices,
                'availability': availability,
                'rules': rules
            })
            .map(self._validate_booking)
            .map(self._calculate_totals)
            .map(self._finalize_booking)
        )
        
        try:
            result = booking_pipeline.run()
            return Either.right(result)
        except Exception as e:
            return Either.left(str(e))
    
    def hold_booking(self,
                    guest: Dict[str, Any],
                    cart_items: Tuple[CartItem, ...],
                    prices: Tuple[Any, ...],
                    availability: Tuple[Any, ...]) -> Either[str, Dict[str, Any]]:
        """Холд бронирования с валидацией"""
        
        def validate_hold(hold_data: Dict[str, Any]) -> Either[str, Dict[str, Any]]:
            """Валидация холда"""
            for item in hold_data['items']:
                validation_result = validate_cart_item(
                    item, hold_data['availability'], hold_data.get('rules', ())
                )
                if validation_result.is_left():
                    return validation_result.map_error(lambda e: f"Item validation failed: {e}")
            
            return Either.right(hold_data)
        
        def calculate_hold_totals(hold_data: Dict[str, Any]) -> Either[str, Dict[str, Any]]:
            """Расчет стоимости холда"""
            try:
                total = sum(
                    self.quoter(item, hold_data['prices']) 
                    for item in hold_data['items']
                )
                return Either.right({**hold_data, 'total': total, 'status': 'held'})
            except Exception as e:
                return Either.left(f"Calculation failed: {str(e)}")
        
        hold_data = {
            'guest': guest,
            'items': cart_items,
            'prices': prices,
            'availability': availability,
            'hold_id': f"hold_{uuid4().hex[:8]}",
            'created_at': datetime.now().isoformat()
        }
        
        # Композиция операций холда
        return (
            Either.right(hold_data)
            .bind(validate_hold)
            .bind(calculate_hold_totals)
        )
    
    def _validate_booking(self, booking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация бронирования через композицию валидаторов"""
        for validator in self.validators:
            result = validator(booking_data)
            if result.is_left():
                raise ValueError(result.get_error())
        return booking_data
    
    def _calculate_totals(self, booking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет итоговой стоимости"""
        total = sum(
            self.quoter(item, booking_data['prices']) 
            for item in booking_data['items']
        )
        return {**booking_data, 'total': total}
    
    def _finalize_booking(self, booking_data: Dict[str, Any]) -> Booking:
        """Финальное подтверждение бронирования"""
        return self.finalizer(booking_data)


# Фабрики сервисов - Dependency Injection
def create_search_service() -> SearchService:
    """Фабрика для создания сервиса поиска"""
    
    def city_filter(result: Dict[str, Any]) -> bool:
        """Фильтр по городу"""
        city = result.get('search_city')
        if not city:
            return True
        return result['hotel'].city.lower() == city.lower()
    
    def capacity_filter(result: Dict[str, Any]) -> bool:
        """Фильтр по вместимости"""
        guests = result.get('search_guests', 1)
        return result['room_type'].capacity >= guests
    
    def price_filter(result: Dict[str, Any]) -> bool:
        """Фильтр по цене"""
        max_price = result.get('search_max_price')
        if not max_price:
            return True
        return result['price_per_night'] <= max_price
    
    def price_scorer(result: Dict[str, Any]) -> float:
        """Скорер по цене (ниже цена - выше рейтинг)"""
        return -result['price_per_night']
    
    def rating_scorer(result: Dict[str, Any]) -> float:
        """Скорер по рейтингу отеля"""
        return result['hotel'].stars
    
    def availability_scorer(result: Dict[str, Any]) -> float:
        """Скорер по доступности"""
        return 1.0 if result['available'] else 0.0
    
    return SearchService(
        filters=(
            city_filter,
            capacity_filter,
            price_filter,
        ),
        scorers=(
            price_scorer,
            rating_scorer,
            availability_scorer,
        )
    )


def create_quote_service() -> QuoteService:
    """Фабрика для создания сервиса котировок"""
    
    def calculate_base_price(quote: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет базовой цены"""
        # В реальном приложении здесь сложная логика расчета
        nights = quote.get('nights', 1)
        price_per_night = quote.get('base_price_per_night', 100)
        base_price = nights * price_per_night * quote.get('guests', 1)
        return {**quote, 'base_price': base_price}
    
    def calculate_taxes(quote: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет налогов"""
        base_price = quote.get('base_price', 0)
        
        # Налог на проживание (10%)
        occupancy_tax = base_price * 0.1
        
        # Городской налог (5%)
        city_tax = base_price * 0.05
        
        total_tax = occupancy_tax + city_tax
        total_with_tax = base_price + total_tax
        
        return {
            **quote, 
            'occupancy_tax': occupancy_tax,
            'city_tax': city_tax,
            'total_tax': total_tax,
            'total_with_tax': total_with_tax
        }
    
    def apply_discounts(quote: Dict[str, Any]) -> Dict[str, Any]:
        """Применение скидок"""
        total = quote.get('total_with_tax', 0)
        
        # Скидка 5% за раннее бронирование
        from datetime import datetime
        checkin_str = quote.get('checkin')
        if checkin_str:
            checkin = datetime.strptime(checkin_str, '%Y-%m-%d')
            today = datetime.now()
            days_until_checkin = (checkin - today).days
            
            if days_until_checkin > 30:
                total *= 0.95
                quote = {**quote, 'early_booking_discount': 0.05}
        
        # Скидка 10% за длительное проживание
        nights = quote.get('nights', 1)
        if nights > 7:
            total *= 0.90
            quote = {**quote, 'long_stay_discount': 0.10}
        
        return {
            **quote,
            'final_total': round(total, 2),
            'currency': 'USD'
        }
    
    return QuoteService(
        price_calculator=calculate_base_price,
        tax_calculator=calculate_taxes,
        discount_calculator=apply_discounts
    )


def create_booking_service() -> BookingService:
    """Фабрика для создания сервиса бронирования"""
    
    def validate_guest_data(booking_data: Dict[str, Any]) -> Either[str, Dict[str, Any]]:
        """Валидация данных гостя"""
        guest = booking_data.get('guest', {})
        
        if not guest.get('name'):
            return Either.left("Guest name is required")
        
        if not guest.get('email'):
            return Either.left("Guest email is required")
        
        # Простая валидация email
        if '@' not in guest.get('email', ''):
            return Either.left("Invalid email format")
        
        return Either.right(booking_data)
    
    def validate_availability(booking_data: Dict[str, Any]) -> Either[str, Dict[str, Any]]:
        """Валидация доступности номеров"""
        for item in booking_data.get('items', []):
            # Упрощенная проверка доступности
            # В реальном приложении здесь была бы проверка в базе данных
            if hasattr(item, 'room_type_id') and item.room_type_id.startswith('sold_out'):
                return Either.left(f"Room {item.room_type_id} is not available")
        
        return Either.right(booking_data)
    
    def calculate_item_price(item: CartItem, prices: Tuple[Any, ...]) -> int:
        """Упрощенный расчет цены для элемента"""
        # В реальном приложении здесь сложная логика расчета
        nights = 3  # упрощенный расчет
        return item.guests * 100 * nights
    
    def finalize_booking(booking_data: Dict[str, Any]) -> Booking:
        """Финальное создание бронирования"""
        return Booking(
            id=f"booking_{uuid4().hex[:8]}",
            guest_id=booking_data['guest']['id'],
            items=booking_data['items'],
            total=booking_data['total'],
            status='confirmed',
            created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    return BookingService(
        validators=(
            validate_guest_data,
            validate_availability,
        ),
        quoter=calculate_item_price,
        finalizer=finalize_booking
    )


# Асинхронные сервисы для лабораторной работы 8
class AsyncSearchService:
    """Асинхронный сервис поиска"""
    
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
    
    async def search_parallel(self, 
                            requests: List[Dict[str, Any]],
                            hotels: Tuple[Hotel, ...],
                            room_types: Tuple[RoomType, ...],
                            rates: Tuple[RatePlan, ...],
                            prices: Tuple[Any, ...],
                            availability: Tuple[Any, ...]) -> List[List[Dict[str, Any]]]:
        """Параллельный поиск по нескольким запросам"""
        
        async def execute_search(request: Dict[str, Any]):
            """Выполнить один поисковый запрос"""
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    self.search_service.search,
                    hotels, room_types, rates, prices, availability, request
                )
            return result
        
        # Запускаем все поисковые запросы параллельно
        tasks = [execute_search(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Фильтруем исключения
        return [r for r in results if not isinstance(r, Exception)]
    
    async def search_with_timeout(self,
                                request: Dict[str, Any],
                                hotels: Tuple[Hotel, ...],
                                room_types: Tuple[RoomType, ...],
                                rates: Tuple[RatePlan, ...],
                                prices: Tuple[Any, ...],
                                availability: Tuple[Any, ...],
                                timeout: float = 5.0) -> Optional[List[Dict[str, Any]]]:
        """Поиск с таймаутом"""
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    self.search_service.search,
                    hotels, room_types, rates, prices, availability, request
                ),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None


class AsyncQuoteService:
    """Асинхронный сервис котировок"""
    
    def __init__(self, quote_service: QuoteService):
        self.quote_service = quote_service
    
    async def quote_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Пакетный расчет котировок"""
        
        async def calculate_single_quote(item: Dict[str, Any]):
            """Расчет одной котировки"""
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                # Используем lambda, чтобы передать kwargs, так как run_in_executor принимает только *args
                result = await loop.run_in_executor(
                    executor,
                    lambda: self.quote_service.calculate_quote(**item)
                )
            return result
        
        # Запускаем расчеты параллельно
        tasks = [calculate_single_quote(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Группируем результаты
        successful_quotes = []
        failed_quotes = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_quotes.append({
                    'item': items[i],
                    'error': str(result)
                })
            else:
                successful_quotes.append(result)
        
        return {
            'successful': successful_quotes,
            'failed': failed_quotes,
            'total_count': len(items),
            'success_count': len(successful_quotes),
            'failure_count': len(failed_quotes)
        }
    
    async def quote_with_fallback(self,
                                item: Dict[str, Any],
                                fallback_price: int = 100) -> Dict[str, Any]:
        """Расчет котировки с fallback"""
        try:
            # Используем lambda, чтобы передать kwargs
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.quote_service.calculate_quote(**item)
            )
        except Exception:
            # Fallback: базовая цена
            return {
                **item,
                'base_price': fallback_price,
                'total_with_tax': fallback_price,
                'final_total': fallback_price,
                'currency': 'USD',
                'fallback_used': True
            }


# Интеграционный сервис для сквозного workflow
class IntegrationService:
    """Сервис сквозной интеграции - объединяет все компоненты"""
    
    def __init__(self, 
                 search_service: SearchService,
                 quote_service: QuoteService, 
                 booking_service: BookingService):
        self.search_service = search_service
        self.quote_service = quote_service
        self.booking_service = booking_service
        
        # Асинхронные версии сервисов
        self.async_search = AsyncSearchService(search_service)
        self.async_quote = AsyncQuoteService(quote_service)
    
    async def end_to_end_workflow(self, 
                                search_request: Dict[str, Any],
                                guest_data: Dict[str, Any],
                                hotels: Tuple[Hotel, ...],
                                room_types: Tuple[RoomType, ...],
                                rates: Tuple[RatePlan, ...],
                                prices: Tuple[Any, ...],
                                availability: Tuple[Any, ...],
                                rules: Tuple[Any, ...]) -> Dict[str, Any]:
        """Сквозной workflow от поиска до бронирования"""
        
        # 1. Поиск отелей
        search_results = self.search_service.search(
            hotels, room_types, rates, prices, availability, search_request
        )
        
        if not search_results:
            return {
                'success': False, 
                'error': 'No hotels found matching your criteria',
                'step': 'search'
            }
        
        # 2. Расчет котировок для топ-3 результатов
        top_results = search_results[:3]
        quote_items = [
            {
                'hotel_id': result['hotel'].id,
                'room_type_id': result['room_type'].id,
                'rate_id': result['rate'].id,
                'checkin': search_request.get('checkin', '2024-01-15'),
                'checkout': search_request.get('checkout', '2024-01-18'),
                'guests': search_request.get('guests', 2),
                'nights': 3,  # упрощенный расчет
                'base_price_per_night': result['price_per_night']
            }
            for result in top_results
        ]
        
        # Асинхронный расчет котировок
        quote_result = await self.async_quote.quote_batch(quote_items)
        
        if not quote_result['successful']:
            return {
                'success': False,
                'error': 'Failed to calculate quotes',
                'step': 'quotation',
                'failed_quotes': quote_result['failed']
            }
        
        # 3. Создание элементов корзины
        cart_items = []
        for i, (result, quote) in enumerate(zip(top_results, quote_result['successful'])):
            cart_item = CartItem(
                id=f"cart_{uuid4().hex[:8]}",
                hotel_id=result['hotel'].id,
                room_type_id=result['room_type'].id,
                rate_id=result['rate'].id,
                checkin=search_request.get('checkin', '2024-01-15'),
                checkout=search_request.get('checkout', '2024-01-18'),
                guests=search_request.get('guests', 2)
            )
            cart_items.append(cart_item)
            
            # Публикуем событие холда
            await create_hold_event({
                'cart_item_id': cart_item.id,
                'hotel_id': cart_item.hotel_id,
                'room_type_id': cart_item.room_type_id,
                'checkin': cart_item.checkin,
                'checkout': cart_item.checkout,
                'guests': cart_item.guests
            })
        
        # 4. Создание бронирования
        booking_result = self.booking_service.create_booking(
            guest_data,
            tuple(cart_items),
            prices,
            availability,
            rules
        )
        
        if booking_result.is_right():
            booking = booking_result.get_value()
            
            # Публикуем событие бронирования
            await create_booking_event({
                'booking_id': booking.id,
                'guest_id': booking.guest_id,
                'total': booking.total,
                'items_count': len(booking.items),
                'status': booking.status
            })
            
            return {
                'success': True,
                'booking_id': booking.id,
                'total': booking.total,
                'status': booking.status,
                'search_results_count': len(search_results),
                'items_booked': len(booking.items),
                'quotes_calculated': quote_result['success_count']
            }
        else:
            return {
                'success': False,
                'error': booking_result.get_error(),
                'step': 'booking',
                'search_results_count': len(search_results),
                'quotes_calculated': quote_result['success_count']
            }
    
    async def parallel_search_workflow(self,
                                     search_requests: List[Dict[str, Any]],
                                     hotels: Tuple[Hotel, ...],
                                     room_types: Tuple[RoomType, ...],
                                     rates: Tuple[RatePlan, ...],
                                     prices: Tuple[Any, ...],
                                     availability: Tuple[Any, ...]) -> Dict[str, Any]:
        """Параллельный workflow поиска по нескольким запросам"""
        
        # Параллельный поиск
        search_results = await self.async_search.search_parallel(
            search_requests,
            hotels, room_types, rates, prices, availability
        )
        
        # Агрегация результатов
        all_results = []
        for results in search_results:
            all_results.extend(results)
        
        # Уникальные отели (по ID)
        unique_hotels = {}
        for result in all_results:
            hotel_id = result['hotel'].id
            if hotel_id not in unique_hotels:
                unique_hotels[hotel_id] = result
        
        return {
            'total_searches': len(search_requests),
            'total_results': len(all_results),
            'unique_hotels': len(unique_hotels),
            'results_per_search': [len(r) for r in search_results],
            'unique_results': list(unique_hotels.values())
        }


# Фабрика для интеграционного сервиса
def create_integration_service() -> IntegrationService:
    """Создать сервис сквозной интеграции"""
    return IntegrationService(
        search_service=create_search_service(),
        quote_service=create_quote_service(),
        booking_service=create_booking_service()
    )


# Утилиты для работы с сервисами
def create_service_registry() -> Dict[str, Any]:
    """Создать реестр сервисов"""
    return {
        'search': create_search_service(),
        'quote': create_quote_service(),
        'booking': create_booking_service(),
        'integration': create_integration_service(),
        'async_search': AsyncSearchService(create_search_service()),
        'async_quote': AsyncQuoteService(create_quote_service())
    }


# Пример использования композиции в доменной области
def create_booking_workflow() -> Callable[[Dict[str, Any]], Either[str, Booking]]:
    """Создать workflow бронирования через композицию"""
    
    def prepare_booking_data(request: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка данных бронирования"""
        return {
            'guest': request['guest'],
            'items': tuple(CartItem(**item) for item in request['items']),
            'search_params': request.get('search_params', {}),
            'metadata': {
                'source': request.get('source', 'web'),
                'user_agent': request.get('user_agent', ''),
                'ip_address': request.get('ip_address', '')
            }
        }
    
    def enrich_with_pricing(booking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обогащение данными о ценах"""
        quote_service = create_quote_service()
        
        priced_items = []
        for item in booking_data['items']:
            quote = quote_service.calculate_quote(
                item.hotel_id,
                item.room_type_id,
                item.rate_id,
                item.checkin,
                item.checkout,
                item.guests
            )
            priced_items.append({
                'item': item,
                'quote': quote
            })
        
        return {
            **booking_data,
            'priced_items': priced_items,
            'total_quotes': sum(q['quote']['final_total'] for q in priced_items)
        }
    
    def finalize_booking_creation(booking_data: Dict[str, Any]) -> Booking:
        """Финальное создание бронирования"""
        booking_service = create_booking_service()
        
        result = booking_service.create_booking(
            booking_data['guest'],
            booking_data['items'],
            (),  # prices would come from external source
            (),  # availability would come from external source  
            ()   # rules would come from external source
        )
        
        if result.is_right():
            return result.get_value()
        else:
            raise ValueError(result.get_error())
    
    # Композиция workflow
    workflow = compose(
        prepare_booking_data,
        enrich_with_pricing,
        finalize_booking_creation
    )
    
    # Обертка с Either для обработки ошибок
    def safe_workflow(request: Dict[str, Any]) -> Either[str, Booking]:
        try:
            result = workflow(request)
            return Either.right(result)
        except Exception as e:
            return Either.left(str(e))
    
    return safe_workflow