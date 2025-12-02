from typing import Callable, Any, TypeVar, Generic, List
from functools import reduce
from datetime import datetime
from .ftypes import Maybe, Either

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

def compose(*functions: Callable) -> Callable:
    """Композиция функций: compose(f, g, h)(x) = f(g(h(x)))"""
    def composed(arg: Any) -> Any:
        return reduce(lambda acc, f: f(acc), reversed(functions), arg)
    return composed

def pipe(value: T, *functions: Callable[..., U]) -> U:
    """Пайплайн: pipe(x, f, g, h) = h(g(f(x)))"""
    return compose(*functions)(value)

def safe_compose(*functions: Callable) -> Callable[[T], Maybe[U]]:
    """Безопасная композиция с использованием Maybe"""
    def composed(arg: T) -> Maybe[U]:
        result = Maybe.just(arg)
        for func in functions:
            result = result.bind(lambda x: Maybe.from_nullable(func(x)))
        return result
    return composed

def either_compose(*functions: Callable) -> Callable[[T], Either[str, U]]:
    """Композиция с использованием Either для обработки ошибок"""
    def composed(arg: T) -> Either[str, U]:
        result = Either.right(arg)
        for func in functions:
            result = result.bind(lambda x: Either.try_except(lambda: func(x), Exception))
        return result
    return composed

class PipeLine(Generic[T, U]):
    """Класс для построения пайплайнов"""
    
    def __init__(self, value: T = None):
        self._value = value
        self._functions: List[Callable] = []
    
    @classmethod
    def of(cls, value: T) -> 'PipeLine[T, T]':
        return cls(value)
    
    def map(self, func: Callable[[T], U]) -> 'PipeLine[T, U]':
        """Применить функцию к значению"""
        self._functions.append(func)
        return self
    
    def bind(self, func: Callable[[T], 'PipeLine[U, V]']) -> 'PipeLine[T, V]':
        """Монадический bind"""
        def binder(x: T) -> V:
            return func(x)._execute()
        self._functions.append(binder)
        return self
    
    def _execute(self) -> Any:
        """Выполнить пайплайн"""
        if self._value is None:
            raise ValueError("No value to execute pipeline")
        return pipe(self._value, *self._functions)
    
    def run(self) -> U:
        """Запустить пайплайн"""
        return self._execute()

# Примеры использования композиции в доменной области
def create_booking_pipeline() -> Callable[[dict], Either[str, dict]]:
    """Создать пайплайн бронирования"""
    return either_compose(
        validate_booking_dates,
        validate_guest_count,
        check_room_availability,
        calculate_total_price,
        apply_discounts,
        create_booking_record
    )

def validate_booking_dates(booking: dict) -> dict:
    """Валидация дат бронирования"""
    from datetime import datetime
    checkin = datetime.strptime(booking['checkin'], '%Y-%m-%d')
    checkout = datetime.strptime(booking['checkout'], '%Y-%m-%d')
    
    if checkout <= checkin:
        raise ValueError("Checkout date must be after checkin date")
    
    if (checkout - checkin).days > 30:
        raise ValueError("Maximum stay is 30 days")
    
    return booking

def validate_guest_count(booking: dict) -> dict:
    """Валидация количества гостей"""
    if booking['guests'] <= 0:
        raise ValueError("Number of guests must be positive")
    
    if booking['guests'] > 10:
        raise ValueError("Maximum 10 guests per booking")
    
    return booking

def check_room_availability(booking: dict) -> dict:
    """Проверка доступности номера"""
    # В реальном приложении здесь была бы проверка в базе данных
    if booking.get('room_type_id') == 'room_sold_out':
        raise ValueError("Room not available for selected dates")
    
    return {**booking, 'available': True}

def calculate_total_price(booking: dict) -> dict:
    """Расчет общей стоимости"""
    from datetime import datetime
    checkin = datetime.strptime(booking['checkin'], '%Y-%m-%d')
    checkout = datetime.strptime(booking['checkout'], '%Y-%m-%d')
    nights = (checkout - checkin).days
    price_per_night = booking.get('price_per_night', 100)
    total = nights * price_per_night * booking['guests']
    return {**booking, 'total_price': total, 'nights': nights}

def apply_discounts(booking: dict) -> dict:
    """Применение скидок"""
    total = booking['total_price']
    
    # Скидка 10% за раннее бронирование
    from datetime import datetime
    checkin = datetime.strptime(booking['checkin'], '%Y-%m-%d')
    today = datetime.now()
    
    if (checkin - today).days > 30:
        total *= 0.9
        booking = {**booking, 'early_booking_discount': 0.1}
    
    return {**booking, 'final_price': total}

def create_booking_record(booking: dict) -> dict:
    """Создание записи бронирования"""
    from uuid import uuid4
    from datetime import datetime
    return {
        **booking,
        'booking_id': f"BK_{uuid4().hex[:8].upper()}",
        'status': 'confirmed',
        'created_at': datetime.now().isoformat()
    }