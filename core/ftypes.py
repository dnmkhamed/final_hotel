from typing import TypeVar, Generic, Callable, Any, Union
from dataclasses import dataclass
import json

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')


class Maybe(Generic[T]):
    """Maybe monad for handling optional values."""
    
    @classmethod
    def just(cls, value: T) -> 'Maybe[T]':
        return _Just(value)
    
    @classmethod
    def nothing(cls) -> 'Maybe[T]':
        return _Nothing()
    
    @classmethod
    def from_nullable(cls, value: T) -> 'Maybe[T]':
        """Create Maybe from nullable value."""
        return cls.just(value) if value is not None else cls.nothing()
    
    def map(self, func: Callable[[T], U]) -> 'Maybe[U]':
        raise NotImplementedError
    
    def bind(self, func: Callable[[T], 'Maybe[U]']) -> 'Maybe[U]':
        raise NotImplementedError
    
    def get_or_else(self, default: T) -> T:
        raise NotImplementedError
    
    def is_just(self) -> bool:
        raise NotImplementedError
    
    def is_nothing(self) -> bool:
        raise NotImplementedError
    
    def to_either(self, error_msg: str) -> 'Either[str, T]':
        """Convert Maybe to Either."""
        if self.is_just():
            return Either.right(self.get_or_else(None))
        else:
            return Either.left(error_msg)


class _Just(Maybe[T]):
    def __init__(self, value: T):
        self._value = value
    
    def map(self, func: Callable[[T], U]) -> 'Maybe[U]':
        return Maybe.just(func(self._value))
    
    def bind(self, func: Callable[[T], Maybe[U]]) -> 'Maybe[U]':
        return func(self._value)
    
    def get_or_else(self, default: T) -> T:
        return self._value
    
    def is_just(self) -> bool:
        return True
    
    def is_nothing(self) -> bool:
        return False
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Just) and self._value == other._value
    
    def __str__(self) -> str:
        return f"Just({self._value})"


class _Nothing(Maybe[T]):
    def map(self, func: Callable[[T], U]) -> 'Maybe[U]':
        return Maybe.nothing()
    
    def bind(self, func: Callable[[T], Maybe[U]]) -> 'Maybe[U]':
        return Maybe.nothing()
    
    def get_or_else(self, default: T) -> T:
        return default
    
    def is_just(self) -> bool:
        return False
    
    def is_nothing(self) -> bool:
        return True
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Nothing)
    
    def __str__(self) -> str:
        return "Nothing"


class Either(Generic[E, T]):
    """Either monad for handling computations that may fail."""
    
    @classmethod
    def right(cls, value: T) -> 'Either[E, T]':
        return _Right(value)
    
    @classmethod
    def left(cls, error: E) -> 'Either[E, T]':
        return _Left(error)
    
    @classmethod
    def try_except(cls, func: Callable[[], T], error_type: type = Exception) -> 'Either[str, T]':
        """Create Either from function that might raise exception."""
        try:
            return cls.right(func())
        except error_type as e:
            return cls.left(str(e))
    
    def map(self, func: Callable[[T], U]) -> 'Either[E, U]':
        raise NotImplementedError
    
    def bind(self, func: Callable[[T], 'Either[E, U]']) -> 'Either[E, U]':
        raise NotImplementedError
    
    def get_or_else(self, default: T) -> T:
        raise NotImplementedError
    
    def get_value(self) -> T:
        """Get the value if Right, raise if Left."""
        raise NotImplementedError
    
    def get_error(self) -> E:
        """Get the error if Left, raise if Right."""
        raise NotImplementedError
    
    def is_right(self) -> bool:
        raise NotImplementedError
    
    def is_left(self) -> bool:
        return not self.is_right()
    
    def map_error(self, func: Callable[[E], E]) -> 'Either[E, T]':
        """Map over the error value."""
        raise NotImplementedError


class _Right(Either[Any, T]):
    def __init__(self, value: T):
        self._value = value
    
    def map(self, func: Callable[[T], U]) -> 'Either[Any, U]':
        return Either.right(func(self._value))
    
    def bind(self, func: Callable[[T], Either[Any, U]]) -> 'Either[Any, U]':
        return func(self._value)
    
    def get_or_else(self, default: T) -> T:
        return self._value
    
    def get_value(self) -> T:
        return self._value
    
    def get_error(self) -> Any:
        raise ValueError("Cannot get error from Right")
    
    def is_right(self) -> bool:
        return True
    
    def map_error(self, func: Callable[[Any], Any]) -> 'Either[Any, T]':
        return self
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Right) and self._value == other._value
    
    def __str__(self) -> str:
        return f"Right({self._value})"


class _Left(Either[E, Any]):
    def __init__(self, error: E):
        self._error = error
    
    def map(self, func: Callable[[Any], Any]) -> 'Either[E, Any]':
        return Either.left(self._error)
    
    def bind(self, func: Callable[[Any], Either[E, Any]]) -> 'Either[E, Any]':
        return Either.left(self._error)
    
    def get_or_else(self, default: Any) -> Any:
        return default
    
    def get_value(self) -> Any:
        raise ValueError("Cannot get value from Left")
    
    def get_error(self) -> E:
        return self._error
    
    def is_right(self) -> bool:
        return False
    
    def map_error(self, func: Callable[[E], E]) -> 'Either[E, Any]':
        return Either.left(func(self._error))
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Left) and self._error == other._error
    
    def __str__(self) -> str:
        return f"Left({self._error})"


# Domain-specific functions using Maybe/Either
def safe_rate(rates: tuple, rate_id: str) -> Maybe:
    """Safely find rate by ID using Maybe monad."""
    for rate in rates:
        if rate.id == rate_id:
            return Maybe.just(rate)
    return Maybe.nothing()


def safe_room_lookup(room_types: tuple, room_id: str) -> Maybe:
    """Safely find room by ID using Maybe monad."""
    for room in room_types:
        if room.id == room_id:
            return Maybe.just(room)
    return Maybe.nothing()


def validate_dates(checkin: str, checkout: str) -> Either[str, tuple]:
    """Validate date range using Either monad."""
    from datetime import datetime
    
    try:
        checkin_date = datetime.strptime(checkin, '%Y-%m-%d')
        checkout_date = datetime.strptime(checkout, '%Y-%m-%d')
        
        if checkout_date <= checkin_date:
            return Either.left("Checkout date must be after checkin date")
        
        if (checkout_date - checkin_date).days > 30:
            return Either.left("Maximum stay is 30 days")
            
        return Either.right((checkin, checkout))
    
    except ValueError:
        return Either.left("Invalid date format. Use YYYY-MM-DD")


def validate_guests(guests: int, room_capacity: int) -> Either[str, int]:
    """Validate guest count against room capacity."""
    if guests <= 0:
        return Either.left("Number of guests must be positive")
    
    if guests > room_capacity:
        return Either.left(f"Room capacity exceeded. Maximum: {room_capacity} guests")
    
    return Either.right(guests)


def validate_cart_item(item, availability: tuple, rules: tuple) -> Either[str, Any]:
    """Validate cart item using Either monad."""
    # Check availability
    available = any(
        avail.room_type_id == item.room_type_id and 
        avail.date >= item.checkin and 
        avail.date < item.checkout and 
        avail.available > 0
        for avail in availability
    ) if availability else True  # If no availability data, assume available
    
    if not available:
        return Either.left("No availability for selected dates")
    
    # Check guest capacity (simplified)
    if item.guests > 4:  # Assuming max capacity
        return Either.left("Guest count exceeds maximum room capacity")
    
    return Either.right(item)


def validate_booking(booking, prices: tuple, availability: tuple, rules: tuple) -> Either[str, Any]:
    """Validate complete booking using Either monad."""
    
    def validate_item(item):
        return validate_cart_item(item, availability, rules)
    
    # Validate all items using bind (railway oriented programming)
    validation_result = Either.right(booking)
    
    for item in booking.items:
        validation_result = validation_result.bind(
            lambda b, item=item: validate_item(item).map(lambda _: b)
        )
    
    # Check total price (simplified)
    return validation_result.bind(lambda b: validate_booking_total(b, prices))


def validate_booking_total(booking, prices: tuple) -> Either[str, Any]:
    """Validate booking total price."""
    calculated_total = sum(
        item.guests * 100 for item in booking.items  # Simplified pricing
    )
    
    if abs(calculated_total - booking.total) > 10:  # Allow small difference
        return Either.left(f"Price mismatch detected. Expected: {calculated_total}, Got: {booking.total}")
    
    return Either.right(booking)


# Functional composition examples
def compose(*functions):
    """Compose multiple functions into a pipeline."""
    def composed(arg):
        result = arg
        for func in functions:
            result = func(result)
        return result
    return composed


def safe_booking_pipeline(booking_data, rates, rooms, availability):
    """Complete booking validation pipeline using monads."""
    pipeline = compose(
        lambda data: validate_dates(data['checkin'], data['checkout']),
        lambda dates_result: dates_result.bind(
            lambda dates: validate_guests(data['guests'], 4)  # Assume max 4 guests
        ),
        lambda guests_result: guests_result.bind(
            lambda guests: safe_rate(rates, data['rate_id'])
                .to_either("Rate not found")
        )
    )
    
    return pipeline(booking_data)