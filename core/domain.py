from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


@dataclass(frozen=True)
class Hotel:
    id: str
    name: str
    stars: int
    city: str
    features: Tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class RoomType:
    id: str
    hotel_id: str
    name: str
    capacity: int
    beds: Tuple[str, ...]
    features: Tuple[str, ...]
    size: str = ""


# Остальные классы без изменений...


@dataclass(frozen=True)
class RatePlan:
    id: str
    hotel_id: str
    room_type_id: str
    title: str
    meal: str
    refundable: bool
    cancel_before_days: Optional[int]
    description: str = ""


@dataclass(frozen=True)
class Price:
    id: str
    rate_id: str
    date: str
    amount: int
    currency: str


@dataclass(frozen=True)
class Availability:
    id: str
    room_type_id: str
    date: str
    available: int


@dataclass(frozen=True)
class Guest:
    id: str
    name: str
    email: str


@dataclass(frozen=True)
class CartItem:
    id: str
    hotel_id: str
    room_type_id: str
    rate_id: str
    checkin: str
    checkout: str
    guests: int


@dataclass(frozen=True)
class Booking:
    id: str
    guest_id: str
    items: Tuple[CartItem, ...]
    total: int
    status: str
    created_at: str = ""


@dataclass(frozen=True)
class Payment:
    id: str
    booking_id: str
    amount: int
    ts: str
    method: str


@dataclass(frozen=True)
class Event:
    id: str
    ts: str
    name: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class SearchFilters:
    city: Optional[str] = None
    checkin: Optional[str] = None
    checkout: Optional[str] = None
    guests: Optional[int] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    features: Tuple[str, ...] = ()
    stars: Tuple[int, ...] = ()


    # ... (существующие импорты)

@dataclass(frozen=True)
class User:
    id: str
    email: str
    password_hash: str  # Храним только хеш!
    name: str