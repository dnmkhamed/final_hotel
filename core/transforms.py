import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from .domain import (
    Hotel, RoomType, RatePlan, Price, Availability, Guest, 
    CartItem, Booking, Payment, Event, Rule, SearchFilters
)


def load_seed(path: str) -> Dict[str, Any]:
    """Load seed data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_seed_data(seed_data: Dict[str, Any]) -> Tuple[
    Tuple[Hotel, ...],
    Tuple[RoomType, ...],
    Tuple[RatePlan, ...],
    Tuple[Price, ...],
    Tuple[Availability, ...],
    Tuple[Guest, ...],
    Tuple[Rule, ...]
]:
    """Parse seed data into domain objects."""
    hotels = tuple(Hotel(**hotel) for hotel in seed_data.get('hotels', []))
    room_types = tuple(RoomType(**room) for room in seed_data.get('room_types', []))
    rate_plans = tuple(RatePlan(**rate) for rate in seed_data.get('rate_plans', []))
    prices = tuple(Price(**price) for price in seed_data.get('prices', []))
    availability = tuple(Availability(**avail) for avail in seed_data.get('availability', []))
    guests = tuple(Guest(**guest) for guest in seed_data.get('guests', []))
    rules = tuple(Rule(**rule) for rule in seed_data.get('rules', []))
    
    return hotels, room_types, rate_plans, prices, availability, guests, rules


def hold_item(cart: Tuple[CartItem, ...], item: CartItem) -> Tuple[CartItem, ...]:
    """Add item to cart (immutable)."""
    return cart + (item,)


def remove_hold(cart: Tuple[CartItem, ...], item_id: str) -> Tuple[CartItem, ...]:
    """Remove item from cart (immutable)."""
    return tuple(item for item in cart if item.id != item_id)


def nightly_sum(
    prices: Tuple[Price, ...], 
    checkin: str, 
    checkout: str, 
    rate_id: str
) -> int:
    """Calculate total price for stay duration."""
    def date_range(start: str, end: str) -> List[str]:
        try:
            start_date = datetime.strptime(start, '%Y-%m-%d').date()
            end_date = datetime.strptime(end, '%Y-%m-%d').date()
            return [
                (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range((end_date - start_date).days)
            ]
        except ValueError:
            return []
    
    stay_dates = date_range(checkin, checkout)
    rate_prices = [p for p in prices if p.rate_id == rate_id and p.date in stay_dates]
    
    return sum(price.amount for price in rate_prices) if rate_prices else 0


def get_hotel_aggregates(
    hotels: Tuple[Hotel, ...],
    room_types: Tuple[RoomType, ...],
    prices: Tuple[Price, ...]
) -> Dict[str, Any]:
    """Calculate aggregates for overview."""
    hotel_count = len(hotels)
    room_count = len(room_types)
    
    if hotel_count == 0:
        return {
            'hotel_count': 0,
            'room_count': 0,
            'cities': [],
            'city_stats': {},
            'avg_prices': {}
        }
    
    cities = list(set(hotel.city for hotel in hotels))
    city_stats = {}
    
    for city in cities:
        city_hotels = [h for h in hotels if h.city == city]
        if city_hotels:
            city_stats[city] = {
                'hotels': len(city_hotels),
                'avg_stars': sum(h.stars for h in city_hotels) / len(city_hotels)
            }
    
    # Calculate average prices by city
    avg_prices = {}
    for city in cities:
        city_hotels = [h.id for h in hotels if h.city == city]
        city_room_types = [r.id for r in room_types if r.hotel_id in city_hotels]
        city_prices = [p.amount for p in prices if any(
            r_id == p.rate_id for r_id in city_room_types
        )]
        avg_prices[city] = sum(city_prices) / len(city_prices) if city_prices else 100
    
    return {
        'hotel_count': hotel_count,
        'room_count': room_count,
        'cities': cities,
        'city_stats': city_stats,
        'avg_prices': avg_prices
    }