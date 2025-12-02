import time
from functools import lru_cache
from typing import Tuple, Dict, Any
from datetime import datetime, timedelta
from .domain import Hotel, RoomType, RatePlan, Price, CartItem


@lru_cache(maxsize=128)
def quote_offer(
    hotel_id: str,
    room_type_id: str,
    rate_id: str,
    checkin: str,
    checkout: str,
    guests: int
) -> Dict[str, Any]:
    """
    Calculate offer quote with memoization.
    Simulate complex calculation with sleep.
    """
    # Simulate complex calculation
    time.sleep(0.001)  # 1ms delay
    
    # Calculate base price (in real app, this would use actual price data)
    checkin_date = datetime.strptime(checkin, '%Y-%m-%d')
    checkout_date = datetime.strptime(checkout, '%Y-%m-%d')
    nights = (checkout_date - checkin_date).days
    
    base_price = nights * 100  # Simplified pricing
    
    return {
        'hotel_id': hotel_id,
        'room_type_id': room_type_id,
        'rate_id': rate_id,
        'checkin': checkin,
        'checkout': checkout,
        'guests': guests,
        'nights': nights,
        'total_price': base_price,
        'currency': 'USD',
        'calculated_at': datetime.now().isoformat()
    }


def benchmark_quotes(
    iterations: int = 300
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Benchmark quote_offer with and without memoization."""
    
    def quote_without_cache(
        hotel_id: str,
        room_type_id: str,
        rate_id: str,
        checkin: str,
        checkout: str,
        guests: int
    ) -> Dict[str, Any]:
        """Quote function without cache for comparison."""
        time.sleep(0.001)
        checkin_date = datetime.strptime(checkin, '%Y-%m-%d')
        checkout_date = datetime.strptime(checkout, '%Y-%m-%d')
        nights = (checkout_date - checkin_date).days
        base_price = nights * 100
        
        return {
            'hotel_id': hotel_id,
            'room_type_id': room_type_id,
            'rate_id': rate_id,
            'total_price': base_price,
            'calculated_at': datetime.now().isoformat()
        }
    
    # Test parameters
    params = {
        'hotel_id': 'hotel_1',
        'room_type_id': 'room_1',
        'rate_id': 'rate_1',
        'checkin': '2024-01-01',
        'checkout': '2024-01-03',
        'guests': 2
    }
    
    # Benchmark without cache
    times_without_cache = []
    for i in range(iterations):
        start_time = time.time()
        quote_without_cache(**params)
        end_time = time.time()
        times_without_cache.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Clear cache for fair comparison
    quote_offer.cache_clear()
    
    # Benchmark with cache
    times_with_cache = []
    for i in range(iterations):
        start_time = time.time()
        quote_offer(**params)
        end_time = time.time()
        times_with_cache.append((end_time - start_time) * 1000)  # Convert to ms
    
    return tuple(times_without_cache), tuple(times_with_cache)


def get_memoization_stats() -> Dict[str, Any]:
    """Get memoization cache statistics."""
    cache_info = quote_offer.cache_info()
    return {
        'hits': cache_info.hits,
        'misses': cache_info.misses,
        'maxsize': cache_info.maxsize,
        'currsize': cache_info.currsize,
        'hit_ratio': (
            cache_info.hits / (cache_info.hits + cache_info.misses) 
            if (cache_info.hits + cache_info.misses) > 0 else 0
        )
    }