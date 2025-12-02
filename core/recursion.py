from typing import Tuple, List, Dict, Any, Callable
from datetime import datetime, timedelta
from .domain import RatePlan, Rule, SearchFilters, Hotel, RoomType, Price, Availability


def split_date_range(checkin: str, checkout: str) -> Tuple[str, ...]:
    """Split date range into individual dates using recursion."""
    
    def _split_dates(current: str, end: str, acc: List[str]) -> Tuple[str, ...]:
        if current >= end:
            return tuple(acc)
        next_date = (datetime.strptime(current, '%Y-%m-%d') + 
                    timedelta(days=1)).strftime('%Y-%m-%d')
        return _split_dates(next_date, end, acc + [current])
    
    return _split_dates(checkin, checkout, [])


def apply_rate_inheritance(rate: RatePlan, rules: Tuple[Rule, ...]) -> RatePlan:
    """Apply inheritance rules recursively."""
    
    def _apply_rule(current_rate: RatePlan, remaining_rules: Tuple[Rule, ...]) -> RatePlan:
        if not remaining_rules:
            return current_rate
        
        rule = remaining_rules[0]
        if rule.kind == 'meal_inheritance' and not current_rate.meal:
            new_rate = RatePlan(
                **{**current_rate.__dict__, 'meal': rule.payload.get('default_meal', 'RO')}
            )
            return _apply_rule(new_rate, remaining_rules[1:])
        
        return _apply_rule(current_rate, remaining_rules[1:])
    
    return _apply_rule(rate, rules)


def build_policy_tree(rate: RatePlan) -> Tuple[Dict[str, Any], ...]:
    """Build cancellation policy tree using recursion."""
    
    def _build_node(rate_plan: RatePlan, level: int) -> Dict[str, Any]:
        node = {
            'level': level,
            'refundable': rate_plan.refundable,
            'cancel_before_days': rate_plan.cancel_before_days,
            'meal': rate_plan.meal
        }
        
        if rate_plan.cancel_before_days and rate_plan.cancel_before_days > 0:
            child = _build_node(
                RatePlan(
                    **{**rate_plan.__dict__, 'cancel_before_days': rate_plan.cancel_before_days - 1}
                ),
                level + 1
            )
            node['children'] = (child,)
        else:
            node['children'] = ()
        
        return node
    
    return (_build_node(rate, 0),)


# Filter functions (Higher Order Functions)
def by_city(city: str) -> Callable[[Hotel], bool]:
    """Create city filter function."""
    return lambda hotel: hotel.city.lower() == city.lower()


def by_capacity(guests: int) -> Callable[[RoomType], bool]:
    """Create capacity filter function."""
    return lambda room: room.capacity >= guests


def by_features(required: Tuple[str, ...]) -> Callable[[Any], bool]:
    """Create features filter function."""
    def _filter(obj: Any) -> bool:
        if hasattr(obj, 'features'):
            return all(feature in obj.features for feature in required)
        return False
    return _filter


def by_price_range(min_amt: int, max_amt: int, currency: str) -> Callable[[Price], bool]:
    """Create price range filter function."""
    return lambda price: (min_amt <= price.amount <= max_amt and 
                         price.currency == currency)


def compose_filters(*filters: Callable) -> Callable:
    """Compose multiple filters into one."""
    def composed(item: Any) -> bool:
        return all(f(item) for f in filters)
    return composed


def filter_hotels(
    hotels: Tuple[Hotel, ...],
    filters: SearchFilters
) -> Tuple[Hotel, ...]:
    """Apply all filters to hotels."""
    filter_funcs = []
    
    if filters.city:
        filter_funcs.append(by_city(filters.city))
    
    if filters.features:
        filter_funcs.append(by_features(filters.features))
    
    if filters.stars:
        filter_funcs.append(lambda hotel: hotel.stars in filters.stars)
    
    if filter_funcs:
        composed_filter = compose_filters(*filter_funcs)
        return tuple(hotel for hotel in hotels if composed_filter(hotel))
    
    return hotels