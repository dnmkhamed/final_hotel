from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from .domain import Hotel, Booking, Payment, Price
from .ftypes import Maybe, Either


def generate_revenue_report(
    bookings: Tuple[Booking, ...],
    payments: Tuple[Payment, ...],
    hotels: Tuple[Hotel, ...],
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """Generate revenue report."""
    
    def filter_by_date_range(items, date_field: str) -> List:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        return [
            item for item in items
            if start <= datetime.strptime(getattr(item, date_field), '%Y-%m-%d') <= end
        ]
    
    period_bookings = filter_by_date_range(bookings, 'created_at')
    period_payments = [p for p in payments if any(
        b.id == p.booking_id for b in period_bookings
    )]
    
    total_revenue = sum(p.amount for p in period_payments)
    
    # Revenue by hotel
    revenue_by_hotel = {}
    for hotel in hotels:
        hotel_bookings = [b for b in period_bookings if any(
            item.hotel_id == hotel.id for item in b.items
        )]
        hotel_payments = [p for p in period_payments if any(
            b.id == p.booking_id for b in hotel_bookings
        )]
        revenue_by_hotel[hotel.name] = sum(p.amount for p in hotel_payments)
    
    # Daily revenue
    daily_revenue = {}
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end_dt:
        date_str = current_date.strftime('%Y-%m-%d')
        date_payments = [p for p in period_payments if p.ts.startswith(date_str)]
        daily_revenue[date_str] = sum(p.amount for p in date_payments)
        current_date += timedelta(days=1)
    
    return {
        'period': f"{start_date} to {end_date}",
        'total_revenue': total_revenue,
        'booking_count': len(period_bookings),
        'revenue_by_hotel': revenue_by_hotel,
        'daily_revenue': daily_revenue,
        'average_booking_value': (
            total_revenue / len(period_bookings) if period_bookings else 0
        )
    }


def generate_occupancy_report(
    bookings: Tuple[Booking, ...],
    hotels: Tuple[Hotel, ...],
    room_types: Tuple[Any, ...],
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """Generate occupancy report."""
    
    def get_dates_in_range(start: str, end: str) -> List[str]:
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')
        return [
            (start_dt + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range((end_dt - start_dt).days + 1)
        ]
    
    dates = get_dates_in_range(start_date, end_date)
    
    # Calculate occupancy by hotel and date
    occupancy_data = {}
    for hotel in hotels:
        hotel_rooms = [rt for rt in room_types if rt.hotel_id == hotel.id]
        total_rooms = len(hotel_rooms)
        
        if total_rooms == 0:
            continue
        
        hotel_occupancy = {}
        for date in dates:
            # Count booked rooms for this hotel on this date
            booked_rooms = sum(
                1 for booking in bookings
                for item in booking.items
                if item.hotel_id == hotel.id and
                item.checkin <= date < item.checkout
            )
            occupancy_rate = (booked_rooms / total_rooms) * 100 if total_rooms > 0 else 0
            hotel_occupancy[date] = {
                'booked_rooms': booked_rooms,
                'total_rooms': total_rooms,
                'occupancy_rate': occupancy_rate
            }
        
        occupancy_data[hotel.name] = hotel_occupancy
    
    # Overall statistics
    total_rooms_all = sum(1 for rt in room_types)
    overall_occupancy = {}
    
    for date in dates:
        total_booked = sum(
            1 for booking in bookings
            for item in booking.items
            if item.checkin <= date < item.checkout
        )
        overall_rate = (total_booked / total_rooms_all) * 100 if total_rooms_all > 0 else 0
        overall_occupancy[date] = overall_rate
    
    return {
        'period': f"{start_date} to {end_date}",
        'occupancy_by_hotel': occupancy_data,
        'overall_occupancy': overall_occupancy,
        'total_rooms': total_rooms_all
    }


def generate_cancellation_report(
    bookings: Tuple[Booking, ...],
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """Generate cancellation report."""
    
    period_bookings = [b for b in bookings if start_date <= b.created_at <= end_date]
    
    total_bookings = len(period_bookings)
    cancelled_bookings = [b for b in period_bookings if b.status == 'cancelled']
    cancellation_rate = (len(cancelled_bookings) / total_bookings * 100) if total_bookings > 0 else 0
    
    # Cancellations by reason (simplified)
    cancellation_reasons = {
        'price': len([b for b in cancelled_bookings if any(
            'price' in str(b).lower() for b in cancelled_bookings
        )]),
        'schedule': len([b for b in cancelled_bookings if any(
            'schedule' in str(b).lower() for b in cancelled_bookings
        )]),
        'other': len([b for b in cancelled_bookings if not any(
            reason in str(b).lower() for reason in ['price', 'schedule']
        )])
    }
    
    return {
        'period': f"{start_date} to {end_date}",
        'total_bookings': total_bookings,
        'cancelled_bookings': len(cancelled_bookings),
        'cancellation_rate': cancellation_rate,
        'cancellation_reasons': cancellation_reasons,
        'revenue_lost': sum(b.total for b in cancelled_bookings)
    }