from fastapi import FastAPI, Request, Depends, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import json
import os
from typing import Dict, Any, List, Tuple, Optional
import asyncio
from uuid import uuid4
from datetime import datetime, timedelta

from core.domain import (
    Hotel, RoomType, RatePlan, Price, Availability, Guest, 
    CartItem, Booking, Payment, Event, Rule, SearchFilters, User
)
from core.transforms import load_seed, parse_seed_data, get_hotel_aggregates, hold_item, remove_hold
from core.recursion import filter_hotels, by_capacity, by_features, by_price_range, compose_filters
from core.memo import quote_offer, benchmark_quotes, get_memoization_stats
from core.ftypes import Maybe, Either, safe_rate, validate_booking, validate_cart_item
from core.frp import event_bus, publish_search_event, publish_hold_event, publish_booked_event
from core.report import generate_revenue_report, generate_occupancy_report, generate_cancellation_report

from core.lazy import lazy_search_results, lazy_calendar_generator
from core.service import (
    create_search_service, create_quote_service, create_booking_service,
    AsyncSearchService, AsyncQuoteService, IntegrationService
)
from core.compose import create_booking_pipeline
from pydantic import BaseModel

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from core.auth import register_user, authenticate_user, create_access_token, get_user_from_token, ACCESS_TOKEN_EXPIRE_MINUTES


app = FastAPI(title="Hotel Booking FP", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


class CartItemPayload(BaseModel):
    hotel_id: str
    room_type_id: str
    rate_id: str
    checkin: str
    checkout: str
    guests: int




# Global state (in production, use proper database)
class State:
    def __init__(self):
        self.seed_data = None
        self.hotels: Tuple[Hotel, ...] = ()
        self.room_types: Tuple[RoomType, ...] = ()
        self.rate_plans: Tuple[RatePlan, ...] = ()
        self.prices: Tuple[Price, ...] = ()
        self.availability: Tuple[Availability, ...] = ()
        self.guests: Tuple[Guest, ...] = ()
        self.rules: Tuple[Rule, ...] = ()
        self.bookings: Tuple[Booking, ...] = ()
        self.payments: Tuple[Payment, ...] = ()
        self.cart: Tuple[CartItem, ...] = ()
        self.users: Tuple[User, ...] = ()
        
        # Load initial data
        self.load_initial_data()
    
    def load_initial_data(self):
        """Load initial seed data."""
        try:
            # Try to load from seed.json
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            seed_path = os.path.join(current_dir, "data", "seed.json")
            
            if os.path.exists(seed_path):
                self.seed_data = load_seed(seed_path)
                (
                    self.hotels,
                    self.room_types,
                    self.rate_plans,
                    self.prices,
                    self.availability,
                    self.guests,
                    self.rules
                ) = parse_seed_data(self.seed_data)
            else:
                # Create fallback data if seed file doesn't exist
                self._create_fallback_data()
        except Exception as e:
            print(f"Error loading seed data: {e}")
            # Create minimal fallback data
            self._create_fallback_data()
    
    def _create_fallback_data(self):
         """Create fallback data if seed loading fails."""
         # Create 10 hotels with 5 rooms each (simplified version)
         self.hotels = (
             Hotel("hotel_1", "Grand Plaza Hotel", 5, "New York", ("wifi", "pool", "spa", "gym"), "Luxury hotel in Manhattan"),
             Hotel("hotel_2", "Seaside Resort", 4, "Miami", ("beach", "pool", "restaurant", "spa"), "Beachfront resort"),
             Hotel("hotel_3", "Mountain Lodge", 3, "Denver", ("wifi", "restaurant", "parking"), "Cozy mountain lodge"),
             Hotel("hotel_4", "City Center Hotel", 4, "Chicago", ("wifi", "gym", "restaurant", "business_center"), "Modern hotel in downtown"),
             Hotel("hotel_5", "Garden Inn", 3, "San Francisco", ("wifi", "breakfast", "garden", "parking"), "Charming inn with gardens"),
             Hotel("hotel_6", "Royal Palace Hotel", 5, "Las Vegas", ("casino", "pool", "spa", "restaurants"), "Luxury hotel and casino"),
             Hotel("hotel_7", "Harbor View Hotel", 4, "Seattle", ("harbor_view", "restaurant", "bar", "concierge"), "Hotel with waterfront views"),
             Hotel("hotel_8", "Historic Inn", 3, "Boston", ("historic", "breakfast", "free_parking", "wifi"), "Restored historic inn"),
             Hotel("hotel_9", "Business Tower", 4, "Atlanta", ("business_center", "gym", "restaurant", "conference_rooms"), "Modern business hotel"),
             Hotel("hotel_10", "Sunset Resort", 5, "Los Angeles", ("pool", "spa", "restaurant", "beach_access"), "Luxury beach resort"),
         )
         
         # Create 50 room types (5 per hotel)
         self.room_types = (
             # Hotel 1 rooms
             RoomType("room_1", "hotel_1", "Deluxe King", 2, ("king",), ("wifi", "tv", "ac", "minibar"), "45 m²"),
             RoomType("room_2", "hotel_1", "Executive Suite", 3, ("king", "sofa"), ("wifi", "tv", "ac", "minibar", "jacuzzi"), "65 m²"),
             RoomType("room_3", "hotel_1", "Presidential Suite", 4, ("king", "queen"), ("wifi", "multiple_tvs", "ac", "minibar", "jacuzzi"), "120 m²"),
             RoomType("room_4", "hotel_1", "Standard Double", 2, ("double",), ("wifi", "tv", "ac", "work_desk"), "35 m²"),
             RoomType("room_5", "hotel_1", "Family Room", 4, ("queen", "bunk"), ("wifi", "tv", "ac", "minifridge"), "50 m²"),
             # Hotel 2 rooms
             RoomType("room_6", "hotel_2", "Ocean View Room", 2, ("queen",), ("wifi", "tv", "ac", "balcony", "ocean_view"), "40 m²"),
             RoomType("room_7", "hotel_2", "Beachfront Suite", 3, ("king", "sofa"), ("wifi", "tv", "ac", "private_terrace"), "55 m²"),
             RoomType("room_8", "hotel_2", "Garden Room", 2, ("double",), ("wifi", "tv", "ac", "garden_view"), "38 m²"),
             RoomType("room_9", "hotel_2", "Pool View Room", 2, ("queen",), ("wifi", "tv", "ac", "pool_view"), "42 m²"),
             RoomType("room_10", "hotel_2", "Penthouse Suite", 4, ("king", "queen"), ("wifi", "multiple_tvs", "ac", "private_pool"), "85 m²"),
             # Add more rooms as needed...
         )
         
         # Create rate plans
         self.rate_plans = (
             RatePlan("rate_1", "hotel_1", "room_1", "Standard Rate", "RO", True, 2, "Flexible rate"),
             RatePlan("rate_2", "hotel_1", "room_1", "Non-refundable", "RO", False, None, "Best price"),
             RatePlan("rate_3", "hotel_2", "room_6", "Beach Package", "BB", True, 3, "Includes breakfast"),
         )
         
         # Empty collections
         self.prices = ()
         self.availability = ()
         self.guests = ()
         self.rules = ()
         self.bookings = ()
         self.payments = ()
         self.cart = ()

state = State()

# Dependency to get state
def get_state() -> State:
    return state

# --- Helper for Template Context ---

# Обновляем get_template_context чтобы передавать пользователя в шаблоны
async def get_template_context(request: Request, **extra) -> Dict[str, Any]:
    user = await get_current_user_optional(request)
    return {
        "request": request,
        "cart_size": len(state.cart),
        "user": user,  # Теперь user доступен во всех шаблонах
        **extra
    }


async def get_current_user_optional(request: Request) -> Optional[User]:
    token = request.cookies.get("access_token")
    if not token:
        return None
    # Очищаем префикс "Bearer " если он есть (обычно в куках мы храним чистый токен, но для надежности)
    if token.startswith("Bearer "):
        token = token.split(" ")[1]
    
    return get_user_from_token(token, state.users)





# --- HTML Page Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to overview."""
    return templates.TemplateResponse("overview.html", await get_template_context(
        request,
        aggregates=get_hotel_aggregates(state.hotels, state.room_types, state.prices),
        hotels=state.hotels[:5]
    ))


@app.get("/overview", response_class=HTMLResponse)
async def overview(request: Request):
    """Overview page with aggregates."""
    aggregates = get_hotel_aggregates(state.hotels, state.room_types, state.prices)
    return templates.TemplateResponse(
        "overview.html", 
        await get_template_context(
            request,
            aggregates=aggregates,
            hotels=state.hotels[:5]
        )
    )


@app.get("/data", response_class=HTMLResponse)
async def data_page(request: Request):
    """Data exploration page."""
    return templates.TemplateResponse(
        "data.html",
        await get_template_context(
            request,
            hotels=state.hotels,
            room_types=state.room_types,
            rate_plans=state.rate_plans
        )
    )


@app.get("/functional-core", response_class=HTMLResponse)
async def functional_core_page(request: Request):
    """Functional core demonstration page."""
    return templates.TemplateResponse(
        "functional_core.html",
        await get_template_context(
            request,
            hotels=state.hotels,
            room_types=state.room_types
        )
    )


@app.get("/pipelines", response_class=HTMLResponse)
async def pipelines_page(request: Request):
    """Filter pipelines page."""
    cities = list(set(hotel.city for hotel in state.hotels))
    return templates.TemplateResponse(
        "pipelines.html",
        await get_template_context(
            request,
            hotels=state.hotels,
            room_types=state.room_types,
            cities=cities
        )
    )


@app.get("/async", response_class=HTMLResponse)
async def async_page(request: Request):
    """Async/FRP events page."""
    return templates.TemplateResponse("async.html", await get_template_context(request))


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Reports page."""
    revenue_report = generate_revenue_report(
        state.bookings,
        state.payments,
        state.hotels,
        "2024-01-01",
        "2024-01-31"
    )
    
    occupancy_report = generate_occupancy_report(
        state.bookings,
        state.hotels,
        state.room_types,
        "2024-01-01",
        "2024-01-31"
    )
    
    return templates.TemplateResponse(
        "reports.html",
        await get_template_context(
            request,
            revenue_report=revenue_report,
            occupancy_report=occupancy_report
        )
    )


@app.get("/tests", response_class=HTMLResponse)
async def tests_page(request: Request):
    """Tests information page."""
    return templates.TemplateResponse("tests.html", await get_template_context(request))


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About page."""
    return templates.TemplateResponse("about.html", await get_template_context(request))


@app.get("/hotels", response_class=HTMLResponse)
async def hotels_page(request: Request):
    """Страница списка отелей"""
    cities = list(set(hotel.city for hotel in state.hotels))
    return templates.TemplateResponse(
        "hotels.html",
        await get_template_context(
            request,
            hotels=state.hotels,
            cities=cities
        )
    )


@app.get("/booking", response_class=HTMLResponse)
async def booking_page(request: Request):
    """Страница моих бронирований"""
    return templates.TemplateResponse(
        "booking.html",
        await get_template_context(
            request,
            bookings=state.bookings
        )
    )


@app.get("/labs", response_class=HTMLResponse)
async def labs_page(request: Request):
    """Страница лабораторных функций"""
    return templates.TemplateResponse("labs.html", await get_template_context(request))


# app/main.py

@app.get("/cart", response_class=HTMLResponse)
async def cart_page(request: Request):
    """Страница корзины"""
    
    # --- НОВАЯ ЛОГИКА ОБОГАЩЕНИЯ ---
    
    # Создаем словари для быстрого поиска
    hotels_by_id = {h.id: h for h in state.hotels}
    rooms_by_id = {r.id: r for r in state.room_types}

    rich_cart = []
    total_price = 0

    for item in state.cart:
        hotel = hotels_by_id.get(item.hotel_id)
        room = rooms_by_id.get(item.room_type_id)

        # Используем quote_offer для получения цены (она мемоизирована)
        quote = quote_offer(
            hotel_id=item.hotel_id,
            room_type_id=item.room_type_id,
            rate_id=item.rate_id,
            checkin=item.checkin,
            checkout=item.checkout,
            guests=item.guests
        )
        price = quote.get('total_price', 0) # Получаем цену из мемо-функции
        total_price += price

        rich_cart.append({
            'item_id': item.id,
            'hotel': hotel,  # Теперь это полный объект Hotel
            'room': room,    # Теперь это полный объект RoomType
            'checkin': item.checkin,
            'checkout': item.checkout,
            'guests': item.guests,
            'price': price   # Рассчитанная цена
        })
    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    return templates.TemplateResponse(
        "cart.html", 
        await get_template_context(
            request,
            cart=rich_cart,          # <-- Передаем новый "богатый" список
            cart_total=total_price   # <-- Передаем итоговую сумму
        )
    )
# --- API Endpoints ---

@app.get("/api/hotels")
async def get_hotels(
    city: Optional[str] = None,
    guests: Optional[int] = None,
    features: Optional[str] = None
):
    """Get hotels with filtering."""
    filters = SearchFilters(
        city=city,
        guests=guests,
        features=tuple(features.split(',')) if features else ()
    )
    
    filtered_hotels = filter_hotels(state.hotels, filters)
    
    # Publish search event
    await publish_search_event({
        "city": city,
        "guests": guests,
        "features": features
    })
    
    return {
        "hotels": [
            {
                "id": hotel.id,
                "name": hotel.name,
                "stars": hotel.stars,
                "city": hotel.city,
                "features": hotel.features,
                "description": hotel.description
                # "photo_url": hotel.photo_url # photo_url не был в fallback data
            }
            for hotel in filtered_hotels
        ]
    }


# app/main.py (строка 347)
@app.post("/api/cart")
async def add_to_cart(item: CartItemPayload):  # <--- ИЗМЕНЕНИЕ ЗДЕСЬ
    """Add item to cart."""
    cart_item = CartItem(
        id=f"cart_{uuid4().hex[:8]}",
        hotel_id=item.hotel_id,         # <--- .hotel_id
        room_type_id=item.room_type_id, # <--- .room_type_id
        rate_id=item.rate_id,           # <--- .rate_id
        checkin=item.checkin,           # <--- .checkin
        checkout=item.checkout,         # <--- .checkout
        guests=item.guests              # <--- .guests
    )
    
    state.cart = hold_item(state.cart, cart_item)
    
    # ... (остальной код)
    
    # Publish hold event
    await publish_hold_event({
        "id": cart_item.id,
        "hotel_id": cart_item.hotel_id,
        "room_type_id": cart_item.room_type_id,
        "checkin": cart_item.checkin,
        "checkout": cart_item.checkout,
        "guests": cart_item.guests
    })
    
    return {"success": True, "cart_size": len(state.cart)}


@app.get("/api/quote")
async def get_quote(
    hotel_id: str,
    room_type_id: str,
    rate_id: str,
    checkin: str,
    checkout: str,
    guests: int
):
    """Get quote for booking."""
    quote = quote_offer(hotel_id, room_type_id, rate_id, checkin, checkout, guests)
    return quote


@app.get("/api/memo-stats")
async def get_memo_stats():
    """Get memoization statistics."""
    return get_memoization_stats()


@app.get("/api/events")
async def get_events(limit: int = 20):
    """Get recent events."""
    events = event_bus.get_event_history(limit)
    return {
        "events": [
            {
                "id": event.id,
                "timestamp": event.ts,
                "name": event.name,
                "payload": event.payload
            }
            for event in events
        ]
    }


@app.post("/api/validate-booking")
async def validate_booking_api(booking_data: Dict[str, Any]):
    """Validate booking using functional types."""
    booking = Booking(
        id=booking_data.get("id", f"booking_{uuid4().hex[:8]}"),
        guest_id=booking_data.get("guest_id", "guest_1"),
        items=tuple(CartItem(**item) for item in booking_data.get("items", [])),
        total=booking_data.get("total", 0),
        status="pending",
        created_at=datetime.now().strftime("%Y-%m-%d")
    )
    
    # Validate using Either monad
    validation_result = validate_booking(
        booking,
        state.prices,
        state.availability,
        state.rules
    )
    
    if validation_result.is_right():
        # Publish booked event
        await publish_booked_event({
            "id": booking.id,
            "guest_id": booking.guest_id,
            "total": booking.total,
            "status": booking.status
        })
        return {"valid": True, "booking": {
            "id": booking.id,
            "guest_id": booking.guest_id,
            "total": booking.total,
            "status": booking.status
        }}
    else:
        return {"valid": False, "error": str(validation_result)}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "hotels_count": len(state.hotels),
        "room_types_count": len(state.room_types),
        "rate_plans_count": len(state.rate_plans)
    }


@app.get("/api/rooms/{hotel_id}")
async def get_rooms_by_hotel(hotel_id: str):
    """Get rooms for a specific hotel."""
    rooms = [rt for rt in state.room_types if rt.hotel_id == hotel_id]
    return {
        "rooms": [
            {
                "id": room.id,
                "name": room.name,
                "capacity": room.capacity,
                "features": room.features,
                "size": room.size
            }
            for room in rooms
        ]
    }


@app.get("/api/availability")
async def check_availability(room_type_id: str, checkin: str, checkout: str):
    """Check room availability for dates."""
    available = any(
        avail.room_type_id == room_type_id and 
        avail.date >= checkin and 
        avail.date < checkout
        for avail in state.availability
    ) if state.availability else True  # If no availability data, assume available
    
    return {
        "available": available,
        "room_type_id": room_type_id,
        "checkin": checkin,
        "checkout": checkout
    }


@app.post("/api/booking")
async def create_booking(booking_data: Dict[str, Any]):
    """Create a new booking."""
    booking_id = f"booking_{uuid4().hex[:8]}"
    booking = Booking(
        id=booking_id,
        guest_id=booking_data.get("guest_id", "guest_1"),
        items=tuple(CartItem(**item) for item in booking_data.get("items", [])),
        total=booking_data.get("total", 0),
        status="confirmed",
        created_at=datetime.now().strftime("%Y-%m-%d")
    )
    
    # Add to state (in production, save to database)
    state.bookings = state.bookings + (booking,)
    
    # Create payment record
    payment = Payment(
        id=f"payment_{uuid4().hex[:8]}",
        booking_id=booking_id,
        amount=booking.total,
        ts=datetime.now().isoformat(),
        method=booking_data.get("payment_method", "credit_card")
    )
    state.payments = state.payments + (payment,)
    
    # Publish booked event
    await publish_booked_event({
        "booking_id": booking_id,
        "total": booking.total,
        "guest_id": booking.guest_id
    })
    
    return {
        "success": True,
        "booking_id": booking_id,
        "total": booking.total,
        "status": "confirmed"
    }


@app.get("/api/lazy-search")
async def lazy_search(
    city: Optional[str] = None,
    guests: Optional[int] = None,
    max_price: Optional[int] = None,
    limit: int = 10
):
    """Ленивый поиск отелей"""
    filters = {
        'city': city,
        'guests': guests,
        'max_price': max_price
    }
    
    results = list(lazy_search_results(
        state.hotels,
        state.room_types,
        state.rate_plans,
        state.prices,
        state.availability,
        filters,
        limit=limit
    ))
    
    return {
        "results": [
            {
                "hotel": {
                    "id": r['hotel'].id,
                    "name": r['hotel'].name,
                    "city": r['hotel'].city,
                    "stars": r['hotel'].stars
                },
                "room_type": {
                    "id": r['room_type'].id,
                    "name": r['room_type'].name,
                    "capacity": r['room_type'].capacity
                },
                "price_per_night": r['price_per_night'],
                "total_price": r['total_price'],
                "available": r['available']
            }
            for r in results
        ],
        "count": len(results)
    }


@app.get("/api/frp-state")
async def get_frp_state():
    """Получить состояние FRP системы"""
    state_data = event_bus.get_state()
    return {
        "active_holds": state_data.get('active_holds', []),
        "recent_bookings": state_data.get('recent_bookings', []),
        "price_changes": state_data.get('price_changes', []),
        "search_queries": len(state_data.get('search_queries', [])),
        "cancellations": state_data.get('cancellations', [])
    }


@app.post("/api/compose-booking")
async def compose_booking(booking_data: dict):
    """Бронирование через композицию функций"""
    pipeline = create_booking_pipeline()
    result = pipeline(booking_data)
    
    if result.is_right():
        booking = result.get_value()
        return {"success": True, "booking": booking}
    else:
        return {"success": False, "error": result.get_error()}


@app.post("/api/async-quote")
async def async_quote_batch(quote_requests: list):
    """Асинхронный пакетный расчет котировок"""
    quote_service = create_quote_service()
    async_service = AsyncQuoteService(quote_service)
    
    result = await async_service.quote_batch(quote_requests)
    return result


@app.post("/api/end-to-end")
async def end_to_end_workflow(workflow_data: dict):
    """Сквозной workflow от поиска до бронирования"""
    integration_service = IntegrationService(
        create_search_service(),
        create_quote_service(),
        create_booking_service()
    )
    
    result = await integration_service.end_to_end_workflow(
        workflow_data.get('search', {}),
        workflow_data.get('guest', {}),
        state.hotels,
        state.room_types,
        state.rate_plans,
        state.prices,
        state.availability,
        state.rules
    )
    
    return result


@app.get("/api/parallel-search")
async def parallel_search(search_requests: list):
    """Параллельный поиск по нескольким запросам"""
    search_service = create_search_service()
    async_service = AsyncSearchService(search_service)
    
    results = await async_service.search_parallel(
        search_requests,
        state.hotels,
        state.room_types,
        state.rate_plans,
        state.prices,
        state.availability
    )
    
    return {"results": results}


@app.post("/api/checkout")
async def checkout_cart(request: Request):
    """Превращает корзину в бронирование и очищает корзину."""
    user = await get_current_user_optional(request)
    
    if not state.cart:
        return {"success": False, "error": "Корзина пуста"}
        
    # Считаем общую сумму (в реальном проекте используем QuoteService)
    # Здесь упрощенно: сумма цен из элементов корзины, которые мы считали на фронте
    # Но для надежности пересчитаем базовую стоимость
    total_amount = 0
    for item in state.cart:
        # Простая логика цены для лабы: 100$ * гости * дни (или берем из прайсов)
        # Для простоты демонстрации ставим заглушку суммы, или считаем честно:
        dt1 = datetime.strptime(item.checkin, "%Y-%m-%d")
        dt2 = datetime.strptime(item.checkout, "%Y-%m-%d")
        nights = (dt2 - dt1).days
        total_amount += (150 * nights * item.guests) # Примерная цена

    # Создаем объект бронирования
    new_booking = Booking(
        id=f"bk_{uuid4().hex[:8]}",
        guest_id=user.id if user else "guest_anon",
        items=state.cart, # Переносим товары из корзины в бронь
        total=total_amount,
        status="PAID", # Сразу ставим статус ОПЛАЧЕНО
        created_at=datetime.now().strftime("%Y-%m-%d")
    )
    
    # Добавляем в историю бронирований (иммутабельно обновляем кортеж)
    state.bookings = state.bookings + (new_booking,)
    
    # Очищаем корзину
    state.cart = ()
    
    # Генерируем событие (FRP)
    await publish_booked_event({
        "booking_id": new_booking.id,
        "total": total_amount,
        "items_count": len(new_booking.items)
    })
    
    return {"success": True, "booking_id": new_booking.id}







# --- Main execution ---

if __name__ == "__main__":
    import uvicorn
    # Обратите внимание: для uvicorn.run путь к app должен быть 'main:app'
    # Этот файл следует запускать из командной строки:
    # uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    print("Starting Uvicorn server... (Run with: uvicorn app.main:app --reload)")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)



@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", await get_template_context(request))

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", await get_template_context(request))

@app.post("/auth/register")
async def register_endpoint(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(...)
):
    # Используем чистую функцию register_user
    result = register_user(
        {"email": email, "password": password, "name": name}, 
        state.users
    )
    
    if result.is_right():
        new_user = result.get_value()
        # Обновляем стейт (иммутабельно заменяем кортеж)
        state.users = state.users + (new_user,)
        
        # Создаем токен и логиним сразу
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": new_user.email}, expires_delta=access_token_expires
        )
        
        response = RedirectResponse(url="/overview", status_code=303)
        response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
        return response
    else:
        # Возвращаем ошибку на страницу
        return templates.TemplateResponse(
            "register.html", 
            await get_template_context(request, error=result.get_error())
        )

@app.post("/auth/login")
async def login_endpoint(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    result = authenticate_user(
        {"email": email, "password": password},
        state.users
    )
    
    if result.is_right():
        user = result.get_value()
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        
        response = RedirectResponse(url="/overview", status_code=303)
        response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
        return response
    else:
        return templates.TemplateResponse(
            "login.html", 
            await get_template_context(request, error=result.get_error())
        )

@app.get("/auth/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    return response