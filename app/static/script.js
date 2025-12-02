// Hotel Booking FP - Functional Programming SPA
// Main JavaScript file with all functionality

// --- Инициализация приложения ---

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupMobileMenu();
    setupEventListeners();
    loadInitialData();
    setupCurrentPage();
    updateCartDisplay(); // Обновляем корзину при загрузке
    initializeDateInputs(); // Инициализируем даты
}

function setupMobileMenu() {
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
        });
    }
    
    // Close menu when clicking outside
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.nav-container')) {
            if (navMenu) navMenu.classList.remove('active');
        }
    });
}

function setupCurrentPage() {
    // Add active class to current page link
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

function setupEventListeners() {
    // Setup filter listeners for pipelines page
    setupFilterListeners();
    
    // Setup functional core demo listeners
    setupFunctionalCoreListeners();
    
    // Setup async/FRP listeners
    setupAsyncListeners();

    // Setup enhanced filter listeners (if they exist)
    const applyEnhanced = document.getElementById('applyFiltersEnhanced');
    if (applyEnhanced) {
        applyEnhanced.addEventListener('click', applyFiltersEnhanced);
    }
}

function setupFilterListeners() {
    const cityFilter = document.getElementById('cityFilter');
    const guestsFilter = document.getElementById('guestsFilter');
    
    if (cityFilter) {
        cityFilter.addEventListener('change', applyFilters);
    }
    
    if (guestsFilter) {
        guestsFilter.addEventListener('input', debounce(applyFilters, 300));
    }
    
    // Feature checkboxes
    const featureCheckboxes = document.querySelectorAll('input[name="features"]');
    featureCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', applyFilters);
    });
}

function setupFunctionalCoreListeners() {
    // Validation demo buttons
    const successBtn = document.querySelector('[onclick*="testSuccessfulValidation"]');
    const failBtn = document.querySelector('[onclick*="testFailedValidation"]');
    
    if (successBtn) {
        successBtn.onclick = testSuccessfulValidation;
    }
    
    if (failBtn) {
        failBtn.onclick = testFailedValidation;
    }
}

function setupAsyncListeners() {
    // Event streaming setup
    if (document.getElementById('eventList')) {
        startEventPolling();
    }
}

// --- API функции ---

async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        showError(`API Error: ${error.message}`);
        throw error;
    }
}

async function searchHotels(filters) {
    const params = new URLSearchParams();
    if (filters.city) params.append('city', filters.city);
    if (filters.guests) params.append('guests', filters.guests);
    if (filters.features && filters.features.length > 0) {
        params.append('features', filters.features.join(','));
    }
    
    return await apiCall(`/api/hotels?${params}`);
}

async function addToCart(item) {
    // Эта функция вызывает API бэкенда
    return await apiCall('/api/cart', {
        method: 'POST',
        body: JSON.stringify(item)
    });
}

async function getQuote(params) {
    const queryParams = new URLSearchParams(params);
    return await apiCall(`/api/quote?${queryParams}`);
}

async function getMemoStats() {
    return await apiCall('/api/memo-stats');
}

async function getEvents(limit = 20) {
    return await apiCall(`/api/events?limit=${limit}`);
}

async function validateBooking(bookingData) {
    return await apiCall('/api/validate-booking', {
        method: 'POST',
        body: JSON.stringify(bookingData)
    });
}

// --- Управление корзиной (localStorage) ---

function getCart() {
    return JSON.parse(localStorage.getItem('hotel_cart') || '[]');
}

function saveCart(cart) {
    localStorage.setItem('hotel_cart', JSON.stringify(cart));
}

function updateCartDisplay() {
    const cart = getCart();
    const cartCount = document.querySelector('.cart-link'); // Селектор из блока 3
    if (cartCount) {
        const count = cart.length;
        if (count > 0) {
            cartCount.innerHTML = `🛒 Корзина (${count})`;
        } else {
            cartCount.innerHTML = '🛒 Корзина';
        }
    }
}

// app/static/script.js
function addToCartWithRoom(hotelId, roomId, roomName, price, checkin, checkout, guests) {
    const nights = calculateNights(checkin, checkout);
    const total = price * nights;
    
    // --- 1. Объект для localStorage (для UI) ---
    const cartItemForStorage = {
        id: 'cart_' + Date.now(),
        hotel_id: hotelId,
        room_type_id: roomId,
        room_name: roomName,
        price_per_night: price,
        checkin: checkin,
        checkout: checkout,
        nights: nights,
        guests: guests,
        total: total
    };
    
    // --- 2. Объект для API бэкенда (строго по Pydantic-модели) ---
    const apiItemPayload = {
        hotel_id: hotelId,
        room_type_id: roomId,
        rate_id: "rate_1", // ВНИМАНИЕ: Это заглушка, т.к. UI не дает выбрать rate_id
        checkin: checkin,
        checkout: checkout,
        guests: parseInt(guests, 10)
    };

    // --- 3. Сохраняем в localStorage ---
    const cart = getCart();
    cart.push(cartItemForStorage);
    saveCart(cart);
    updateCartDisplay();
    closeModal();
    
    // --- 4. ОТПРАВЛЯЕМ НА БЭКЕНД ---
    addToCart(apiItemPayload)
        .then(response => {
            if (response.success) {
                showAlert(`Добавлено: ${roomName}! Всего: ${formatPrice(total)} за ${nights} ночи`, 'success');
            } else {
                showError('Ошибка бэкенда при добавлении в корзину');
            }
        })
        .catch(err => {
            console.error('Failed to save to backend:', err);
            // showError уже сработает внутри apiCall
        });
}

function viewCart() {
    const cart = getCart();
    if (cart.length === 0) {
        showMessage('Your cart is empty', 'info');
        return;
    }
    
    // Show cart modal or navigate to cart page
    console.log('Cart items:', cart);
    showMessage(`Cart has ${cart.length} items`, 'success');
    // window.location.href = '/cart'; // (Раскомментируйте для перехода)
}


// --- Фильтры и поиск (Pipelines) ---

function applyFilters() {
    showLoading('filterResults');
    
    const filters = {
        city: document.getElementById('cityFilter')?.value || '',
        guests: document.getElementById('guestsFilter')?.value || '',
        features: Array.from(document.querySelectorAll('input[name="features"]:checked'))
            .map(cb => cb.value)
    };
    
    searchHotels(filters)
        .then(data => {
            displayFilteredHotels(data.hotels);
        })
        .catch(error => {
            showError('Failed to load hotels');
            console.error('Filter error:', error);
        });
}

function displayFilteredHotels(hotels) {
    const resultsDiv = document.getElementById('filterResults');
    if (!resultsDiv) return;
    
    if (hotels.length === 0) {
        resultsDiv.innerHTML = `
            <div class="text-center p-3">
                <p>No hotels found matching your criteria.</p>
                <button class="btn-secondary" onclick="clearFilters()">Clear Filters</button>
            </div>
        `;
        return;
    }
    
    resultsDiv.innerHTML = hotels.map(hotel => `
        <div class="hotel-card fade-in">
            <div class="hotel-header">
                <div class="hotel-emoji">
                    ${getHotelEmoji(hotel.stars)}
                </div>
                <div>
                    <h3>${escapeHtml(hotel.name)}</h3>
                    <p class="location">${escapeHtml(hotel.city)} • ${'⭐'.repeat(hotel.stars)}</p>
                </div>
            </div>
            <p class="description">${escapeHtml(hotel.description)}</p>
            <div class="features">
                ${hotel.features.map(feature => 
                    `<span class="feature-tag">${escapeHtml(feature)}</span>`
                ).join('')}
            </div>
            <button class="btn-primary" onclick="selectHotel('${hotel.id}')">
                Select Hotel
            </button>
        </div>
    `).join('');
}

function clearFilters() {
    const cityFilter = document.getElementById('cityFilter');
    const guestsFilter = document.getElementById('guestsFilter');
    const featureCheckboxes = document.querySelectorAll('input[name="features"]');
    
    if (cityFilter) cityFilter.value = '';
    if (guestsFilter) guestsFilter.value = '';
    featureCheckboxes.forEach(checkbox => checkbox.checked = false);
    
    applyFilters();
}

function selectHotel(hotelId) {
    showSuccess(`Selected hotel: ${hotelId}`);
    // Здесь можно показать модальное окно с выбором номера
    // Например, вызовем showRoomSelection с датами по умолчанию
    const checkin = document.getElementById('checkinFilter')?.value || formatDateISO(new Date());
    const checkout = document.getElementById('checkoutFilter')?.value || formatDateISO(getTomorrow());
    const guests = document.getElementById('guestsFilter')?.value || 2;
    showRoomSelection(hotelId, checkin, checkout, guests);
}

// --- Расширенные фильтры и выбор номера ---

async function applyFiltersEnhanced() {
    showLoading('filterResults');
    const city = document.getElementById('cityFilter')?.value || '';
    const checkin = document.getElementById('checkinFilter')?.value || '';
    const checkout = document.getElementById('checkoutFilter')?.value || '';
    const guests = document.getElementById('guestsFilter')?.value || '';
    const features = Array.from(document.querySelectorAll('input[name="features"]:checked'))
        .map(cb => cb.value);
    
    try {
        const data = await searchHotels({ city, guests, features: features.join(',') });
        displaySearchResults(data.hotels, { checkin, checkout, guests });
    } catch (error) {
        console.error('Filter error:', error);
        showError('Failed to search hotels');
    }
}

function displaySearchResults(hotels, searchParams) {
    const resultsDiv = document.getElementById('filterResults');
    if (!resultsDiv) return;
    
    if (hotels.length === 0) {
        resultsDiv.innerHTML = `
            <div class="text-center p-3">
                <p>No hotels found matching your criteria.</p>
                <button class="btn-secondary" onclick="clearFilters()">Clear Filters</button>
            </div>
        `;
        return;
    }
    
    resultsDiv.innerHTML = hotels.map(hotel => `
        <div class="hotel-card">
            <div class="hotel-header">
                <div class="hotel-emoji">
                    ${getHotelEmoji(hotel.stars)}
                </div>
                <div>
                    <h3>${escapeHtml(hotel.name)}</h3>
                    <p class="location">${escapeHtml(hotel.city)} • ${'⭐'.repeat(hotel.stars)}</p>
                </div>
            </div>
            <p class="description">${escapeHtml(hotel.description)}</p>
            <div class="features">
                ${hotel.features.map(feature => 
                    `<span class="feature-tag">${escapeHtml(feature)}</span>`
                ).join('')}
            </div>
            <div class="booking-actions">
                <div class="price">From ${formatPrice(200)}/night</div>
                <button class="btn-primary" onclick="showRoomSelection('${hotel.id}', '${searchParams.checkin}', '${searchParams.checkout}', ${searchParams.guests || 2})">
                    Book Now
                </button>
            </div>
        </div>
    `).join('');
}

function showRoomSelection(hotelId, checkin, checkout, guests) {
    // TODO: Загружать комнаты по API
    // (пока используем заглушку)
    
    const modalHTML = `
        <div class="room-selection-modal">
            <h3>Select Room (Hotel: ${hotelId})</h3>
            <div class="room-options">
                <div class="room-option">
                    <h4>Deluxe King Room</h4>
                    <p>45 m² • King bed • Sleeps 2</p>
                    <p class="price">${formatPrice(200)} per night</p>
                    <button class="btn-primary" onclick="addToCartWithRoom('${hotelId}', 'room_1', 'Deluxe King Room', 200, '${checkin}', '${checkout}', ${guests})">
                        Select Room
                    </button>
                </div>
                <div class="room-option">
                    <h4>Executive Suite</h4>
                    <p>65 m² • King bed + Sofa • Sleeps 3</p>
                    <p class="price">${formatPrice(300)} per night</p>
                    <button class="btn-primary" onclick="addToCartWithRoom('${hotelId}', 'room_2', 'Executive Suite', 300, '${checkin}', '${checkout}', ${guests})">
                        Select Room
                    </button>
                </div>
                <div class="room-option">
                    <h4>Standard Double</h4>
                    <p>35 m² • Double bed • Sleeps 2</p>
                    <p class="price">${formatPrice(150)} per night</p>
                    <button class="btn-primary" onclick="addToCartWithRoom('${hotelId}', 'room_3', 'Standard Double', 150, '${checkin}', '${checkout}', ${guests})">
                        Select Room
                    </button>
                </div>
            </div>
            <button class="btn-secondary" onclick="closeModal()">Cancel</button>
        </div>
    `;
    
    showModal(modalHTML);
}

// --- Функциональное ядро (Demos) ---

async function testSuccessfulValidation() {
    const bookingData = {
        id: "booking_" + Date.now(),
        guest_id: "guest_1", 
        items: [{
            id: "item_" + Date.now(),
            hotel_id: "hotel_1",
            room_type_id: "room_1",
            rate_id: "rate_1",
            checkin: "2024-01-15",
            checkout: "2024-01-17",
            guests: 2
        }],
        total: 400
    };

    showLoading('validationResult');
    
    try {
        const result = await validateBooking(bookingData);
        displayValidationResult(result);
    } catch (error) {
        showError('Validation request failed');
    }
}

async function testFailedValidation() {
    const bookingData = {
        id: "booking_" + Date.now(),
        guest_id: "guest_1",
        items: [{
            id: "item_" + Date.now(), 
            hotel_id: "hotel_1",
            room_type_id: "room_999", // Invalid room
            rate_id: "rate_1",
            checkin: "2024-01-15",
            checkout: "2024-01-17", 
            guests: 2
        }],
        total: 400
    };

    showLoading('validationResult');
    
    try {
        const result = await validateBooking(bookingData);
        displayValidationResult(result);
    } catch (error) {
        showError('Validation request failed');
    }
}

function displayValidationResult(result) {
    const resultDiv = document.getElementById('validationResult');
    if (!resultDiv) return;
    
    if (result.valid) {
        resultDiv.innerHTML = `
            <div class="monad-example monad-success">
                <strong>✓ Validation Successful</strong>
                <p>Booking ID: ${result.booking.id}</p>
                <p>Guest ID: ${result.booking.guest_id}</p>
                <p>Total: ${formatPrice(result.booking.total)}</p>
                <p>Status: ${result.booking.status}</p>
            </div>
        `;
    } else {
        resultDiv.innerHTML = `
            <div class="monad-example monad-error">
                <strong>✗ Validation Failed</strong>
                <p>Error: ${result.error}</p>
            </div>
        `;
    }
}

// --- Асинхронность / FRP ---

function startEventPolling() {
    // Poll for new events every 3 seconds
    setInterval(updateEvents, 3000);
    updateEvents(); // Initial load
}

async function updateEvents() {
    try {
        const data = await getEvents(10);
        displayEvents(data.events);
    } catch (error) {
        console.error('Failed to fetch events:', error);
    }
}

function displayEvents(events) {
    const eventList = document.getElementById('eventList');
    if (!eventList) return;
    
    if (events.length === 0) {
        eventList.innerHTML = '<div class="event-item">No events yet</div>';
        return;
    }
    
    eventList.innerHTML = events.map(event => `
        <div class="event-item slide-in">
            <strong>${escapeHtml(event.name)}</strong> 
            <span style="color: #666;">${formatTimestamp(event.timestamp)}</span>
            <br>
            <small>${escapeHtml(JSON.stringify(event.payload))}</small>
        </div>
    `).join('');
}

// --- Отчеты / Мемоизация ---

async function runMemoizationBenchmark() {
    const benchmarkBtn = document.getElementById('runBenchmark');
    const resultsDiv = document.getElementById('benchmarkResults');
    
    if (!benchmarkBtn || !resultsDiv) return;
    
    benchmarkBtn.disabled = true;
    benchmarkBtn.innerHTML = '<span class="loading"></span> Running Benchmark...';
    
    try {
        // Simulate benchmark (in real app, this would call an API)
        await simulateBenchmark();
        
        const stats = await getMemoStats();
        displayMemoStats(stats);
        
        showSuccess('Benchmark completed successfully!');
    } catch (error) {
        showError('Benchmark failed');
    } finally {
        benchmarkBtn.disabled = false;
        benchmarkBtn.textContent = 'Run Benchmark Again';
    }
}

async function simulateBenchmark() {
    // Simulate API calls to populate cache
    const params = {
        hotel_id: 'hotel_1',
        room_type_id: 'room_1', 
        rate_id: 'rate_1',
        checkin: '2024-01-01',
        checkout: '2024-01-03',
        guests: 2
    };
    
    // Make multiple calls to populate cache
    for (let i = 0; i < 5; i++) {
        await getQuote(params);
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}

function displayMemoStats(stats) {
    const statsDiv = document.getElementById('memoStats');
    if (!statsDiv) return;
    
    statsDiv.innerHTML = `
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">${stats.hits}</div>
                <div class="metric-label">Cache Hits</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${stats.misses}</div>
                <div class="metric-label">Cache Misses</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(stats.hit_ratio * 100).toFixed(1)}%</div>
                <div class="metric-label">Hit Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${stats.currsize}/${stats.maxsize}</div>
                <div class="metric-label">Cache Usage</div>
            </div>
        </div>
    `;
}

// --- Загрузка данных при старте ---

async function loadInitialData() {
    // Load any initial data needed for the current page
    const currentPage = getCurrentPage();
    
    switch (currentPage) {
        case 'reports':
            await loadReportsData();
            break;
        case 'async':
            await loadAsyncData();
            break;
        case 'functional-core':
            await loadFunctionalCoreData();
            break;
        case 'pipelines':
            // Запускаем первый поиск на странице
            applyFilters();
            break;
    }
}

function getCurrentPage() {
    const path = window.location.pathname;
    if (path.includes('reports')) return 'reports';
    if (path.includes('async')) return 'async';
    if (path.includes('functional-core')) return 'functional-core';
    if (path.includes('pipelines')) return 'pipelines';
    if (path.includes('hotels')) return 'pipelines'; // Страница hotels использует те же фильтры
    return 'overview';
}

async function loadReportsData() {
    // Load memoization stats
    try {
        const stats = await getMemoStats();
        displayMemoStats(stats);
    } catch (error) {
        console.error('Failed to load memo stats:', error);
    }
}

async function loadAsyncData() {
    // Initial events load
    await updateEvents();
}

async function loadFunctionalCoreData() {
    // Pre-load any data needed for functional core demos
}

// --- Утилиты ---

// (из Блока 2)
function initializeDateInputs() {
    // Initialize date inputs
    const today = new Date();
    const tomorrow = getTomorrow();
    
    const checkinInputs = document.querySelectorAll('input[type="date"]');
    checkinInputs.forEach(input => {
        if (input.id.includes('checkin') && !input.value) {
            input.valueAsDate = today;
        }
        if (input.id.includes('checkout') && !input.value) {
            input.valueAsDate = tomorrow;
        }
    });
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') return unsafe;
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleTimeString();
}

// (из Блока 2)
function formatDateISO(date) {
    return date.toISOString().split('T')[0];
}

// (из Блока 3 - переименовано)
function formatDateLocale(dateString) {
    return new Date(dateString).toLocaleDateString('ru-RU');
}

// (из Блока 3)
function formatPrice(amount) {
    return new Intl.NumberFormat('ru-RU', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// (из Блока 2)
function getTomorrow() {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    return tomorrow;
}

// (из Блока 2)
function calculateNights(checkin, checkout) {
    const start = new Date(checkin);
    const end = new Date(checkout);
    const diff = Math.ceil((end - start) / (1000 * 60 * 60 * 24));
    return diff > 0 ? diff : 1; // Минимум 1 ночь
}

function getHotelEmoji(stars) {
    if (stars >= 4) return '🏨';
    if (stars >= 3) return '🏢';
    return '🏠';
}

function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<div class="text-center"><span class="loading"></span> Loading...</div>';
    }
}

// (из Блока 1 - Toast-уведомления)
function showSuccess(message) {
    showMessage(message, 'success');
}

function showError(message) {
    showMessage(message, 'error');
}

function showWarning(message) {
    showMessage(message, 'warning');
}

function showMessage(message, type = 'info') {
    // Remove existing messages
    const existingMessages = document.querySelectorAll('.temp-message');
    existingMessages.forEach(msg => msg.remove());
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `temp-message ${type}-message`;
    messageDiv.innerHTML = message;
    messageDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        max-width: 300px;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(messageDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.parentNode.removeChild(messageDiv);
        }
    }, 5000);
}

// (из Блока 3 - Alert-уведомления)
function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.custom-alert');
    existingAlerts.forEach(alert => alert.remove());
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `custom-alert alert-${type}`;
    alertDiv.innerHTML = `
        <div class="alert-content">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentElement) {
            alertDiv.remove();
        }
    }, 5000);
}

// --- Модальные окна (из Блока 2) ---

function showModal(content) {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content">
            ${content}
        </div>
    `;
    
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    `;
    
    modal.onclick = (e) => {
        if (e.target === modal) {
            closeModal();
        }
    };
    
    document.body.appendChild(modal);
}

function closeModal() {
    const modal = document.querySelector('.modal-overlay');
    if (modal) {
        document.body.removeChild(modal);
    }
}

// --- Глобальные функции (для вызова из HTML) ---

window.searchHotels = searchHotels;
window.getQuote = getQuote;
window.applyFilters = applyFilters;
window.applyFiltersEnhanced = applyFiltersEnhanced; // Добавлено
window.clearFilters = clearFilters;
window.selectHotel = selectHotel;
window.testSuccessfulValidation = testSuccessfulValidation;
window.testFailedValidation = testFailedValidation;
window.runMemoizationBenchmark = runMemoizationBenchmark;
window.viewCart = viewCart;
window.addToCartWithRoom = addToCartWithRoom; // Добавлено
window.showRoomSelection = showRoomSelection; // Добавлено
window.closeModal = closeModal; // Добавлено