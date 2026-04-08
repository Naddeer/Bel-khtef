document.addEventListener('DOMContentLoaded', () => {
    const grid = document.getElementById('vehicleGrid');
    const searchInput = document.getElementById('searchInput');
    const filterBtns = document.querySelectorAll('.filter-btn');

    let allVehicles = [];

    // Load Data
    fetch('/api/vehicles')
        .then(response => response.json())
        .then(data => {
            allVehicles = data;
            renderVehicles(allVehicles);
        })
        .catch(err => {
            console.error("Error loading vehicles:", err);
            grid.innerHTML = '<p class="error">Failed to load vehicle data. Please try again later.</p>';
        });

    function renderVehicles(vehicles) {
        grid.innerHTML = '';

        if (vehicles.length === 0) {
            grid.innerHTML = '<p class="no-results">No vehicles found matching your criteria.</p>';
            return;
        }

        vehicles.forEach(vehicle => {
            const card = document.createElement('div');
            card.className = 'card';

            // Format price
            let displayPrice = vehicle.price_tnd
                ? vehicle.price_tnd.toLocaleString() + ' DT'
                : (vehicle.raw_price || 'N/A');

            // Check if it's a "luxury" car (simplified logic for demo)
            const isLuxury = ["audi", "mercedes", "jeep", "bmw"].some(brand =>
                vehicle.title.toLowerCase().includes(brand)
            );

            card.innerHTML = `
                <div class="image-container">
                    <img src="${vehicle.image || 'https://via.placeholder.com/400x300?text=No+Image'}" alt="${vehicle.title}" loading="lazy">
                    ${vehicle.year ? `<span class="year-badge">${vehicle.year}</span>` : ''}
                    ${vehicle.is_good_deal ? `<span class="badge-deal">🔥 Good Deal</span>` : ''}
                </div>
                <div class="card-content">
                    <h3 class="card-title">${vehicle.title}</h3>
                    <div class="card-price">${displayPrice}</div>
                    <div class="card-footer">
                        <a href="${vehicle.link}" target="_blank" class="view-btn">
                            View on Tayara
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                        </a>
                    </div>
                </div>
            `;
            grid.appendChild(card);
        });
    }

    // Search functionality
    searchInput.addEventListener('input', (e) => {
        const term = e.target.value.toLowerCase();
        const filtered = allVehicles.filter(v =>
            v.title.toLowerCase().includes(term) ||
            (v.year && v.year.toString().includes(term)) ||
            (v.price_tnd && v.price_tnd.toString().includes(term))
        );
        renderVehicles(filtered);
    });

    // Filter functionality
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active state
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const filter = btn.dataset.filter;
            let filtered = [];

            if (filter === 'all') {
                filtered = allVehicles;
            } else if (filter === '2023') {
                filtered = allVehicles.filter(v => v.year === 2023);
            } else if (filter === '2019') {
                filtered = allVehicles.filter(v => v.year && v.year >= 2019);
            } else if (filter === 'luxury') {
                filtered = allVehicles.filter(v =>
                    ["audi", "mercedes", "jeep", "bmw"].some(brand =>
                        v.title.toLowerCase().includes(brand)
                    )
                );
            } else if (filter === 'good_deals') {
                filtered = allVehicles.filter(v => v.is_good_deal === true);
            }

            renderVehicles(filtered);
        });
    });
});
