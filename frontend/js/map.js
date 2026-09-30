// Map Manager for CycloneGuard AI using Leaflet
class CycloneMapManager {
    constructor() {
        this.predictorMap = null;
        this.predictorMarker = null;
        this.radarMap = null;
        this.radarLayers = {
            track: null,
            cones: [],
            markers: [],
            dangerRings: []
        };
        this.historyMap = null;
        this.historyLayers = {
            path: null,
            markers: []
        };
    }

    initPredictorMap(initialLat = 15.5, initialLon = 87.5, onCoordChange = null) {
        const container = document.getElementById('predictorMiniMap');
        if (!container || this.predictorMap) return;

        this.predictorMap = L.map('predictorMiniMap', {
            center: [initialLat, initialLon],
            zoom: 5,
            zoomControl: true,
            attributionControl: false
        });

        // Dark Matter tiles for meteorology command center feel
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 18,
            subdomains: 'abcd'
        }).addTo(this.predictorMap);

        // Custom Cyclone Center Pulsing Icon
        const cycloneIcon = L.divIcon({
            className: 'cyclone-pulse-icon',
            html: '<div class="pulse-ring"></div><div class="pulse-core">🌪️</div>',
            iconSize: [36, 36],
            iconAnchor: [18, 18]
        });

        this.predictorMarker = L.marker([initialLat, initialLon], {
            draggable: true,
            icon: cycloneIcon
        }).addTo(this.predictorMap);

        this.predictorMarker.on('dragend', (e) => {
            const pos = e.target.getLatLng();
            if (onCoordChange) onCoordChange(pos.lat, pos.lng);
        });

        this.predictorMap.on('click', (e) => {
            this.predictorMarker.setLatLng(e.latlng);
            if (onCoordChange) onCoordChange(e.latlng.lat, e.latlng.lng);
        });
    }

    updatePredictorMarker(lat, lon) {
        if (this.predictorMarker && this.predictorMap) {
            this.predictorMarker.setLatLng([lat, lon]);
            this.predictorMap.panTo([lat, lon]);
        }
    }

    initRadarMap() {
        const container = document.getElementById('radarMap');
        if (!container || this.radarMap) return;

        this.radarMap = L.map('radarMap', {
            center: [16.0, 86.5],
            zoom: 5,
            zoomControl: true,
            attributionControl: false
        });

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 18,
            subdomains: 'abcd'
        }).addTo(this.radarMap);

        // Add Bay of Bengal / Arabian Sea key coastal landmarks
        const ports = [
            { name: "Paradip (Odisha)", coords: [20.26, 86.66], alert: "High Watch" },
            { name: "Visakhapatnam (AP)", coords: [17.68, 83.21], alert: "Alert Zone" },
            { name: "Chennai (TN)", coords: [13.08, 80.27], alert: "Watch" },
            { name: "Kolkata / Haldia (WB)", coords: [22.02, 88.06], alert: "Surge Risk" },
            { name: "Chittagong (BD)", coords: [22.33, 91.83], alert: "Watch" },
            { name: "Puri (Odisha)", coords: [19.81, 85.83], alert: "High Alert" }
        ];

        ports.forEach(p => {
            L.circleMarker(p.coords, {
                radius: 5,
                color: '#38bdf8',
                fillColor: '#0284c7',
                fillOpacity: 0.8
            }).bindTooltip(`<b>${p.name}</b><br>Coastal Hub: ${p.alert}`).addTo(this.radarMap);
        });
    }

    renderForecastTrack(forecastData) {
        if (!this.radarMap) this.initRadarMap();
        if (!this.radarMap || !forecastData || !forecastData.horizons) return;

        // Clear existing layers
        if (this.radarLayers.track) this.radarMap.removeLayer(this.radarLayers.track);
        this.radarLayers.cones.forEach(l => this.radarMap.removeLayer(l));
        this.radarLayers.markers.forEach(l => this.radarMap.removeLayer(l));
        this.radarLayers.dangerRings.forEach(l => this.radarMap.removeLayer(l));
        this.radarLayers.cones = [];
        this.radarLayers.markers = [];
        this.radarLayers.dangerRings = [];

        const initial = forecastData.initial_point;
        const horizons = forecastData.horizons;

        const latlngs = [[initial.latitude, initial.longitude]];
        
        // Add initial marker
        const initMarker = L.circleMarker([initial.latitude, initial.longitude], {
            radius: 9,
            color: '#ffffff',
            fillColor: initial.category.color,
            fillOpacity: 1,
            weight: 2
        }).bindPopup(`<b>Current Storm Eye</b><br>Intensity: ${initial.wind_kts} kts (${initial.category.name})`).addTo(this.radarMap);
        this.radarLayers.markers.push(initMarker);

        // Render forecast trajectory and uncertainty cones
        horizons.forEach((h, index) => {
            const pt = [h.projected_latitude, h.projected_longitude];
            latlngs.push(pt);

            // Uncertainty circle / cone
            const coneCircle = L.circle(pt, {
                radius: h.cone_radius_km * 1000,
                color: h.category.color,
                fillColor: h.category.color,
                fillOpacity: 0.12,
                weight: 1,
                dashArray: '4, 4'
            }).addTo(this.radarMap);
            this.radarLayers.cones.push(coneCircle);

            // Point marker
            const marker = L.circleMarker(pt, {
                radius: 7,
                color: '#ffffff',
                fillColor: h.category.color,
                fillOpacity: 0.9,
                weight: 1.5
            }).bindPopup(`
                <div style="font-family: 'Outfit', sans-serif; min-width: 170px;">
                    <div style="font-weight:700; color: ${h.category.color}; font-size:14px;">+${h.forecast_horizon_h}h Forecast</div>
                    <div><b>Predicted Wind:</b> ${h.predicted_wind_kts} kts</div>
                    <div><b>Metric Speed:</b> ${h.predicted_wind_kmh} km/h</div>
                    <div><b>Category:</b> ${h.category.name} [${h.category.code}]</div>
                    <div><b>Engine:</b> Tuned XGBoost</div>
                </div>
            `).addTo(this.radarMap);
            this.radarLayers.markers.push(marker);
        });

        // Polylines with glowing gradient style
        this.radarLayers.track = L.polyline(latlngs, {
            color: '#00f2fe',
            weight: 4,
            opacity: 0.9,
            dashArray: '6, 6'
        }).addTo(this.radarMap);

        this.radarMap.fitBounds(L.latLngBounds(latlngs).pad(0.3));
    }

    initHistoryMap() {
        const container = document.getElementById('historyMap');
        if (!container || this.historyMap) return;

        this.historyMap = L.map('historyMap', {
            center: [16.5, 87.0],
            zoom: 5,
            zoomControl: true,
            attributionControl: false
        });

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 18,
            subdomains: 'abcd'
        }).addTo(this.historyMap);
    }

    renderHistoricalStorm(points, stormName) {
        if (!this.historyMap) this.initHistoryMap();
        if (!this.historyMap || !points || points.length === 0) return;

        // Clear existing
        if (this.historyLayers.path) this.historyMap.removeLayer(this.historyLayers.path);
        this.historyLayers.markers.forEach(m => this.historyMap.removeLayer(m));
        this.historyLayers.markers = [];

        const latlngs = [];
        points.forEach((p, idx) => {
            const pt = [p.latitude, p.longitude];
            latlngs.push(pt);

            const marker = L.circleMarker(pt, {
                radius: 6,
                color: '#ffffff',
                fillColor: p.category ? p.category.color : '#38bdf8',
                fillOpacity: 0.85,
                weight: 1
            }).bindPopup(`
                <div style="font-family: 'Outfit', sans-serif;">
                    <div style="font-weight:700; color:#38bdf8;">${stormName} (Pt #${idx+1})</div>
                    <div><b>Time:</b> ${p.timestamp || 'N/A'}</div>
                    <div><b>Actual Wind:</b> ${p.actual_target_wind_kts} kts (${p.category ? p.category.code : ''})</div>
                    <div><b>AI Predicted:</b> ${p.predicted_wind_kts} kts</div>
                    <div><b>Abs Error:</b> ${p.error_kts} kts</div>
                </div>
            `).addTo(this.historyMap);

            this.historyLayers.markers.push(marker);
        });

        this.historyLayers.path = L.polyline(latlngs, {
            color: '#38bdf8',
            weight: 3,
            opacity: 0.8
        }).addTo(this.historyMap);

        this.historyMap.fitBounds(L.latLngBounds(latlngs).pad(0.2));
    }

    highlightHistoryPoint(index) {
        if (this.historyLayers.markers[index]) {
            this.historyLayers.markers[index].openPopup();
            this.historyMap.panTo(this.historyLayers.markers[index].getLatLng());
        }
    }
}

window.CycloneMap = new CycloneMapManager();
