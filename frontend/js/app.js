// Main Frontend Controller for CycloneGuard AI
const API_BASE = (window.CYCLONE_CONFIG?.API_BASE || "").replace(/\/$/, "");

let multiHorizonChart = null;
let historicalChart = null;
let featureChart = null;
let activeStormData = null;
let playInterval = null;
let meteoDebounceTimer = null;
let healthPollTimer = null;
let isBackendReady = false;
let hasInitialPredictionSucceeded = false;
let isMetricsLoaded = false;
let isHistoricalLoaded = false;

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initPresets();
    initChips();
    initForm();
    initHealthMonitor();

    // Load initial default preset data into form inputs & map
    loadPreset('amphan_super_cyclone', false);

    // Initial prediction attempt (will also auto-retry if backend is still waking up)
    runPrediction();
});

// Navigation Tabs
function initNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const targetId = tab.getAttribute('data-tab');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');

            // Refresh Leaflet maps when visible
            if (targetId === 'tab-predictor' && window.CycloneMap && window.CycloneMap.predictorMap) {
                setTimeout(() => window.CycloneMap.predictorMap.invalidateSize(), 150);
            } else if (targetId === 'tab-radar' && window.CycloneMap) {
                setTimeout(() => {
                    if (!window.CycloneMap.radarMap) window.CycloneMap.initRadarMap();
                    window.CycloneMap.radarMap.invalidateSize();
                }, 150);
            } else if (targetId === 'tab-historical' && window.CycloneMap) {
                setTimeout(() => {
                    if (!window.CycloneMap.historyMap) window.CycloneMap.initHistoryMap();
                    window.CycloneMap.historyMap.invalidateSize();
                }, 150);
            }
        });
    });
}

// ------------------------------------------------------------------------------
// Health Monitoring & Render Cold Start Management
// ------------------------------------------------------------------------------
function initHealthMonitor() {
    checkSystemHealth();
}

async function checkSystemHealth() {
    if (healthPollTimer) clearTimeout(healthPollTimer);

    const statusText = document.getElementById('healthStatusText');
    const statusDot = document.querySelector('.status-dot');
    const featureStat = document.getElementById('featureCountStat');
    const datasetStat = document.getElementById('datasetRecordsStat');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);

        const res = await fetch(`${API_BASE}/api/health`, { signal: controller.signal });
        clearTimeout(timeoutId);

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        if (data.status === 'online') {
            isBackendReady = true;

            if (statusText) statusText.textContent = 'AI Engine Active';
            if (statusDot) {
                statusDot.className = 'status-dot';
            }
            if (featureStat && data.feature_count) {
                featureStat.textContent = `${data.feature_count} Feats`;
            }
            if (datasetStat && data.total_historical_records) {
                datasetStat.textContent = `${data.total_historical_records.toLocaleString()} Recs`;
            }

            // Lazy load metrics and historical explorer once backend is confirmed online
            if (!isMetricsLoaded) {
                loadModelMetrics();
            }
            if (!isHistoricalLoaded) {
                initHistoricalExplorer();
            }

            // If the initial prediction hasn't succeeded yet, run it now that the backend is awake
            if (!hasInitialPredictionSucceeded) {
                runPrediction();
            }

            // Schedule background keep-alive check every 30s to keep Render active
            healthPollTimer = setTimeout(checkSystemHealth, 30000);
            return;
        }
    } catch (e) {
        console.warn('Backend waking up or unreachable:', e.message);
    }

    // Backend is initializing or spun down (Render cold start)
    isBackendReady = false;
    if (statusText) statusText.textContent = 'Waking up AI Engine (Render cold start)...';
    if (statusDot) {
        statusDot.className = 'status-dot connecting';
    }

    // Fast retry loop while waking up
    healthPollTimer = setTimeout(checkSystemHealth, 3500);
}

// ------------------------------------------------------------------------------
// Open-Meteo Live Atmospheric Integration (Free API, No Key Required)
// ------------------------------------------------------------------------------
async function fetchOpenMeteoWeather(lat, lon, autoPredict = true) {
    if (isNaN(lat) || isNaN(lon)) return;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return;

    const weatherStatus = document.getElementById('weatherSyncStatus');
    if (weatherStatus) {
        weatherStatus.textContent = '🔄 Open-Meteo Syncing...';
        weatherStatus.style.opacity = '1';
    }

    try {
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${Number(lat).toFixed(4)}&longitude=${Number(lon).toFixed(4)}&current=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation&hourly=precipitation&wind_speed_unit=kmh&past_days=7&forecast_days=1`;
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000);

        const response = await fetch(url, { signal: controller.signal });
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`Open-Meteo HTTP ${response.status}`);
        }

        const data = await response.json();
        if (data && data.current) {
            const c = data.current;

            const setField = (id, val, decimals = 1) => {
                const el = document.getElementById(id);
                if (el && val !== undefined && val !== null && !isNaN(val)) {
                    el.value = Number(val).toFixed(decimals);
                    el.classList.add('flash-update');
                    setTimeout(() => el.classList.remove('flash-update'), 700);
                }
            };

            // Update exclusively environmental atmospheric and hydrological features
            setField('input_temperature_2m_c', c.temperature_2m, 1);
            setField('input_relative_humidity_pct', c.relative_humidity_2m, 1);
            setField('input_dew_point_2m_c', c.dew_point_2m, 1);
            setField('input_surface_pressure_hpa', c.surface_pressure, 1);
            setField('input_wind_speed_10m_kmh', c.wind_speed_10m, 1);
            setField('input_wind_direction_10m_deg', c.wind_direction_10m, 1);
            setField('input_wind_gusts_10m_kmh', c.wind_gusts_10m, 1);

            if (data.elevation !== undefined && data.elevation !== null) {
                setField('input_weather_elevation_m', data.elevation, 1);
            }

            setField('input_precipitation_1h_mm', c.precipitation, 1);

            // Compute actual historical accumulated precipitation intervals
            if (data.hourly && Array.isArray(data.hourly.precipitation) && Array.isArray(data.hourly.time)) {
                const precip = data.hourly.precipitation;
                const times = data.hourly.time;
                let currIdx = times.indexOf(c.time);
                if (currIdx === -1) currIdx = precip.length - 1;

                const sum24 = precip.slice(Math.max(0, currIdx - 23), currIdx + 1).reduce((acc, v) => acc + (v || 0), 0);
                setField('input_precipitation_24h_mm', sum24, 1);

                const sum72 = precip.slice(Math.max(0, currIdx - 71), currIdx + 1).reduce((acc, v) => acc + (v || 0), 0);
                setField('input_precipitation_72h_mm', sum72, 1);

                const sum168 = precip.slice(Math.max(0, currIdx - 167), currIdx + 1).reduce((acc, v) => acc + (v || 0), 0);
                setField('input_precipitation_168h_mm', sum168, 1);
            }

            if (weatherStatus) {
                weatherStatus.textContent = `⚡ Open-Meteo Live (${Number(lat).toFixed(1)}°, ${Number(lon).toFixed(1)}°)`;
                weatherStatus.style.opacity = '1';
            }

            // Auto-refresh predictions if requested
            if (autoPredict) {
                runPrediction();
            }
        }
    } catch (err) {
        console.warn('Open-Meteo live sync notice:', err.message);
        if (weatherStatus) {
            weatherStatus.textContent = '⚠️ Weather Offline (Manual Preserved)';
            weatherStatus.style.opacity = '0.9';
        }
    }
}

function handleCoordinateChange(lat, lon, source = 'input', debounceMs = 350) {
    if (isNaN(lat) || isNaN(lon)) return;

    // Update latitude and longitude input controls if change came from map or preset
    if (source !== 'input') {
        const latInput = document.getElementById('input_latitude');
        const lonInput = document.getElementById('input_longitude');
        if (latInput) latInput.value = Number(lat).toFixed(2);
        if (lonInput) lonInput.value = Number(lon).toFixed(2);
    }

    // Update Leaflet map center and marker position
    if (source !== 'map' && window.CycloneMap) {
        window.CycloneMap.updatePredictorMarker(lat, lon);
    }

    // Debounce live weather fetch
    if (meteoDebounceTimer) clearTimeout(meteoDebounceTimer);
    if (debounceMs > 0) {
        meteoDebounceTimer = setTimeout(() => {
            fetchOpenMeteoWeather(lat, lon, true);
        }, debounceMs);
    } else {
        fetchOpenMeteoWeather(lat, lon, true);
    }
}

// ------------------------------------------------------------------------------
// Preset Loader
// ------------------------------------------------------------------------------
function initPresets() {
    const selector = document.getElementById('presetSelector');
    if (!selector) return;

    selector.addEventListener('change', (e) => {
        loadPreset(e.target.value, true);
    });
}

function loadPreset(key, triggerPrediction = true) {
    if (!window.CYCLONE_PRESETS || !window.CYCLONE_PRESETS[key]) return;
    const preset = window.CYCLONE_PRESETS[key];
    const data = preset.data;

    const descEl = document.getElementById('presetDesc');
    if (descEl) descEl.textContent = preset.desc;

    // Populate inputs
    for (const [k, v] of Object.entries(data)) {
        const el = document.getElementById(`input_${k}`);
        if (el) {
            el.value = v;
        }
    }

    // Update Horizon chip active state
    const horizon = data.forecast_horizon_h || 24;
    document.querySelectorAll('.horizon-chip').forEach(chip => {
        if (parseInt(chip.getAttribute('data-horizon')) === horizon) {
            chip.classList.add('active');
        } else {
            chip.classList.remove('active');
        }
    });

    // Update Map
    if (window.CycloneMap) {
        if (!window.CycloneMap.predictorMap) {
            window.CycloneMap.initPredictorMap(data.latitude, data.longitude, (lat, lon, src) => {
                handleCoordinateChange(lat, lon, src || 'map');
            });
        } else {
            window.CycloneMap.updatePredictorMarker(data.latitude, data.longitude);
        }
    }

    if (triggerPrediction) {
        runPrediction();
    }
}

// ------------------------------------------------------------------------------
// Horizon Chips (Lead times: 3, 6, 9, 12, 18, 24, 36, 48)
// ------------------------------------------------------------------------------
function initChips() {
    const chips = document.querySelectorAll('.horizon-chip');
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            chips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            const h = chip.getAttribute('data-horizon');
            const hiddenInput = document.getElementById('input_forecast_horizon_h');
            if (hiddenInput) hiddenInput.value = h;
            runPrediction();
        });
    });
}

// ------------------------------------------------------------------------------
// Form Telemetry Collector
// ------------------------------------------------------------------------------
function getFormData() {
    const fields = [
        'latitude', 'longitude', 'season', 'forecast_horizon_h',
        'storm_speed_kts', 'storm_direction_deg', 'movement_distance_from_previous_km',
        'wind_change_from_previous_kts', 'pressure_change_from_previous_hpa',
        'wmo_wind_kts', 'wmo_pressure_hpa', 'imd_newdelhi_wind_kts', 'imd_newdelhi_pressure_hpa',
        'usa_wind_kts', 'usa_pressure_hpa', 'weather_elevation_m', 'temperature_2m_c',
        'relative_humidity_pct', 'dew_point_2m_c', 'surface_pressure_hpa',
        'wind_speed_10m_kmh', 'wind_direction_10m_deg', 'wind_gusts_10m_kmh',
        'precipitation_1h_mm', 'precipitation_24h_mm', 'precipitation_72h_mm', 'precipitation_168h_mm',
        'soil_moisture_0_7cm_m3m3', 'landfall_indicator', 'distance_to_land_source'
    ];

    const data = {};
    fields.forEach(f => {
        const el = document.getElementById(`input_${f}`);
        if (el) {
            data[f] = parseFloat(el.value) || 0;
        }
    });
    return data;
}

function initForm() {
    const form = document.getElementById('telemetryForm');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            runPrediction();
        });
    }

    // Coordinate inputs manual change sync with map and Open-Meteo live weather
    const latInput = document.getElementById('input_latitude');
    const lonInput = document.getElementById('input_longitude');
    if (latInput && lonInput) {
        const onCoordInput = () => {
            const lat = parseFloat(latInput.value);
            const lon = parseFloat(lonInput.value);
            if (!isNaN(lat) && !isNaN(lon)) {
                handleCoordinateChange(lat, lon, 'input', 400);
            }
        };
        latInput.addEventListener('input', onCoordInput);
        latInput.addEventListener('change', onCoordInput);
        lonInput.addEventListener('input', onCoordInput);
        lonInput.addEventListener('change', onCoordInput);
    }
}

// ------------------------------------------------------------------------------
// Prediction Execution & Forecast Rendering
// ------------------------------------------------------------------------------
async function runPrediction() {
    const payload = getFormData();
    const btn = document.getElementById('btnPredict');
    const errorBanner = document.getElementById('predictionErrorBanner');

    if (btn) {
        btn.innerHTML = '<span class="loading-spin">🌀</span> <span>Forecasting with XGBoost...</span>';
        btn.disabled = true;
    }

    // Provide visual loading hint on stat elements if they are empty
    const windKtsEl = document.getElementById('resultWindKts');
    const windKmhEl = document.getElementById('resultWindKmh');
    const deltaEl = document.getElementById('resultDeltaKts');
    if (windKtsEl && windKtsEl.textContent.trim() === '-- kts') windKtsEl.textContent = '...';
    if (windKmhEl && windKmhEl.textContent.trim() === '-- km/h') windKmhEl.textContent = '...';
    if (deltaEl && deltaEl.textContent.trim() === '-- kts') deltaEl.textContent = '...';

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 45000); // 45s for Render cold start tolerance

        // Run single horizon and full multi-horizon trajectory in parallel
        const [resSingle, resMulti] = await Promise.all([
            fetch(`${API_BASE}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: controller.signal
            }),
            fetch(`${API_BASE}/api/predict/multi-horizon`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: controller.signal
            })
        ]);

        clearTimeout(timeoutId);

        if (!resSingle.ok) throw new Error(`Single prediction failed (HTTP ${resSingle.status})`);
        if (!resMulti.ok) throw new Error(`Multi-horizon forecast failed (HTTP ${resMulti.status})`);

        const singleData = await resSingle.json();
        const multiData = await resMulti.json();

        if (!singleData.success || !singleData.prediction) {
            throw new Error(singleData.error || 'Invalid single prediction response');
        }
        if (!multiData.success || !multiData.forecast) {
            throw new Error(multiData.error || 'Invalid multi-horizon response');
        }

        hasInitialPredictionSucceeded = true;
        isBackendReady = true;

        // Render forecast deck and chart
        renderPredictionResults(singleData.prediction);
        renderMultiHorizonChart(multiData.forecast);

        if (window.CycloneMap) {
            window.CycloneMap.renderForecastTrack(multiData.forecast);
        }

        // Clear error banner on success
        if (errorBanner) {
            errorBanner.style.display = 'none';
            errorBanner.textContent = '';
        }

        // Update health text to active
        const statusText = document.getElementById('healthStatusText');
        const statusDot = document.querySelector('.status-dot');
        if (statusText) statusText.textContent = 'AI Engine Active';
        if (statusDot) statusDot.className = 'status-dot';

    } catch (e) {
        console.error('Prediction failed:', e);

        // Never display fabricated or fake predictions
        if (windKtsEl && windKtsEl.textContent === '...') windKtsEl.textContent = '-- kts';
        if (windKmhEl && windKmhEl.textContent === '...') windKmhEl.textContent = '-- km/h';
        if (deltaEl && deltaEl.textContent === '...') deltaEl.textContent = '-- kts';

        if (errorBanner) {
            errorBanner.style.display = 'block';
            if (!isBackendReady) {
                errorBanner.innerHTML = '⏳ <b>AI Backend is starting up on Render</b> (cold start takes ~30–50s). Predictions will automatically populate once the server is ready.';
            } else {
                errorBanner.innerHTML = `⚠️ <b>Forecast request could not be completed:</b> ${e.message || 'API error'}. Please check your inputs or retry.`;
            }
        }

        // If backend was offline, kick off health check to auto-refresh once awake
        if (!isBackendReady) {
            checkSystemHealth();
        }
    } finally {
        if (btn) {
            btn.innerHTML = '<span>⚡ Execute Cyclone AI Inference</span>';
            btn.disabled = false;
        }
    }
}

// Render Results Deck
function renderPredictionResults(p) {
    const cat = p.category;
    const catBadge = document.getElementById('categoryBadge');
    if (catBadge && cat) {
        catBadge.textContent = `${cat.name} (${cat.code})`;
        catBadge.style.backgroundColor = cat.color || '#dc2626';
        catBadge.style.color = '#ffffff';
    }

    const windKtsEl = document.getElementById('resultWindKts');
    const windKmhEl = document.getElementById('resultWindKmh');
    const deltaEl = document.getElementById('resultDeltaKts');
    const modelEl = document.getElementById('resultModelUsed');

    if (windKtsEl) windKtsEl.textContent = `${p.predicted_wind_kts} kts`;
    if (windKmhEl) windKmhEl.textContent = `${p.predicted_wind_kmh} km/h`;
    if (deltaEl) {
        const sign = p.wind_change_kts > 0 ? '+' : '';
        deltaEl.textContent = `${sign}${p.wind_change_kts} kts`;
    }
    if (modelEl) modelEl.textContent = p.model_used || "Tuned XGBoost Regression";
}

// Multi-Horizon Chart using cycloneguard_xgboost_final.joblib predictions
function renderMultiHorizonChart(forecast) {
    const ctx = document.getElementById('multiHorizonChart');
    if (!ctx || !forecast || !forecast.horizons) return;

    const horizons = forecast.horizons;
    const labels = horizons.map(h => `+${h.forecast_horizon_h}h`);
    const predictedWinds = horizons.map(h => h.predicted_wind_kts);
    const initialWind = forecast.initial_point ? forecast.initial_point.wind_kts : predictedWinds[0];
    const currentWind = horizons.map(() => initialWind);

    if (multiHorizonChart) {
        multiHorizonChart.destroy();
        multiHorizonChart = null;
    }

    multiHorizonChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Tuned XGBoost Regression (kts)',
                    data: predictedWinds,
                    borderColor: '#00f2fe',
                    backgroundColor: 'rgba(0, 242, 254, 0.15)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.35,
                    pointBackgroundColor: '#00f2fe',
                    pointRadius: 5,
                    pointHoverRadius: 7
                },
                {
                    label: 'Initial Eye Wind Speed (kts)',
                    data: currentWind,
                    borderColor: 'rgba(255, 255, 255, 0.35)',
                    borderDash: [4, 4],
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Outfit', size: 12 } }
                },
                tooltip: {
                    callbacks: {
                        afterBody: function(context) {
                            const idx = context[0].dataIndex;
                            const h = horizons[idx];
                            if (!h) return '';
                            return `Metric: ${h.predicted_wind_kmh} km/h\nCategory: ${h.category?.name || 'N/A'} (${h.category?.code || ''})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Forecast Horizon Lead Time', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
                },
                y: {
                    title: { display: true, text: 'Wind Speed (Knots)', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
                }
            }
        }
    });
}

// ------------------------------------------------------------------------------
// Historical Explorer & Timeline Player
// ------------------------------------------------------------------------------
async function initHistoricalExplorer() {
    const stormSelect = document.getElementById('historyStormSelect');
    if (!stormSelect) return;

    try {
        const res = await fetch(`${API_BASE}/api/historical/storms`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        stormSelect.innerHTML = '';
        if (data.storms && data.storms.length > 0) {
            isHistoricalLoaded = true;
            data.storms.forEach(s => {
                const opt = document.createElement('option');
                opt.value = s.storm_id;
                opt.textContent = `${s.storm_name} (${s.season}) - Peak: ${s.peak_wind_kts} kts [${s.peak_category?.code || ''}]`;
                stormSelect.appendChild(opt);
            });

            stormSelect.addEventListener('change', (e) => {
                loadHistoricalStorm(e.target.value);
            });

            loadHistoricalStorm(data.storms[0].storm_id);
        } else {
            stormSelect.innerHTML = '<option>No historical records available</option>';
        }
    } catch (e) {
        console.warn('Failed to load historical storms:', e.message);
    }

    const scrubber = document.getElementById('historyScrubber');
    if (scrubber) {
        scrubber.addEventListener('input', (e) => {
            updatePlaybackPoint(parseInt(e.target.value));
        });
    }

    const playBtn = document.getElementById('btnPlayHistory');
    if (playBtn) {
        playBtn.addEventListener('click', togglePlayback);
    }
}

async function loadHistoricalStorm(stormId) {
    try {
        const res = await fetch(`${API_BASE}/api/historical/storm/${stormId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!data.success || !data.timeline) return;

        activeStormData = data.timeline;
        const pts = activeStormData.points;

        const scrubber = document.getElementById('historyScrubber');
        if (scrubber) {
            scrubber.max = Math.max(0, pts.length - 1);
            scrubber.value = 0;
        }

        if (window.CycloneMap) {
            window.CycloneMap.renderHistoricalStorm(pts, activeStormData.storm_name);
        }

        renderHistoricalChart(pts, activeStormData.storm_name);
        updatePlaybackPoint(0);
    } catch (e) {
        console.error('Error loading storm timeline:', e);
    }
}

function updatePlaybackPoint(idx) {
    if (!activeStormData || !activeStormData.points || !activeStormData.points[idx]) return;
    const pt = activeStormData.points[idx];

    const timeEl = document.getElementById('histPointTime');
    const actualEl = document.getElementById('histActualWind');
    const predEl = document.getElementById('histPredWind');
    const errorEl = document.getElementById('histError');
    const catEl = document.getElementById('histCategory');

    if (timeEl) timeEl.textContent = pt.timestamp || `Step #${idx+1}`;
    if (actualEl) actualEl.textContent = `${pt.actual_target_wind_kts} kts`;
    if (predEl) predEl.textContent = `${pt.predicted_wind_kts} kts`;
    if (errorEl) errorEl.textContent = `${pt.error_kts} kts`;
    if (catEl) catEl.textContent = `${pt.category ? pt.category.name : 'N/A'}`;

    if (window.CycloneMap) {
        window.CycloneMap.highlightHistoryPoint(idx);
    }
}

function togglePlayback() {
    const playBtn = document.getElementById('btnPlayHistory');
    const scrubber = document.getElementById('historyScrubber');
    if (!playBtn || !scrubber || !activeStormData) return;

    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
        playBtn.textContent = '▶ Play Track';
    } else {
        playBtn.textContent = '⏸ Pause';
        playInterval = setInterval(() => {
            let cur = parseInt(scrubber.value);
            if (cur >= activeStormData.points.length - 1) {
                cur = 0;
            } else {
                cur++;
            }
            scrubber.value = cur;
            updatePlaybackPoint(cur);
        }, 800);
    }
}

function renderHistoricalChart(points, stormName) {
    const ctx = document.getElementById('historicalChart');
    if (!ctx || !points) return;

    const labels = points.map((p, i) => `#${i+1}`);
    const actuals = points.map(p => p.actual_target_wind_kts);
    const preds = points.map(p => p.predicted_wind_kts);

    if (historicalChart) {
        historicalChart.destroy();
        historicalChart = null;
    }

    historicalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Actual Wind Speed (kts)',
                    data: actuals,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.15)',
                    borderWidth: 2.5,
                    fill: false,
                    tension: 0.2,
                    pointRadius: 3
                },
                {
                    label: 'AI Model Predicted Wind (kts)',
                    data: preds,
                    borderColor: '#f59e0b',
                    borderWidth: 2.5,
                    borderDash: [4, 4],
                    fill: false,
                    tension: 0.2,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Outfit' } } }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Timeline Observations', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    title: { display: true, text: 'Wind Speed (Knots)', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

// ------------------------------------------------------------------------------
// Model Benchmarks & Feature Importance
// ------------------------------------------------------------------------------
async function loadModelMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/metrics`);
        if (!res.ok) return;
        const data = await res.json();
        const m = data.metrics;

        if (m && m.tuned_xgboost) {
            isMetricsLoaded = true;
            const maeEl = document.getElementById('metricXgbMae');
            const r2El = document.getElementById('metricXgbR2');
            if (maeEl) maeEl.textContent = `${m.tuned_xgboost.mae} kts`;
            if (r2El) r2El.textContent = `${m.tuned_xgboost.r2}`;
        }

        // Top features chart
        if (data.top_features) {
            renderFeatureImportance(data.top_features);
        }
    } catch (e) {
        console.warn('Failed to load metrics:', e.message);
    }
}

function renderFeatureImportance(features) {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx || !features) return;

    const top15 = features.slice(0, 12);
    const labels = top15.map(f => f.feature);
    const vals = top15.map(f => (f.importance * 100).toFixed(1));

    if (featureChart) {
        featureChart.destroy();
        featureChart = null;
    }

    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Relative Feature Importance (%)',
                data: vals,
                backgroundColor: 'rgba(0, 242, 254, 0.75)',
                borderColor: '#00f2fe',
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f8fafc', font: { family: 'JetBrains Mono', size: 11 } }
                }
            }
        }
    });
}
