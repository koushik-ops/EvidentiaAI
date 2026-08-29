/**
 * DriveOps Forensic AI — Modern Web UI Client
 * Handles telemetry upload, Chart.js telemetry rendering, AI predictions, and PDF reports.
 */

// Global State
let telemetryChartInstance = null;
let currentAnalysisData = null;
let currentChartChannel = 'all';

// Color Mappings
const BEHAVIOR_COLORS = {
    'NORMAL': '#10b981',
    'AGGRESSIVE': '#ef4444',
    'SUDDEN_BRAKING': '#f59e0b',
    'HIGH_SPEED_TURN': '#a855f7',
    'OVERSPEEDING': '#ec4899',
    'JERK_SPIKE': '#3b82f6'
};

document.addEventListener('DOMContentLoaded', () => {
    checkSystemStatus();
    loadIncidentHistory();
    initCloudStatus();
    
    // Close dropdown if clicked outside
    document.addEventListener('click', (e) => {
        const demoDropdown = document.querySelector('.quick-demo-dropdown');
        if (demoDropdown && !demoDropdown.contains(e.target)) {
            document.getElementById('demo-menu').classList.remove('show');
        }
    });
});

/**
 * Check backend API health
 */
async function checkSystemStatus() {
    try {
        const res = await fetch('/api/status');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('engine-status').textContent = 'ONLINE';
        }
    } catch (e) {
        document.getElementById('engine-status').textContent = 'OFFLINE';
        document.getElementById('engine-status').style.color = '#ef4444';
    }
}

/**
 * Toggle quick demo dropdown
 */
function toggleDemoMenu() {
    const menu = document.getElementById('demo-menu');
    menu.classList.toggle('show');
}

/**
 * Drag and Drop handlers
 */
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('dropzone').classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('dropzone').classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('dropzone').classList.remove('dragover');
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        uploadAndAnalyzeFile(e.dataTransfer.files[0]);
    }
}

function handleFileSelected(e) {
    if (e.target.files && e.target.files.length > 0) {
        uploadAndAnalyzeFile(e.target.files[0]);
    }
}

/**
 * Send file to /api/analyze with stage animations
 */
async function uploadAndAnalyzeFile(file) {
    const validExts = ['.xlsx', '.xls', '.csv', '.json'];
    const fileName = file.name.toLowerCase();
    const isValid = validExts.some(ext => fileName.endsWith(ext));

    if (!isValid) {
        showToast('Please upload a valid .xlsx, .csv, or .json telemetry file.', 'error');
        return;
    }

    const overlay = document.getElementById('analyzing-overlay');
    const stageText = document.getElementById('loading-stage');
    overlay.classList.add('active');

    // Simulate animated pipeline stages
    stageText.textContent = `Ingesting ${file.name}...`;
    const stageTimer1 = setTimeout(() => { stageText.textContent = "Extracting IMU & Kinematic Sensor Channels..."; }, 700);
    const stageTimer2 = setTimeout(() => { stageText.textContent = "Running XGBoost Multi-Class Behavior Classifier..."; }, 1500);
    const stageTimer3 = setTimeout(() => { stageText.textContent = "Evaluating Accident Reconstruction Heuristics..."; }, 2400);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        clearTimeout(stageTimer1);
        clearTimeout(stageTimer2);
        clearTimeout(stageTimer3);

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Analysis failed on server.');
        }

        const data = await response.json();
        renderAnalysisResults(data);
        showToast(`Analysis completed for ${file.name}`, 'success');
        loadIncidentHistory();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        overlay.classList.remove('active');
    }
}

/**
 * Load Demo Sample
 */
async function loadDemoSample(sampleType) {
    document.getElementById('demo-menu').classList.remove('show');
    const overlay = document.getElementById('analyzing-overlay');
    const stageText = document.getElementById('loading-stage');
    overlay.classList.add('active');
    stageText.textContent = `Loading ${sampleType.toUpperCase()} reference telemetry...`;

    try {
        const res = await fetch(`/api/demo?type=${encodeURIComponent(sampleType)}`);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Failed to load demo file.');
        }
        const data = await res.json();
        renderAnalysisResults(data);
        showToast(`Loaded ${sampleType.toUpperCase()} sample telemetry!`, 'success');
        loadIncidentHistory();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        overlay.classList.remove('active');
    }
}

/**
 * Render Complete Analysis Results
 */
function renderAnalysisResults(data) {
    currentAnalysisData = data;
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.style.display = 'flex';
    resultsContainer.scrollIntoView({ behavior: 'smooth' });

    // 1. Header Banner
    document.getElementById('badge-incident-id').textContent = `CASE #${data.id}`;
    document.getElementById('badge-format').textContent = data.input_kind || 'Telemetry Stream';
    
    const severityBadge = document.getElementById('badge-severity');
    severityBadge.textContent = `${data.severity} SEVERITY`;
    severityBadge.className = 'severity-badge';
    if (data.severity === 'HIGH') severityBadge.classList.add('amber');
    else if (data.severity === 'LOW') severityBadge.classList.add('green');

    document.getElementById('incident-primary-cause').textContent = data.primary_cause;
    document.getElementById('incident-primary-reason').textContent = data.primary_reason;

    // Safety Score Ring Animation
    animateSafetyScore(data.safety_score);

    // PDF Actions
    const downloadBtn = document.getElementById('btn-download-pdf');
    if (data.pdf_report) {
        downloadBtn.href = `${data.pdf_report}?download=1`;
        downloadBtn.style.display = 'inline-flex';
    } else {
        downloadBtn.style.display = 'none';
    }

    // 2. Metric KPI Cards
    const metrics = data.metrics || {};
    document.getElementById('val-peak-g').textContent = metrics.peak_g_force !== undefined ? metrics.peak_g_force.toFixed(2) : '0.00';
    
    // G-Force threshold evaluation badge
    const peakGVal = metrics.peak_g_force || 0;
    const footerPeakG = document.getElementById('footer-peak-g');
    if (peakGVal >= 4.0) {
        footerPeakG.innerHTML = '<span class="badge-threshold red">Critical Crash Impact (≥ 4.0G)</span>';
    } else if (peakGVal >= 1.8) {
        footerPeakG.innerHTML = '<span class="badge-threshold amber">High Lateral / Decel Spike (≥ 1.8G)</span>';
    } else {
        footerPeakG.innerHTML = '<span class="badge-threshold green">Within Safe Limits (&lt; 1.8G)</span>';
    }

    document.getElementById('val-speed-drop').textContent = metrics.speed_drop !== undefined ? metrics.speed_drop.toFixed(1) : '0.0';
    document.getElementById('val-max-speed').textContent = metrics.max_speed !== undefined ? metrics.max_speed.toFixed(1) : '0.0';
    document.getElementById('val-avg-speed').textContent = metrics.avg_speed !== undefined ? metrics.avg_speed.toFixed(1) : '0.0';

    document.getElementById('val-peak-yaw').textContent = metrics.peak_yaw !== undefined ? metrics.peak_yaw.toFixed(1) : '0.0';
    const footerYaw = document.getElementById('footer-yaw');
    if ((metrics.peak_yaw || 0) > 60) {
        footerYaw.className = 'badge-threshold red';
        footerYaw.textContent = 'Severe Yaw / Spin-Out Detected';
    } else if ((metrics.peak_yaw || 0) > 30) {
        footerYaw.className = 'badge-threshold amber';
        footerYaw.textContent = 'Moderate Swerve Observed';
    } else {
        footerYaw.className = 'badge-threshold green';
        footerYaw.textContent = 'Lateral Stability Normal';
    }

    document.getElementById('val-peak-jerk').textContent = metrics.peak_jerk !== undefined ? metrics.peak_jerk.toFixed(1) : '0.0';
    document.getElementById('val-samples-info').textContent = `${metrics.total_samples || 0} Telemetry Sample Frames`;

    // 3. Cause Probabilities
    document.getElementById('badge-cause-conf').textContent = `${data.confidence}% AI Confidence`;
    renderCauseProbabilities(data.causes || [], data.primary_cause_key);

    // 4. Driver Behavior Breakdown
    document.getElementById('badge-behavior-dominant').textContent = `DOMINANT: ${data.primary_behavior}`;
    renderBehaviorBreakdown(data.behaviors || []);
    document.getElementById('forensic-insight-text').textContent = data.primary_reason;

    // 5. Render Interactive Telemetry Chart
    renderTelemetryChart(data.chart_data);

    // 6. Detailed Reconstruction Table
    document.getElementById('table-filename').textContent = data.filename;
    renderMetricsTable(metrics, data);
}

/**
 * Animate Safety Score SVG Ring
 */
function animateSafetyScore(score) {
    const scoreValEl = document.getElementById('safety-score-val');
    const scoreRatingEl = document.getElementById('safety-rating');
    const circle = document.getElementById('score-ring-circle');

    scoreValEl.textContent = score;

    // Circumference = 2 * PI * 34 ≈ 213.63
    const circumference = 213.63;
    const offset = circumference - (score / 100) * circumference;
    circle.style.strokeDashoffset = offset;

    if (score >= 75) {
        circle.style.stroke = '#10b981';
        scoreRatingEl.textContent = 'EXCELLENT';
        scoreRatingEl.style.color = '#10b981';
    } else if (score >= 50) {
        circle.style.stroke = '#3b82f6';
        scoreRatingEl.textContent = 'MODERATE';
        scoreRatingEl.style.color = '#3b82f6';
    } else if (score >= 30) {
        circle.style.stroke = '#f59e0b';
        scoreRatingEl.textContent = 'ELEVATED RISK';
        scoreRatingEl.style.color = '#f59e0b';
    } else {
        circle.style.stroke = '#ef4444';
        scoreRatingEl.textContent = 'CRITICAL RISK';
        scoreRatingEl.style.color = '#ef4444';
    }
}

/**
 * Render Cause Probabilities
 */
function renderCauseProbabilities(causes, primaryKey) {
    const container = document.getElementById('cause-probability-list');
    container.innerHTML = '';

    if (!causes || causes.length === 0) {
        container.innerHTML = '<div class="muted text-center">No cause data available.</div>';
        return;
    }

    causes.forEach(item => {
        const isPrimary = item.key === primaryKey;
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        
        probItem.innerHTML = `
            <div class="prob-meta-row">
                <span class="prob-name ${isPrimary ? 'text-white' : ''}">${isPrimary ? '⚡ ' : ''}${item.name}</span>
                <span class="prob-val">${item.percentage}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill ${isPrimary ? 'highlight' : ''}" style="width: ${Math.max(2, item.percentage)}%;"></div>
            </div>
        `;
        container.appendChild(probItem);
    });
}

/**
 * Render Behavior Breakdown Bar & Legend
 */
function renderBehaviorBreakdown(behaviors) {
    const bar = document.getElementById('behavior-bar-stacked');
    const legend = document.getElementById('behavior-legend');
    bar.innerHTML = '';
    legend.innerHTML = '';

    if (!behaviors || behaviors.length === 0) return;

    behaviors.forEach(b => {
        const color = BEHAVIOR_COLORS[b.key] || '#3b82f6';
        
        if (b.percentage > 0) {
            const seg = document.createElement('div');
            seg.className = 'behavior-segment';
            seg.style.width = `${b.percentage}%`;
            seg.style.backgroundColor = color;
            seg.title = `${b.name}: ${b.percentage}%`;
            bar.appendChild(seg);

            const legItem = document.createElement('div');
            legItem.className = 'legend-item';
            legItem.innerHTML = `
                <span class="legend-color" style="background-color: ${color};"></span>
                <span>${b.name} (${b.percentage}%)</span>
            `;
            legend.appendChild(legItem);
        }
    });
}

/**
 * Render Interactive Multi-Channel Telemetry Chart
 */
function renderTelemetryChart(chartData) {
    if (!chartData || !chartData.timestamps || chartData.timestamps.length === 0) return;

    const ctx = document.getElementById('telemetryChart').getContext('2d');

    if (telemetryChartInstance) {
        telemetryChartInstance.destroy();
    }

    const datasets = [
        {
            label: 'Vehicle Speed (km/h)',
            data: chartData.speed,
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37, 99, 235, 0.08)',
            borderWidth: 2.2,
            tension: 0.3,
            fill: true,
            yAxisID: 'ySpeed',
            channelKey: 'speed'
        },
        {
            label: 'Acceleration Magnitude (G)',
            data: chartData.acc_mag,
            borderColor: '#dc2626',
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.2,
            yAxisID: 'yAcc',
            channelKey: 'gforce'
        },
        {
            label: 'Yaw Rate (°/s)',
            data: chartData.yaw,
            borderColor: '#9333ea',
            backgroundColor: 'transparent',
            borderWidth: 1.8,
            borderDash: [4, 4],
            tension: 0.2,
            yAxisID: 'yYaw',
            channelKey: 'yaw'
        },
        {
            label: 'Jerk Rate (G/s)',
            data: chartData.jerk,
            borderColor: '#d97706',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            tension: 0.1,
            yAxisID: 'yAcc',
            channelKey: 'jerk'
        }
    ];

    telemetryChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.timestamps,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#475569',
                        font: { family: 'Inter', size: 12 },
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.92)',
                    titleColor: '#ffffff',
                    bodyColor: '#f1f5f9',
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1,
                    padding: 12,
                    boxPadding: 6,
                    usePointStyle: true,
                    titleFont: { family: 'Outfit', size: 13, weight: 'bold' },
                    bodyFont: { family: 'JetBrains Mono', size: 12 }
                }
            },
            scales: {
                x: {
                    grid: { color: '#e2e8f0' },
                    ticks: { color: '#64748b', font: { family: 'Inter', size: 11 }, maxTicksLimit: 12 }
                },
                ySpeed: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Speed (km/h)', color: '#2563eb' },
                    grid: { color: '#f1f5f9' },
                    ticks: { color: '#475569' }
                },
                yAcc: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'G-Force & Jerk (G)', color: '#dc2626' },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#475569' }
                },
                yYaw: {
                    type: 'linear',
                    display: false,
                    position: 'right',
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });

    applyChartFilter();
}

/**
 * Filter telemetry channels in chart
 */
function filterChartChannel(channel) {
    currentChartChannel = channel;
    document.querySelectorAll('.chart-filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-channel') === channel);
    });
    applyChartFilter();
}

function applyChartFilter() {
    if (!telemetryChartInstance) return;

    telemetryChartInstance.data.datasets.forEach(ds => {
        if (currentChartChannel === 'all' || ds.channelKey === currentChartChannel) {
            ds.hidden = false;
        } else {
            ds.hidden = true;
        }
    });

    telemetryChartInstance.update();
}

/**
 * Render Detailed Forensic Benchmark Table
 */
function renderMetricsTable(metrics, data) {
    const tbody = document.getElementById('metrics-table-body');
    tbody.innerHTML = '';

    const rows = [
        {
            prop: 'Peak Deceleration / G-Force Impact',
            val: `${(metrics.peak_g_force || 0).toFixed(2)}`,
            unit: 'G',
            benchmark: 'Critical Crash Threshold: > 4.0 G',
            eval: (metrics.peak_g_force || 0) >= 4.0 ? { text: 'CRITICAL CRASH', cls: 'critical' } : ((metrics.peak_g_force || 0) >= 1.8 ? { text: 'HIGH ANOMALY', cls: 'warning' } : { text: 'NORMAL', cls: 'normal' })
        },
        {
            prop: 'Delta-V Velocity Drop',
            val: `${(metrics.speed_drop || 0).toFixed(1)}`,
            unit: 'km/h',
            benchmark: 'Severe Deceleration: > 35 km/h',
            eval: (metrics.speed_drop || 0) >= 35 ? { text: 'SUDDEN COLLISION DROP', cls: 'critical' } : { text: 'PASS', cls: 'normal' }
        },
        {
            prop: 'Maximum Recorded Velocity',
            val: `${(metrics.max_speed || 0).toFixed(1)}`,
            unit: 'km/h',
            benchmark: 'Road Speed Limit Benchmark: 80 km/h',
            eval: (metrics.max_speed || 0) > 90 ? { text: 'OVERSPEEDING', cls: 'warning' } : { text: 'WITHIN RANGE', cls: 'normal' }
        },
        {
            prop: 'Peak Lateral Rotation Rate (Yaw)',
            val: `${(metrics.peak_yaw || 0).toFixed(2)}`,
            unit: '°/s',
            benchmark: 'Vehicle Rollover / Spin Risk: > 50 °/s',
            eval: (metrics.peak_yaw || 0) > 50 ? { text: 'SEVERE SKID / SWERVE', cls: 'critical' } : { text: 'STABLE', cls: 'normal' }
        },
        {
            prop: 'Rate of Acceleration Change (Jerk)',
            val: `${(metrics.peak_jerk || 0).toFixed(2)}`,
            unit: 'G/s',
            benchmark: 'Human Tolerance Comfort: < 15 G/s',
            eval: (metrics.peak_jerk || 0) > 25 ? { text: 'SEVERE VIOLENT JERK', cls: 'critical' } : { text: 'NORMAL', cls: 'normal' }
        },
        {
            prop: 'Primary Driver Behavioral State',
            val: `${data.primary_behavior}`,
            unit: 'Classification',
            benchmark: 'Expected Fleet Baseline: NORMAL',
            eval: data.primary_behavior === 'NORMAL' ? { text: 'COMPLIANT', cls: 'normal' } : { text: 'HIGH RISK BEHAVIOR', cls: 'warning' }
        }
    ];

    rows.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${r.prop}</strong></td>
            <td><code>${r.val}</code></td>
            <td><span class="muted">${r.unit}</span></td>
            <td>${r.benchmark}</td>
            <td><span class="eval-badge ${r.eval.cls}">${r.eval.text}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

/**
 * Load Session Incident History
 */
async function loadIncidentHistory() {
    try {
        const res = await fetch('/api/incidents');
        if (!res.ok) return;
        const data = await res.json();
        const tbody = document.getElementById('history-table-body');
        
        if (!data.incidents || data.incidents.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center muted">No previous incidents in current session.</td></tr>';
            return;
        }

        tbody.innerHTML = '';
        data.incidents.forEach(inc => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><code>${inc.analyzed_at || 'Just now'}</code></td>
                <td><strong>${inc.filename}</strong></td>
                <td><span class="badge-threshold ${inc.severity === 'CRITICAL' ? 'red' : (inc.severity === 'HIGH' ? 'amber' : 'green')}">${inc.primary_cause}</span></td>
                <td><code>${(inc.peak_g_force || 0).toFixed(2)} G</code></td>
                <td><strong>${inc.safety_score}/100</strong></td>
                <td>
                    ${inc.pdf_filename ? `<a href="/api/reports/${inc.pdf_filename}?download=1" class="history-btn-report" target="_blank">📥 PDF Report</a>` : '<span class="muted">N/A</span>'}
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Failed to load history:", e);
    }
}

/**
 * PDF Preview Modal
 */
function previewPdf() {
    if (!currentAnalysisData || !currentAnalysisData.pdf_report) {
        showToast('No PDF report generated yet.', 'error');
        return;
    }

    const modal = document.getElementById('pdf-modal');
    const iframe = document.getElementById('pdf-iframe');
    const title = document.getElementById('modal-pdf-title');

    title.textContent = `Forensic Report: ${currentAnalysisData.filename}`;
    iframe.src = currentAnalysisData.pdf_report;
    modal.classList.add('show');
}

function closePdfModal(e) {
    if (!e || e.target.id === 'pdf-modal' || e.target.classList.contains('modal-close-btn')) {
        document.getElementById('pdf-modal').classList.remove('show');
        document.getElementById('pdf-iframe').src = '';
    }
}

/**
 * Toast Notification Utility
 */
function showToast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';

    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';

    toast.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

/* ==========================================================================
   Supabase Cloud Evidence Storage Methods
   ========================================================================== */

let currentCloudConfig = {
    is_configured: false,
    supabase_url: '',
    bucket_name: 'evidentia-evidence',
    folder_prefix: 'accidents'
};

/**
 * Initialize Cloud Status on startup
 */
async function initCloudStatus() {
    try {
        const res = await fetch('/api/cloud/config');
        if (res.ok) {
            currentCloudConfig = await res.json();
            updateCloudUIIndicators();
            if (currentCloudConfig.is_configured) {
                fetchCloudFiles(true); // silent initial load
            }
        }
    } catch (e) {
        console.warn('Could not check Supabase cloud config:', e);
    }
}

function updateCloudUIIndicators() {
    const navIndicator = document.getElementById('cloud-nav-indicator');
    const badge = document.getElementById('cloud-file-count-badge');
    const bucketEl = document.getElementById('cloud-current-bucket');
    const folderEl = document.getElementById('cloud-current-folder');

    if (bucketEl) bucketEl.textContent = currentCloudConfig.bucket_name || 'evidentia-evidence';
    if (folderEl) folderEl.textContent = `/${currentCloudConfig.folder_prefix || 'accidents'}/`;

    if (navIndicator) {
        navIndicator.className = 'cloud-status-indicator ' + (currentCloudConfig.is_configured ? 'connected' : '');
    }
}

/**
 * Switch source between local file upload and Supabase cloud evidence
 */
function switchSourceTab(source) {
    const tabLocal = document.getElementById('tab-local');
    const tabCloud = document.getElementById('tab-cloud');
    const dropzone = document.getElementById('dropzone');
    const cloudPanel = document.getElementById('cloud-panel');

    if (source === 'local') {
        tabLocal.classList.add('active');
        tabCloud.classList.remove('active');
        dropzone.style.display = 'block';
        cloudPanel.style.display = 'none';
    } else {
        tabCloud.classList.add('active');
        tabLocal.classList.remove('active');
        dropzone.style.display = 'none';
        cloudPanel.style.display = 'flex';
        
        if (!currentCloudConfig.is_configured) {
            openCloudModal();
        } else {
            fetchCloudFiles();
        }
    }
}

/**
 * Fetch and list files in Supabase Storage bucket
 */
async function fetchCloudFiles(silent = false) {
    const tbody = document.getElementById('cloud-files-tbody');
    const meta = document.getElementById('cloud-status-meta');
    const badge = document.getElementById('cloud-file-count-badge');

    if (!tbody) return;

    if (!silent) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-center muted">Querying Supabase Storage bucket '${currentCloudConfig.bucket_name}'...</td></tr>`;
        if (meta) meta.textContent = 'Refreshing cloud telemetry files...';
    }

    try {
        const res = await fetch('/api/cloud/files');
        const data = await res.json();

        if (!res.ok || !data.success) {
            const errMsg = data.error || 'Failed to list cloud files';
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center" style="color: var(--status-red); padding: 24px;">
                        <div>⚠️ ${errMsg}</div>
                        <button class="btn btn-secondary btn-sm" style="margin-top: 12px;" onclick="openCloudModal()">Configure Supabase Credentials</button>
                    </td>
                </tr>`;
            if (meta) meta.textContent = 'Connection error. Check settings.';
            return;
        }

        const files = data.files || [];
        if (badge) badge.textContent = `${files.length} Files`;

        if (files.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center muted" style="padding: 24px;">
                        No accident telemetry files (.xlsx, .csv, .json) found in folder <code>/${data.folder}/</code>.<br>
                        Telemetry recorded by your collector will appear here automatically.
                    </td>
                </tr>`;
            if (meta) meta.textContent = `Connected to ${data.bucket} (0 files found)`;
            return;
        }

        if (meta) meta.textContent = `Found ${files.length} cloud accident telemetry file(s)`;

        let html = '';
        files.forEach((file) => {
            const timeStr = file.created_at ? new Date(file.created_at).toLocaleString() : 'Recent';
            html += `
                <tr>
                    <td>
                        <div class="cloud-file-name">
                            <span class="cloud-file-icon">📊</span>
                            <span>${file.name}</span>
                        </div>
                    </td>
                    <td><span class="muted" style="font-family: var(--font-mono); font-size: 0.8rem;">${file.remote_path}</span></td>
                    <td><strong>${file.size_formatted}</strong></td>
                    <td><span class="muted" style="font-size: 0.84rem;">${timeStr}</span></td>
                    <td>
                        <button class="btn-analyze-cloud" onclick="analyzeCloudFile('${file.remote_path}', '${file.name}')">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                            </svg>
                            <span>Analyze with AI</span>
                        </button>
                    </td>
                </tr>
            `;
        });
        tbody.innerHTML = html;

        if (!silent) showToast(`Loaded ${files.length} cloud evidence files.`, 'success');

    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-center" style="color: var(--status-red);">Error: ${e.message}</td></tr>`;
        if (meta) meta.textContent = 'Failed to load cloud files';
    }
}

/**
 * Download a file from Supabase and run AI deep analysis
 */
async function analyzeCloudFile(remotePath, fileName) {
    const overlay = document.getElementById('analyzing-overlay');
    const stageText = document.getElementById('loading-stage');
    
    // Switch visual to dropzone overlay to show loading
    const dropzone = document.getElementById('dropzone');
    dropzone.style.display = 'block';
    overlay.classList.add('active');

    stageText.textContent = `Downloading ${fileName} from Supabase Cloud Storage...`;
    const stageTimer1 = setTimeout(() => { stageText.textContent = "Parsing Telemetry Stream & IMU Channels..."; }, 900);
    const stageTimer2 = setTimeout(() => { stageText.textContent = "Running Behavioral Classifier & Impact Kinematics..."; }, 2000);
    const stageTimer3 = setTimeout(() => { stageText.textContent = "Generating Forensic Accident Report & Visualizations..."; }, 3200);

    try {
        const res = await fetch('/api/cloud/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: remotePath })
        });

        clearTimeout(stageTimer1);
        clearTimeout(stageTimer2);
        clearTimeout(stageTimer3);

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error || 'Cloud file analysis failed.');
        }

        const data = await res.json();
        renderAnalysisResults(data);
        showToast(`AI Analysis complete for ${fileName}!`, 'success');
        loadIncidentHistory();

    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        overlay.classList.remove('active');
        // Restore active tab view
        const tabCloud = document.getElementById('tab-cloud');
        if (tabCloud.classList.contains('active')) {
            dropzone.style.display = 'none';
        }
    }
}

/**
 * Cloud Settings Modal Management
 */
function openCloudModal() {
    const modal = document.getElementById('cloud-modal');
    const feedback = document.getElementById('cloud-cfg-feedback');
    feedback.style.display = 'none';

    document.getElementById('cfg-supabase-url').value = currentCloudConfig.supabase_url || '';
    document.getElementById('cfg-bucket-name').value = currentCloudConfig.bucket_name || 'evidentia-evidence';
    document.getElementById('cfg-folder-prefix').value = currentCloudConfig.folder_prefix || 'accidents';
    
    if (currentCloudConfig.is_configured) {
        document.getElementById('cfg-supabase-key').placeholder = `Configured (${currentCloudConfig.supabase_key_masked || 'Saved'})`;
    }

    modal.classList.add('show');
}

function closeCloudModal(e) {
    if (!e || e.target.id === 'cloud-modal' || e.target.classList.contains('modal-close-btn')) {
        document.getElementById('cloud-modal').classList.remove('show');
    }
}

async function testCloudConnection() {
    const feedback = document.getElementById('cloud-cfg-feedback');
    const btn = document.getElementById('btn-test-cloud');
    btn.textContent = 'Testing...';
    btn.disabled = true;

    try {
        const url = document.getElementById('cfg-supabase-url').value.trim();
        const key = document.getElementById('cfg-supabase-key').value.trim();
        const bucket = document.getElementById('cfg-bucket-name').value.trim() || 'evidentia-evidence';
        const folder = document.getElementById('cfg-folder-prefix').value.trim() || 'accidents';

        const res = await fetch('/api/cloud/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                supabase_url: url,
                supabase_key: key,
                bucket_name: bucket,
                folder_prefix: folder
            })
        });

        const contentType = res.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
            throw new Error(`Server returned non-JSON response (${res.status}). Please reload the page.`);
        }

        const data = await res.json();
        const test = data.connection_test || {};

        feedback.style.display = 'block';
        if (test.success) {
            feedback.className = 'modal-status-feedback success';
            feedback.innerHTML = `✅ <strong>Success:</strong> ${test.message}`;
            currentCloudConfig.is_configured = true;
            currentCloudConfig.supabase_url = url;
            currentCloudConfig.bucket_name = bucket;
            currentCloudConfig.folder_prefix = folder;
            updateCloudUIIndicators();
        } else {
            feedback.className = 'modal-status-feedback error';
            feedback.innerHTML = `❌ <strong>Connection Failed:</strong> ${test.error || 'Check credentials'}`;
        }
    } catch (e) {
        feedback.style.display = 'block';
        feedback.className = 'modal-status-feedback error';
        feedback.innerHTML = `❌ <strong>Error:</strong> ${e.message}`;
    } finally {
        btn.textContent = 'Test Connection';
        btn.disabled = false;
    }
}

async function saveCloudConfig() {
    const btn = document.getElementById('btn-save-cloud');
    btn.textContent = 'Saving...';
    btn.disabled = true;

    try {
        const url = document.getElementById('cfg-supabase-url').value.trim();
        const key = document.getElementById('cfg-supabase-key').value.trim();
        const bucket = document.getElementById('cfg-bucket-name').value.trim() || 'evidentia-evidence';
        const folder = document.getElementById('cfg-folder-prefix').value.trim() || 'accidents';

        const res = await fetch('/api/cloud/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                supabase_url: url,
                supabase_key: key,
                bucket_name: bucket,
                folder_prefix: folder
            })
        });

        const contentType = res.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
            throw new Error(`Server returned non-JSON response (${res.status}). Please reload the page.`);
        }

        const data = await res.json();
        const test = data.connection_test || {};

        if (test.success) {
            showToast('Supabase Cloud Evidence Connected!', 'success');
            currentCloudConfig.is_configured = true;
            currentCloudConfig.supabase_url = url;
            currentCloudConfig.bucket_name = bucket;
            currentCloudConfig.folder_prefix = folder;
            updateCloudUIIndicators();
            closeCloudModal(null);
            fetchCloudFiles();
        } else {
            const feedback = document.getElementById('cloud-cfg-feedback');
            feedback.style.display = 'block';
            feedback.className = 'modal-status-feedback error';
            feedback.innerHTML = `⚠️ Saved, but test failed: ${test.error || 'Invalid credentials'}`;
        }
    } catch (e) {
        showToast(`Save error: ${e.message}`, 'error');
    } finally {
        btn.textContent = 'Save & Sync Files';
        btn.disabled = false;
    }
}


