// PulseGuard AI — Clean Client-Side Logic

let waveformChartInstance = null;
let selectedVideoFile = null;
let scanInterval = null;

document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();
    setupDropzone();
});

// Setup File Drag & Drop
function setupDropzone() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("video-file-input");
    const defaultView = document.getElementById("dropzone-default");
    const previewContainer = document.getElementById("dropzone-preview-container");
    const changeFileBtn = document.getElementById("change-file-btn");
    const startScanBtn = document.getElementById("start-scan-btn");

    // Clicking dropzone in default state opens file picker
    dropzone.addEventListener("click", (e) => {
        if (!previewContainer.classList.contains("hidden")) {
            return;
        }
        fileInput.click();
    });

    // Clicking Change Video button
    changeFileBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.value = "";
        fileInput.click();
    });

    // Manual re-scan button
    if (startScanBtn) {
        startScanBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            submitUpload();
        });
    }

    // Drag & Drop handlers
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dropzone-active");
    });

    dropzone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dropzone-active");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dropzone-active");
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    // File input selection handler
    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });
}

function resetUploadUI() {
    const defaultView = document.getElementById("dropzone-default");
    const previewContainer = document.getElementById("dropzone-preview-container");
    const fileInput = document.getElementById("video-file-input");

    defaultView.classList.remove("hidden");
    previewContainer.classList.add("hidden");
    if (fileInput) fileInput.value = "";
    selectedVideoFile = null;
    lucide.createIcons();
}

function handleFileSelection(file) {
    if (!file) return;

    selectedVideoFile = file;
    const defaultView = document.getElementById("dropzone-default");
    const previewContainer = document.getElementById("dropzone-preview-container");
    const nameText = document.getElementById("file-preview-name");
    const sizeText = document.getElementById("file-preview-size");
    const statusSubtext = document.getElementById("upload-status-subtext");

    nameText.textContent = file.name;
    sizeText.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB • Automatic Analysis`;
    if (statusSubtext) {
        statusSubtext.textContent = "Uploading & extracting vascular pulse traces...";
    }

    defaultView.classList.add("hidden");
    previewContainer.classList.remove("hidden");
    lucide.createIcons();

    // Trigger instant auto-scan
    submitUpload();
}

function showLoading(show) {
    const spinner = document.getElementById("loading-spinner");
    const results = document.getElementById("results-container");

    if (show) {
        spinner.classList.remove("hidden");
        results.classList.add("hidden");

        const steps = [
            "Tracking 478 MediaPipe facial landmarks...",
            "Isolating Forehead & Cheek vascular capillary patches...",
            "Computing Blood Volume Pulse (BVP) via POS...",
            "Bandpass filtering cardiac band (0.7 - 3.0 Hz)...",
            "Measuring cross-spectral spatial coherence & voting..."
        ];
        let idx = 0;
        const stepText = document.getElementById("scan-step-text");
        clearInterval(scanInterval);
        scanInterval = setInterval(() => {
            idx = (idx + 1) % steps.length;
            stepText.textContent = steps[idx];
        }, 1800);

        spinner.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else {
        clearInterval(scanInterval);
        spinner.classList.add("hidden");
    }
}

// 1-Click Sample Test
async function runSample(sampleType) {
    showLoading(true);

    try {
        const response = await fetch(`/api/analyze_sample/${sampleType}?method=pos`, {
            method: "POST"
        });
        const data = await response.json();
        if (!response.ok || data.error) {
            alert(`Error: ${data.error || 'Server error'}`);
            showLoading(false);
            return;
        }
        displayResults(data);
    } catch (err) {
        alert(`Analysis failed: ${err.message}`);
        showLoading(false);
    }
}

// Submit Uploaded File
async function submitUpload() {
    if (!selectedVideoFile) return;

    showLoading(true);
    const formData = new FormData();
    formData.append("video", selectedVideoFile);
    formData.append("method", "pos");

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        if (!response.ok || data.error) {
            alert(`Analysis Error: ${data.error || ('HTTP Error ' + response.status)}`);
            showLoading(false);
            resetUploadUI();
            return;
        }
        displayResults(data);
    } catch (err) {
        alert(`Failed to upload and analyze: ${err.message}`);
        showLoading(false);
        resetUploadUI();
    }
}

// Render Results Dashboard
function displayResults(data) {
    showLoading(false);
    const container = document.getElementById("results-container");
    container.classList.remove("hidden");

    // Video Player
    const player = document.getElementById("result-video-player");
    if (data.video_url) {
        player.src = data.video_url;
        player.load();
    }
    const meta = document.getElementById("video-meta-badge");
    meta.textContent = `${data.video_info.resolution} • ${data.video_info.fps} FPS • ${data.video_info.duration_sec}s`;

    // Short clip badge
    const shortBadge = document.getElementById("clip-length-badge");
    if (data.video_info.duration_sec < 5.0) {
        shortBadge.classList.remove("hidden");
    } else {
        shortBadge.classList.add("hidden");
    }

    // Verdict Banner
    const banner = document.getElementById("verdict-banner");
    const vTitle = document.getElementById("verdict-title");
    const vConf = document.getElementById("verdict-confidence");
    const vDesc = document.getElementById("verdict-desc");
    const vIcon = document.getElementById("verdict-icon");

    banner.className = "relative overflow-hidden rounded-3xl p-6 sm:p-8 border transition-all duration-500 shadow-2xl";

    const isReal = (data.verdict === "REAL");
    if (isReal) {
        banner.classList.add("verdict-real");
        vTitle.textContent = "AUTHENTIC HUMAN";
        vDesc.textContent = "Microvascular blood volume pulses are synchronized across facial capillary beds with natural cardiac frequencies.";
        vIcon.setAttribute("data-lucide", "shield-check");
    } else {
        banner.classList.add("verdict-fake");
        vTitle.textContent = "DETECTED DEEPFAKE";
        vDesc.textContent = "Facial skin patches lack physiological hemodynamic synchrony. Significant inter-patch frequency divergence and generative flicker detected.";
        vIcon.setAttribute("data-lucide", "shield-alert");
    }
    vConf.textContent = `${data.confidence.toFixed(1)}%`;

    // Stats Cards
    document.getElementById("stat-bpm").textContent = data.heart_rate.bpm.toFixed(1);
    document.getElementById("stat-freq").textContent = `${data.heart_rate.peak_freq_hz.toFixed(2)} Hz Dominant Peak`;

    document.getElementById("stat-coherence").textContent = data.spatial_coherence.mean_coherence.toFixed(3);
    const cohTag = document.getElementById("stat-coherence-tag");
    if (data.spatial_coherence.mean_coherence >= 0.50) {
        cohTag.textContent = "Synchronized";
        cohTag.className = "text-[10px] font-bold text-emerald-400 uppercase tracking-wider block";
    } else {
        cohTag.textContent = "Desynchronized";
        cohTag.className = "text-[10px] font-bold text-rose-400 uppercase tracking-wider block";
    }

    document.getElementById("stat-cross-corr").textContent = data.spatial_coherence.cross_correlation.toFixed(3);
    document.getElementById("stat-lag").textContent = `Phase Delay: ${data.spatial_coherence.mean_lag_samples.toFixed(1)} frames`;

    document.getElementById("stat-dispersion").textContent = data.spatial_coherence.bpm_dispersion.toFixed(1);
    document.getElementById("stat-snr").textContent = `Peak SNR: ${data.heart_rate.snr_db.toFixed(1)} dB`;

    // Diagnostic Image
    document.getElementById("diagnostic-plot-img").src = data.diagnostic_plot_url;
    document.getElementById("diagnostic-plot-link").href = data.diagnostic_plot_url;

    // Render Waveform Chart
    renderWaveformChart(data.charts.time_axis, data.charts.waveforms);

    // Render Coherence Matrix Grid
    renderCoherenceMatrix(data.charts.coherence_matrix, data.charts.roi_names);

    lucide.createIcons();

    // Smooth scroll down to results
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderWaveformChart(labels, waveforms) {
    const ctx = document.getElementById("waveform-chart").getContext("2d");
    if (waveformChartInstance) waveformChartInstance.destroy();

    waveformChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Forehead Capillaries",
                    data: waveforms.forehead,
                    borderColor: "#34d399",
                    backgroundColor: "rgba(52, 211, 153, 0.05)",
                    borderWidth: 2,
                    tension: 0.35,
                    pointRadius: 0
                },
                {
                    label: "Left Cheek",
                    data: waveforms.left_cheek,
                    borderColor: "#38bdf8",
                    backgroundColor: "rgba(56, 189, 248, 0.05)",
                    borderWidth: 2,
                    tension: 0.35,
                    pointRadius: 0
                },
                {
                    label: "Right Cheek",
                    data: waveforms.right_cheek,
                    borderColor: "#fbbf24",
                    backgroundColor: "rgba(251, 191, 36, 0.05)",
                    borderWidth: 2,
                    tension: 0.35,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            scales: {
                x: {
                    grid: { color: "rgba(30, 41, 59, 0.6)" },
                    ticks: { color: "#64748b", maxTicksLimit: 10, font: { family: '"JetBrains Mono"', size: 10 } }
                },
                y: {
                    grid: { color: "rgba(30, 41, 59, 0.6)" },
                    ticks: { color: "#64748b", font: { family: '"JetBrains Mono"', size: 10 } }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: "#94a3b8", boxWidth: 10, font: { family: '"Plus Jakarta Sans"', size: 11, weight: '600' } }
                }
            }
        }
    });
}

function renderCoherenceMatrix(matrix, roiNames) {
    const grid = document.getElementById("coherence-matrix-grid");
    grid.innerHTML = "";

    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
            const val = matrix[i][j];
            const cell = document.createElement("div");
            cell.className = "p-3.5 rounded-2xl border text-xs font-bold flex flex-col items-center justify-center transition-all shadow-sm";

            if (val >= 0.70) {
                cell.className += " bg-emerald-500/10 border-emerald-500/30 text-emerald-400";
            } else if (val >= 0.50) {
                cell.className += " bg-teal-500/10 border-teal-500/30 text-teal-400";
            } else if (val >= 0.35) {
                cell.className += " bg-amber-500/10 border-amber-500/30 text-amber-400";
            } else {
                cell.className += " bg-rose-500/10 border-rose-500/30 text-rose-400";
            }

            cell.innerHTML = `
                <span class="text-[10px] font-sans font-medium opacity-60">${roiNames[i]} • ${roiNames[j]}</span>
                <span class="text-sm font-mono mt-1 font-bold">${val.toFixed(2)}</span>
            `;
            grid.appendChild(cell);
        }
    }
}
