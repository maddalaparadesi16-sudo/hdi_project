// Tab Switching Logic
function switchTab(tabId) {
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    document.getElementById('tab-' + tabId).classList.add('active');
    document.getElementById('tab-btn-' + tabId).classList.add('active');
}

// Sync Number Inputs and Range Sliders
function syncInputs(baseId, val) {
    const numInput = document.getElementById(baseId);
    if (numInput && numInput.value !== val) numInput.value = val;
    
    const sliderInput = document.getElementById(baseId + '_slider');
    if (sliderInput && sliderInput.value !== val) sliderInput.value = val;
}

document.addEventListener('DOMContentLoaded', () => {
    const inputs = ['life_expectancy', 'expected_schooling', 'mean_schooling', 'gni_per_capita'];
    inputs.forEach(id => {
        const numInput = document.getElementById(id);
        if (numInput) {
            numInput.addEventListener('input', (e) => syncInputs(id, e.target.value));
        }
    });
});

// Handle the Form Prediction
async function handlePrediction(event) {
    event.preventDefault();

    const initialDiv = document.getElementById('result-initial');
    const loadingDiv = document.getElementById('result-loading');
    const successDiv = document.getElementById('result-success');

    // Show loading state
    initialDiv.classList.add('hidden');
    successDiv.classList.add('hidden');
    loadingDiv.classList.remove('hidden');

    const data = {
        life_expectancy: document.getElementById('life_expectancy').value,
        expected_schooling: document.getElementById('expected_schooling').value,
        mean_schooling: document.getElementById('mean_schooling').value,
        gni_per_capita: document.getElementById('gni_per_capita').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            alert(result.error || 'An error occurred during prediction.');
            loadingDiv.classList.add('hidden');
            initialDiv.classList.remove('hidden');
            return;
        }

        // Update Text Values
        document.getElementById('predicted-hdi-val').innerText = result.predicted_hdi.toFixed(3);
        document.getElementById('predicted-tier-val').innerText = result.tier;
        document.getElementById('tier-badge').className = 'tier-badge ' + result.color;

        // Animate Circle Progress
        const circle = document.getElementById('hdi-progress-bar');
        const circumference = 534.07;
        circle.style.strokeDashoffset = circumference - (result.predicted_hdi * circumference);

        // Assign Colors
        const colors = { 'emerald': '#10b981', 'teal': '#0d9488', 'amber': '#f59e0b', 'rose': '#e11d48' };
        const activeColor = colors[result.color] || '#10b981';
        circle.style.stroke = activeColor;

        // Update Dimension Progress Bars
        document.getElementById('health-index-val').innerText = result.dimensions.health_index.toFixed(3);
        document.getElementById('health-bar-fill').style.width = (result.dimensions.health_index * 100) + '%';
        document.getElementById('health-bar-fill').style.backgroundColor = activeColor;

        document.getElementById('education-index-val').innerText = result.dimensions.education_index.toFixed(3);
        document.getElementById('education-bar-fill').style.width = (result.dimensions.education_index * 100) + '%';
        document.getElementById('education-bar-fill').style.backgroundColor = activeColor;

        document.getElementById('income-index-val').innerText = result.dimensions.income_index.toFixed(3);
        document.getElementById('income-fill').style.width = (result.dimensions.income_index * 100) + '%';
        document.getElementById('income-fill').style.backgroundColor = activeColor;

        // Show Success Panel
        loadingDiv.classList.add('hidden');
        successDiv.classList.remove('hidden');

    } catch (error) {
        alert('Network error. Is the Flask server running?');
        loadingDiv.classList.add('hidden');
        initialDiv.classList.remove('hidden');
    }
}