// Sync inputs (Number and Range Slider)
function syncInputs(id, value) {
    const inputNum = document.getElementById(id);
    const inputSlider = document.getElementById(`${id}_slider`);
    
    if (inputNum && inputSlider) {
        inputNum.value = value;
        inputSlider.value = value;
    }
}

// Ensure the sliders also sync if the user types directly into the number input
document.addEventListener("DOMContentLoaded", () => {
    const inputs = ["life_expectancy", "expected_schooling", "mean_schooling", "gni_per_capita"];
    
    inputs.forEach(id => {
        const inputNum = document.getElementById(id);
        if (inputNum) {
            inputNum.addEventListener("input", (e) => {
                const value = e.target.value;
                const inputSlider = document.getElementById(`${id}_slider`);
                if (inputSlider) {
                    inputSlider.value = value;
                }
            });
        }
    });
});

// Switch Dashboard Tabs
function switchTab(tabName) {
    // Get all tab panels and buttons
    const panels = document.querySelectorAll(".tab-panel");
    const buttons = document.querySelectorAll(".nav-btn");
    
    // Hide all panels
    panels.forEach(panel => {
        panel.classList.remove("active");
    });
    
    // Deactivate all buttons
    buttons.forEach(btn => {
        btn.classList.remove("active");
    });
    
    // Show active panel and button
    const targetPanel = document.getElementById(`tab-${tabName}`);
    const targetBtn = document.getElementById(`tab-btn-${tabName}`);
    
    if (targetPanel && targetBtn) {
        targetPanel.classList.add("active");
        targetBtn.classList.add("active");
    }
}

// Handle Prediction Form Submission
async function handlePrediction(event) {
    event.preventDefault();
    
    const form = document.getElementById("hdi-predict-form");
    const formData = new FormData(form);
    
    // Get result panels
    const initialPanel = document.getElementById("result-initial");
    const loadingPanel = document.getElementById("result-loading");
    const successPanel = document.getElementById("result-display-area");
    
    // Toggle loading states
    if (initialPanel) initialPanel.classList.add("hidden");
    if (successPanel) successPanel.classList.add("hidden");
    if (loadingPanel) loadingPanel.classList.remove("hidden");
    
    // Form data conversion
    const payload = {};
    formData.forEach((value, key) => {
        payload[key] = parseFloat(value);
    });
    
    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || "Failed to retrieve prediction.");
        }
        
        // Hide loading panel
        if (loadingPanel) loadingPanel.classList.add("hidden");
        if (successPanel) successPanel.classList.remove("hidden");
        
        // Update score value
        const scoreValEl = document.getElementById("predicted-hdi-val");
        const score = result.predicted_hdi;
        scoreValEl.innerText = score.toFixed(3);
        
        // Update SVG Progress Circle
        const circle = document.getElementById("hdi-progress-bar");
        const circumference = 2 * Math.PI * 85; // r = 85 -> 534.07
        const offset = circumference - (score * circumference);
        circle.style.strokeDashoffset = offset;
        
        // Update Badge and colors based on tier
        const badge = document.getElementById("tier-badge");
        const tierVal = document.getElementById("predicted-tier-val");
        
        badge.className = `tier-badge ${result.color}`;
        tierVal.innerText = result.tier;
        
        // Update progress bar color theme
        circle.style.stroke = `var(--color-${result.color})`;
        
        // Update Dimension progress bars
        updateDimensionBar("health-bar-fill", "health-index-val", result.dimensions.health_index, result.color);
        updateDimensionBar("education-bar-fill", "education-index-val", result.dimensions.education_index, result.color);
        updateDimensionBar("income-fill", "income-index-val", result.dimensions.income_index, result.color);
        
        // Generate summary text
        const summaryText = document.getElementById("result-text-summary");
        let desc = "";
        
        if (result.color === "emerald") {
            desc = `Excellent performance! With an estimated score of <strong>${score.toFixed(3)}</strong>, this country ranks in the <strong>Very High Human Development</strong> tier. Excellent indicators in longevity, education systems, and income PPP levels represent a highly advanced nation.`;
        } else if (result.color === "teal") {
            desc = `Strong human development. The predicted score of <strong>${score.toFixed(3)}</strong> places the country in the <strong>High Human Development</strong> tier. Continued focus on healthcare infrastructure and expanding tertiary education can bridge the remaining gap to the top tier.`;
        } else if (result.color === "amber") {
            desc = `Moderate development. The predicted score of <strong>${score.toFixed(3)}</strong> places the country in the <strong>Medium Human Development</strong> tier. Significant opportunities exist for growth, particularly through targeted investments in healthcare access, vocational training, and economic diversification.`;
        } else {
            desc = `Critical development needs. The predicted score of <strong>${score.toFixed(3)}</strong> indicates <strong>Low Human Development</strong>. Strategic international aid, investments in basic educational infrastructure, vaccine campaigns, and poverty alleviation programs are highly recommended.`;
        }
        summaryText.innerHTML = desc;
        
    } catch (error) {
        if (loadingPanel) loadingPanel.classList.add("hidden");
        if (initialPanel) initialPanel.classList.remove("hidden");
        alert(`Prediction Error: ${error.message}`);
    }
}

// Function to update progress bars
function updateDimensionBar(barId, valId, score, colorClass) {
    const bar = document.getElementById(barId);
    const label = document.getElementById(valId);
    
    if (bar && label) {
        // Update numerical label
        label.innerText = score.toFixed(2);
        
        // Remove existing class fills
        bar.className = "progress-fill";
        // Add new class fill
        bar.classList.add(`${colorClass}-fill`);
        
        // Set width percentage (score is from 0.00 to 1.00)
        setTimeout(() => {
            bar.style.width = `${score * 100}%`;
        }, 100);
    }
}
