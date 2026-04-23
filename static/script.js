document.addEventListener('DOMContentLoaded', () => {
    const complaintText = document.getElementById('complaint-text');
    const categoryInput = document.getElementById('category-input');
    const submitBtn = document.getElementById('submit-btn');
    const recentList = document.getElementById('recent-list');
    const historyList = document.getElementById('history-list');

    let debounceTimer;

    // Real-time prediction as user types (debounced)
    complaintText.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
            const text = complaintText.value.trim();
            if (text.length > 5) {
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text })
                    });
                    const data = await response.json();
                    if (data.category) {
                        categoryInput.value = data.category;
                    }
                } catch (error) {
                    console.error('Prediction error:', error);
                }
            } else {
                categoryInput.value = '';
            }
        }, 1000);
    });

    // Handle Submission
    submitBtn.addEventListener('click', async () => {
        const text = complaintText.value.trim();
        if (!text) {
            alert('Please describe your issue first.');
            return;
        }

        submitBtn.disabled = true;
        submitBtn.innerHTML = 'Submitting...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await response.json();
            
            if (data.error) {
                alert('Error: ' + data.error);
            } else {
                showSuccessScreen(data);
                refreshHistory();
            }
        } catch (error) {
            console.error('Submission error:', error);
            alert('Failed to submit complaint.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Submit Complaint <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M5 12H19" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 5L19 12L12 19" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
        }
    });

    function showSuccessScreen(data) {
        document.getElementById('res-id').textContent = 'TICKET ID: #' + data.id;
        document.getElementById('res-text').textContent = data.text;
        document.getElementById('res-category').textContent = data.category;
        document.getElementById('res-floor').textContent = data.floor + (data.floor !== 'N/A' ? 'nd Floor' : '');
        document.getElementById('res-room').textContent = data.room;
        document.getElementById('res-date').textContent = data.timestamp;
        
        const urgencyBadge = document.getElementById('res-urgency');
        urgencyBadge.textContent = data.urgency.toUpperCase() + ' URGENCY';
        urgencyBadge.className = 'badge badge-' + data.urgency.toLowerCase();

        showScreen('success-screen');
    }

    async function refreshHistory() {
        try {
            const response = await fetch('/get_history');
            const history = await response.json();
            
            // Update Recent Tracking (Home Screen)
            recentList.innerHTML = history.slice(0, 2).map(item => `
                <div class="tracking-item">
                    <div class="tracking-icon">${getIconForCategory(item.category)}
                        <div class="status-indicator">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" fill="white"/>
                            </svg>
                        </div>
                    </div>
                    <div class="tracking-info">
                        <h4>${item.text.length > 20 ? item.text.substring(0, 20) + '...' : item.text}</h4>
                        <p>Last updated: ${item.update_time}</p>
                    </div>
                    <div class="badge badge-in-progress">${item.status.toUpperCase()}</div>
                </div>
            `).join('');

            // Update History Screen
            historyList.innerHTML = history.map(item => `
                <div class="history-item">
                    <div class="item-tags">
                        <div class="badge badge-in-progress" style="background: var(--primary-light)">${item.category.toUpperCase()}</div>
                        <div class="badge badge-${getStatusClass(item.status)}">${item.status.toUpperCase()}</div>
                    </div>
                    <h3>${item.text}</h3>
                    <p>${item.id} - Room ${item.room} on Floor ${item.floor}</p>
                    <div class="history-footer">
                        <span>Updated ${item.update_time}</span>
                        <div class="arrow-right">&rarr;</div>
                    </div>
                </div>
            `).join('');

        } catch (error) {
            console.error('History fetch error:', error);
        }
    }

    function getIconForCategory(cat) {
        if (cat.includes('Electrical')) return '⚡';
        if (cat.includes('Plumbing')) return '🚰';
        if (cat.includes('Internet')) return '📶';
        if (cat.includes('Clean')) return '🧹';
        if (cat.includes('Food')) return '🍱';
        return '🛠️';
    }

    function getStatusClass(status) {
        if (status.includes('Resolved')) return 'resolved';
        if (status.includes('Pending')) return 'pending';
        return 'in-progress';
    }

    // Initial Load
    refreshHistory();
});

function showScreen(screenId, navEl) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');

    if (navEl) {
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        navEl.classList.add('active');
    }
}
