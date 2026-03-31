document.addEventListener('DOMContentLoaded', () => {
    console.log('App Version: 1.3.2');
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    const engineSelect = document.getElementById('engine-select');
    const scriptUrlInput = document.getElementById('apps-script-url');
    
    const imagePreviewArea = document.getElementById('image-preview');
    const cameraInput = document.getElementById('camera-input');
    const receiptImg = document.getElementById('receipt-img');
    const placeholder = document.getElementById('preview-placeholder');
    const processBtn = document.getElementById('process-ocr-btn');
    
    const loadingIndicator = document.getElementById('loading-indicator');
    const loadingText = document.getElementById('loading-text');
    const submitBtn = document.getElementById('submit-claim-btn');
    const claimForm = document.getElementById('claim-form');

    // --- 2. INITIALIZATION ---
    loadSettings();

    // --- 3. SETTINGS LOGIC ---
    settingsBtn.addEventListener('click', () => {
        settingsModal.classList.add('active');
    });

    // Close modal if clicking outside
    settingsModal.addEventListener('click', (e) => {
        if(e.target === settingsModal) {
            settingsModal.classList.remove('active');
        }
    });

    saveSettingsBtn.addEventListener('click', () => {
        localStorage.setItem('ocrEngine', engineSelect.value);
        localStorage.setItem('scriptUrl', scriptUrlInput.value);
        settingsModal.classList.remove('active');
    });

    // Add Clear Cache Button Dynamically
    const clearCacheBtn = document.createElement('button');
    clearCacheBtn.innerText = "Clear Local AI Cache";
    clearCacheBtn.className = "btn-secondary";
    clearCacheBtn.style.marginTop = "15px";
    clearCacheBtn.style.width = "100%";
    clearCacheBtn.onclick = async () => {
        const { clearCache } = await import('./db-storage.js');
        await clearCache();
        alert("Local AI cache cleared. Next run will re-download models.");
    };
    settingsModal.querySelector('.modal-content').appendChild(clearCacheBtn);

    function loadSettings() {
        const savedEngine = localStorage.getItem('ocrEngine') || 'glm';
        const savedUrl = localStorage.getItem('scriptUrl') || '';
        engineSelect.value = savedEngine;
        scriptUrlInput.value = savedUrl;
        
        // Auto set Date to today
        document.getElementById('date-input').value = new Date().toISOString().split('T')[0];
    }

    // Handle Local Model Picker
    const localModelPicker = document.getElementById('local-model-picker');
    localModelPicker.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (!files.length) return;

        const buffers = {};
        for (const file of files) {
            // Match filenames exactly as worker expects them
            const name = file.name;
            const buffer = await file.arrayBuffer();
            buffers[name] = buffer;
        }

        if (!ocrWorker) {
            runWebWorkerAI('glm'); // Instantiate if needed
        }

        ocrWorker.postMessage({ 
            action: 'LOAD_MODELS', 
            files: buffers 
        }, Object.values(buffers)); // Transferable buffers
        
        alert("Models being stored locally. You can now use the app offline.");
    });
    let currentImageBase64 = null;

    imagePreviewArea.addEventListener('click', () => {
        cameraInput.click();
    });

    cameraInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Display raw file immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            receiptImg.src = e.target.result;
            receiptImg.style.display = 'block';
            placeholder.style.display = 'none';
            processBtn.classList.remove('hidden');
            
            // Compress Image
            compressImage(e.target.result);
        };
        reader.readAsDataURL(file);
    });

    function compressImage(dataUrl) {
        const img = new Image();
        img.src = dataUrl;
        img.onload = () => {
            const canvas = document.createElement('canvas');
            // Scale down to max 1200px width/height to save bandwidth
            const MAX_SIZE = 1200;
            let width = img.width;
            let height = img.height;

            if (width > height) {
                if (width > MAX_SIZE) {
                    height *= MAX_SIZE / width;
                    width = MAX_SIZE;
                }
            } else {
                if (height > MAX_SIZE) {
                    width *= MAX_SIZE / height;
                    height = MAX_SIZE;
                }
            }
            canvas.width = width;
            canvas.height = height;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);
            
            // Convert back to base64, JPEG 70% quality
            currentImageBase64 = canvas.toDataURL('image/jpeg', 0.7);
        };
    }

    // --- 5. OCR PROCESSING LOGIC ---
    let ocrWorker = null;

    processBtn.addEventListener('click', async () => {
        if (!currentImageBase64) return;

        const engine = localStorage.getItem('ocrEngine') || 'glm';
        
        processBtn.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');

        runWebWorkerAI(engine);
    });

    function runWebWorkerAI(engine) {
        if (!ocrWorker) {
             console.log('[App] Instantiating custom GLM worker...');
             // Cache busting version 1.3.2
             ocrWorker = new Worker('worker.js?v=v1.3.2', { type: 'module' });
             
             ocrWorker.onmessage = (e) => {
                 const { status, message, percent, file, text, data, error } = e.data;
                 
                 if (status === 'progress') {
                     updateProgressBar(file || 'Init', percent, message);
                 } else if (status === 'info') {
                     loadingText.innerText = message;
                 } else if (status === 'stream') {
                     showLiveFeed(text);
                 } else if (status === 'success') {
                     loadingIndicator.classList.add('hidden');
                     fillForm(data);
                 } else if (status === 'error') {
                     alert("OCR Error: " + error);
                     loadingIndicator.classList.add('hidden');
                     processBtn.classList.remove('hidden');
                 }
             };
        }

        loadingIndicator.classList.remove('hidden');
        document.getElementById('extraction-feed-container').classList.remove('hidden');
        document.getElementById('extraction-feed').innerText = "AI is thinking...";

        ocrWorker.postMessage({
            action: 'PROCESS_IMAGE',
            imageBase64: currentImageBase64
        });
    }

    const progressStates = {};
    function updateProgressBar(file, percent, message) {
        const container = document.getElementById('progress-bars-container');
        if (message) loadingText.innerText = message;

        if (!progressStates[file]) {
            const wrapper = document.createElement('div');
            wrapper.className = 'progress-item';
            wrapper.style.marginBottom = '8px';
            wrapper.innerHTML = `
                <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                    <span>${file}</span>
                    <span class="pct-label">0%</span>
                </div>
                <div class="progress-wrapper">
                    <div class="progress-bar" style="width:0%"></div>
                </div>
            `;
            container.appendChild(wrapper);
            progressStates[file] = {
                bar: wrapper.querySelector('.progress-bar'),
                label: wrapper.querySelector('.pct-label')
            };
        }
        progressStates[file].bar.style.width = `${percent}%`;
        progressStates[file].label.innerText = `${percent}%`;
    }

    function showLiveFeed(text) {
        const feed = document.getElementById('extraction-feed');
        feed.innerText = text;
        feed.scrollTop = feed.scrollHeight;
    }

    function fillForm(data) {
        if(data.vendor) document.getElementById('vendor-input').value = data.vendor;
        if(data.amount) document.getElementById('amount-input').value = data.amount;
        if(data.date) document.getElementById('date-input').value = data.date;
        submitBtn.disabled = false;
        processBtn.classList.remove('hidden');
        processBtn.innerText = "Re-Scan Document";
    }



    // --- FORM SUBMISSION LOGIC ---
    claimForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const scriptUrl = localStorage.getItem('scriptUrl');
        if (!scriptUrl) {
            alert('Please configure your Google Apps Script URL in settings before submitting.');
            settingsModal.classList.add('active');
            return;
        }

        submitBtn.disabled = true;
        submitBtn.innerText = 'Saving to Sheet...';

        const payload = {
            action: "DATABASE_SAVE",
            imageBase64: currentImageBase64.split(",")[1],
            formData: {
                spender: document.getElementById('spender').value,
                vendor: document.getElementById('vendor-input').value,
                date: document.getElementById('date-input').value,
                amount: document.getElementById('amount-input').value
            }
        };

        try {
            const response = await fetch(scriptUrl, {
                method: "POST",
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            
            if (result.success) {
                alert('Claim Submitted Sucessfully to Google Sheets!');
                // Reset UI
                claimForm.reset();
                document.getElementById('date-input').value = new Date().toISOString().split('T')[0];
                receiptImg.style.display = 'none';
                placeholder.style.display = 'block';
                currentImageBase64 = null;
                submitBtn.disabled = true;
                processBtn.classList.add('hidden');
            } else {
                throw new Error(result.error);
            }
        } catch (e) {
            alert('Failed to save to sheet: ' + e.message);
        } finally {
            submitBtn.innerText = 'Submit Claim';
        }
    });

});
