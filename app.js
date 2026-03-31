document.addEventListener('DOMContentLoaded', () => {
    console.log('App Version: 1.2.0');
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

    function loadSettings() {
        const savedEngine = localStorage.getItem('ocrEngine') || 'glm';
        const savedUrl = localStorage.getItem('scriptUrl') || '';
        engineSelect.value = savedEngine;
        scriptUrlInput.value = savedUrl;
        
        // Auto set Date to today
        document.getElementById('date-input').value = new Date().toISOString().split('T')[0];
    }

    // --- 4. CAMERA & COMPRESSION LOGIC ---
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
             ocrWorker = new Worker('worker.js?v=v1.2.0', { type: 'module' });
             
             ocrWorker.onerror = (err) => {
                console.error('[App] Worker failed to load or experienced a top-level error:', err);
             };
        }

        loadingText.innerText = "Initializing Local AI Inference...";

        ocrWorker.postMessage({
            action: 'PROCESS_IMAGE',
            imageBase64: currentImageBase64,
            engine: engine
        });

        const progressBar = document.getElementById('progress-bar');

        ocrWorker.onmessage = (e) => {
            const { status, message, data, error, percent } = e.data;

            if (status === 'progress' || status === 'info') {
                if (message) loadingText.innerText = message;
                if (percent !== undefined) {
                    progressBar.style.width = `${percent}%`;
                }
            } else if (status === 'success') {
                loadingIndicator.classList.add('hidden');
                progressBar.style.width = '0%';
                
                // Auto fill fields
                if(data.vendor) document.getElementById('vendor-input').value = data.vendor;
                if(data.amount > 0) document.getElementById('amount-input').value = data.amount;
                if(data.date) document.getElementById('date-input').value = data.date;
                
                submitBtn.disabled = false;
                processBtn.innerText = "Re-Scan Document";
                processBtn.classList.remove('hidden');
            } else if (status === 'error') {
                loadingIndicator.classList.add('hidden');
                processBtn.classList.remove('hidden');
                alert('OCR Error: ' + error);
            }
        };
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
