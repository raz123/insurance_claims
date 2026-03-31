document.addEventListener('DOMContentLoaded', () => {
    // --- 1. DOM ELEMENT SELECTIONS ---
    const pinOverlay = document.getElementById('pin-overlay');
    const appContent = document.getElementById('app-content');
    const dots = document.querySelectorAll('.dot');
    const numBtns = document.querySelectorAll('.num-btn');
    
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

    // --- 2. PIN LOCK LOGIC (4321) ---
    const CORRECT_PIN = '4321';
    let enteredPin = '';

    // Check if previously logged in (session only)
    if(sessionStorage.getItem('authenticated') === 'true') {
        unlockApp();
    } else {
        pinOverlay.classList.add('active');
        appContent.classList.add('hidden');
    }

    numBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.classList.contains('clear-btn')) {
                enteredPin = '';
                updateDots();
            } else if (btn.classList.contains('del-btn')) {
                enteredPin = enteredPin.slice(0, -1);
                updateDots();
            } else if (enteredPin.length < 4) {
                enteredPin += btn.dataset.val;
                updateDots();
                if (enteredPin.length === 4) {
                    setTimeout(verifyPin, 100);
                }
            }
        });
    });

    function updateDots() {
        dots.forEach((dot, index) => {
            dot.classList.toggle('filled', index < enteredPin.length);
        });
    }

    function verifyPin() {
        if (enteredPin === CORRECT_PIN) {
            sessionStorage.setItem('authenticated', 'true');
            unlockApp();
        } else {
            // Shake visual feedback
            const display = document.querySelector('.pin-display');
            display.style.transform = 'translateX(-10px)';
            setTimeout(() => display.style.transform = 'translateX(10px)', 100);
            setTimeout(() => display.style.transform = 'translateX(-10px)', 200);
            setTimeout(() => display.style.transform = 'translateX(0)', 300);
            
            enteredPin = '';
            setTimeout(updateDots, 300);
        }
    }

    function unlockApp() {
        pinOverlay.classList.remove('active');
        appContent.classList.remove('hidden');
        loadSettings();
    }

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

        if (engine === 'gemini') {
            await runCloudOcr('GEMINI_OCR');
        } else if (engine === 'drive') {
            await runCloudOcr('DRIVE_OCR');
        } else {
            runWebWorkerAI(engine);
        }
    });

    function runWebWorkerAI(engine) {
        if (!ocrWorker) {
             ocrWorker = new Worker('worker.js', { type: 'module' });
        }

        loadingText.innerText = "Initializing Local AI Inference...";

        ocrWorker.postMessage({
            action: 'PROCESS_IMAGE',
            imageBase64: currentImageBase64,
            engine: engine
        });

        ocrWorker.onmessage = (e) => {
            const { status, message, data, error } = e.data;

            if (status === 'progress' || status === 'info') {
                loadingText.innerText = message;
            } else if (status === 'success') {
                loadingIndicator.classList.add('hidden');
                
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

    async function runCloudOcr(actionType) {
        const scriptUrl = localStorage.getItem('scriptUrl');
        if (!scriptUrl) {
            alert('Please configure your Google Apps Script URL in settings first.');
            loadingIndicator.classList.add('hidden');
            settingsModal.classList.add('active');
            return;
        }

        const actionName = actionType === 'GEMINI_OCR' ? 'Gemini 1.5 Flash' : 'Google Drive';
        loadingText.innerText = `Calling ${actionName} for Intelligent OCR...`;

        try {
            const response = await fetch(scriptUrl, {
                method: "POST",
                body: JSON.stringify({
                    action: actionType,
                    imageBase64: currentImageBase64.split(",")[1],
                    mimeType: "image/jpeg",
                    filename: "receipt_temp.jpg"
                })
            });

            const result = await response.json();
            
            if (result.success) {
                 loadingIndicator.classList.add('hidden');
                 
                 // If the cloud service found JSON (like Gemini)
                 if (result.data) {
                    if(result.data.vendor) document.getElementById('vendor-input').value = result.data.vendor;
                    if(result.data.amount) document.getElementById('amount-input').value = result.data.amount;
                    if(result.data.date) document.getElementById('date-input').value = result.data.date;
                 } else {
                    // Fallback to text parsing if only raw text is returned (Drive)
                    let rawText = result.extractedText || "";
                    const priceMatch = rawText.match(/\$?\s*(\d{1,4}[.,]\d{2})/);
                    if(priceMatch) document.getElementById('amount-input').value = priceMatch[1].replace(',', '.');
                    document.getElementById('vendor-input').value = "Parsed from Cloud";
                 }
                 
                 submitBtn.disabled = false;
                 processBtn.innerText = "Re-Scan Document";
                 processBtn.classList.remove('hidden');
            } else {
                 throw new Error(result.error);
            }

        } catch (e) {
            loadingIndicator.classList.add('hidden');
            processBtn.classList.remove('hidden');
            alert('Server Error: ' + e.message);
        }
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
