let worker;

// Listen for messages from the main thread
self.onmessage = async function(event) {
    const { action, imageBase64, engine } = event.data;

    if (action === 'PROCESS_IMAGE') {
        try {
            self.postMessage({ status: 'info', message: `Initializing ${engine}...` });

            if (engine === 'florence' || engine === 'trocr') {
                await processTransformersJs(imageBase64, engine);
            } else if (engine === 'tesseract') {
                await processTesseractJs(imageBase64);
            } else {
                self.postMessage({ status: 'error', error: 'Unknown engine requested in worker.' });
            }

        } catch (error) {
            self.postMessage({ status: 'error', error: error.toString() });
        }
    }
};

async function processTransformersJs(base64Image, engineName) {
    self.postMessage({ status: 'info', message: 'Importing Transformers.js v3 (WebGPU)...' });
    
    // Using Transformers.js v3 (latest alpha/beta supports Florence-2 and GLM architecture better)
    importScripts('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19');

    const { pipeline, env } = self.Transformers;
    
    // WebGPU optimization
    env.allowLocalModels = false;

    let resultData = { vendor: "", amount: "", date: "" };

    if (engineName === 'glm') {
        self.postMessage({ status: 'progress', message: 'Loading GLM-OCR (0.9B) via WebGPU. This may take a moment...' });
        
        try {
            // we use the specialized GLM-OCR pipeline if supported, or image-to-text
            const modelId = 'brad-agi/glm-ocr-onnx-webgpu';
            
            const vlm = await pipeline('image-to-text', modelId, {
                device: 'webgpu', // Force WebGPU for high speed
                progress_callback: x => {
                    if(x.status === 'downloading') {
                        self.postMessage({ status: 'progress', message: `Downloading GLM-OCR: ${Math.round(x.progress)}%` });
                    }
                }
            });
            
            self.postMessage({ status: 'info', message: 'Running GLM-OCR Inference...' });
            
            // Ask the model for structured receipt data
            const output = await vlm(base64Image, {
                max_new_tokens: 128,
                prompt: "Extract receipt info: Vendor, Total Amount, Date."
            });
            
            const rawText = output[0].generated_text;
            
            // Basic parsing of the LLM output
            const amountMatch = rawText.match(/(\d+[.,]\d{2})/);
            resultData.amount = amountMatch ? amountMatch[1].replace(',', '.') : "";
            resultData.vendor = "GLM: " + rawText.substring(0, 20);
            resultData.date = new Date().toISOString().split('T')[0];

        } catch (e) {
             throw new Error("GLM-OCR WebGPU error (Check if WebGPU is enabled): " + e.message);
        }
    }

    // Send back extracted data
    self.postMessage({ status: 'success', data: resultData });
}


async function processTesseractJs(base64Image) {
    self.postMessage({ status: 'progress', message: 'Loading Tesseract.js (20MB)...' });
    
    // Import Tesseract.js
    importScripts('https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js');
    
    try {
        const worker = await Tesseract.createWorker('eng', 1, {
            logger: m => {
                if(m.status === 'recognizing text') {
                    self.postMessage({ status: 'progress', message: `OCR Progress: ${Math.round(m.progress * 100)}%` });
                }
            }
        });

        self.postMessage({ status: 'info', message: 'Extracting text...' });
        const { data: { text } } = await worker.recognize(base64Image);
        await worker.terminate();

        // Very basic regex to find a price
        const priceMatch = text.match(/\$?\s*(\d{1,4}[.,]\d{2})/);
        const amount = priceMatch ? priceMatch[1].replace(',', '.') : "";
        
        // Very basic date regex
        const dateMatch = text.match(/((0?[1-9]|1[012])[- \/.](0?[1-9]|[12][0-9]|3[01])[- \/.](19|20)\d\d)/);
        const dateStr = dateMatch ? dateMatch[0] : ""; // Needs to be formatted YYYY-MM-DD for input type date

        self.postMessage({ status: 'success', data: {
            vendor: "Extracted via Tesseract",
            amount: amount,
            date: new Date().toISOString().split('T')[0] // Fallback to today
        }});

    } catch (e) {
        throw new Error("Tesseract execution error: " + e.message);
    }
}
