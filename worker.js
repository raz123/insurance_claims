let worker;

// Listen for messages from the main thread
self.onmessage = async function(event) {
    const { action, imageBase64, engine } = event.data;

    if (action === 'PROCESS_IMAGE') {
        try {
            self.postMessage({ status: 'info', message: `Initializing ${engine}...` });

            if (engine === 'glm') {
                await processTransformersJs(imageBase64, engine);
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
    let pipeline, env;
    try {
        const hf = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19');
        pipeline = hf.pipeline;
        env = hf.env;
    } catch(err) {
        throw new Error("Failed to load Transformers.js WebGPU module: " + err.message);
    }
    
    // WebGPU optimization
    env.allowLocalModels = false;

    let resultData = { vendor: "", amount: "", date: "" };

    if (engineName === 'glm') {
        self.postMessage({ status: 'progress', message: 'Loading Florence-2 Base via WebGPU. This may take a moment...' });
        
        try {
            // we use Florence-2 which is fully supported by Transformers.js v3 webgpu pipeline natively
            const modelId = 'onnx-community/Florence-2-base-ft';
            
            const vlm = await pipeline('image-to-text', modelId, {
                device: 'webgpu', // Force WebGPU for high speed
                progress_callback: x => {
                    if(x.status === 'downloading') {
                        self.postMessage({ status: 'progress', message: `Downloading Model: ${Math.round(x.progress)}%` });
                    }
                }
            });
            
            self.postMessage({ status: 'info', message: 'Running GPU Inference...' });
            
            // Florence-2 expects task prompts inside angle brackets
            const output = await vlm(base64Image, {
                max_new_tokens: 128,
                prompt: "<OCR>"
            });
            
            const rawText = output[0].generated_text;
            
            // Basic parsing of the raw string output
            const amountMatch = rawText.match(/(\d+[.,]\d{2})/);
            resultData.amount = amountMatch ? amountMatch[1].replace(',', '.') : "";
            resultData.vendor = "Local AI: " + rawText.substring(0, 15);
            resultData.date = new Date().toISOString().split('T')[0];

        } catch (e) {
             throw new Error("WebGPU error (Check if WebGPU is enabled in browser): " + e.message);
        }
    }

    // Send back extracted data
    self.postMessage({ status: 'success', data: resultData });
}

