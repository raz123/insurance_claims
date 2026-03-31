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
    // Dynamic import of transformers.js (runs only when needed)
    self.postMessage({ status: 'info', message: 'Importing ONNX Runtime...' });
    
    // In a real production app, we would use a bundler or import map. 
    // Here we use the pre-built CDN version for the browser worker.
    importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');

    const { env, pipeline, AutoProcessor, Florence2ForConditionalGeneration, RawImage } = self.transformers;
    
    // Optimize for browser environments: Disable local models, use wasm
    env.allowLocalModels = false;
    // For smaller downloads if needed (though florence is still >150mb)
    env.backends.onnx.wasm.numThreads = 1;

    let resultData = { vendor: "", amount: "", date: "" };

    if (engineName === 'florence') {
        self.postMessage({ status: 'progress', message: 'Downloading Florence-2 Vision Model (~150MB). This happens once...' });
        
        try {
            // Note: Florence-2 isn't fully supported in official v2 @xenova branch easily, 
            // usually requires v3 alphas with WebGPU. 
            // For this mock plan, we simulate the complex init and fallback to a mock response if it fails.
            
            // Convert base64 to Blob to Image
            const imgResponse = await fetch(base64Image);
            const blob = await imgResponse.blob();
            
            self.postMessage({ status: 'info', message: 'Simulating Florence-2 inference for web worker...' });
            
            // SIMULATED INFERENCE TIME (Since Florence-2 is massive and requires v3 WebGPU)
            await new Promise(resolve => setTimeout(resolve, 3000));
            // Simulate extracting JSON
            resultData.vendor = "Mock Vendor (Florence)";
            resultData.amount = "142.50";
            resultData.date = new Date().toISOString().split('T')[0];

        } catch (e) {
             throw new Error("Florence-2 execution error: " + e.message);
        }

    } else if (engineName === 'trocr') {
        self.postMessage({ status: 'progress', message: 'Downloading TrOCR Model (~50MB)...' });
        
        try {
            const processor = await pipeline('image-to-text', 'Xenova/trocr-small-handwritten', {
                progress_callback: x => {
                    if(x.status === 'downloading') {
                        self.postMessage({ status: 'progress', message: `Downloading TrOCR: ${Math.round(x.progress)}%` });
                    }
                }
            });
            
            self.postMessage({ status: 'info', message: 'Running OCR...' });
            const output = await processor(base64Image);
            const rawText = output[0].generated_text;
            
            // We got raw text, now run simple regex (mocked for now)
            resultData.vendor = "Unknown (TrOCR Raw Text: " + rawText.substring(0,10) + ")";
            resultData.amount = "0.00";
            resultData.date = new Date().toISOString().split('T')[0];

        } catch (e) {
             throw new Error("TrOCR execution error: " + e.message);
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
