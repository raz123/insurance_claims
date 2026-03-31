console.log('[Worker] Global Init: v1.3.7');
import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.2.1/dist/transformers.js';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.mjs';
import { getModel, setModel } from './db-storage.js';

// Configuration for GLM-OCR (brad-agi/glm-ocr-onnx-webgpu)
const CONFIG = {
    modelId: 'brad-agi/glm-ocr-onnx-webgpu',
    visionInputSize: 336,
    patchSize: 14,
    tokensPerImage: 144, // 12x12 grid after spatial merge (2x2)
    maxTokens: 256,
    tokens: {
        start: 59256,
        end: 59257,
        block: 59280, // Using 59280 for <image> block
        sop: 59255
    }
};

const BASE_URL = 'https://huggingface.co/brad-agi/glm-ocr-onnx-webgpu/resolve/main/';
const FILES = {
    vision: 'vision_encoder_int8.onnx',
    language: 'language_model_int8.onnx',
    embeds: 'text_embeddings.onnx',
    prefill: 'kv/prefill_int8.onnx',
    decode: 'kv/decode_int8.onnx'
};

// State management
let sessions = {};
let tokenizer = null;

self.onmessage = async (e) => {
    const { action, imageBase64, files } = e.data;
    if (action === 'PROCESS_IMAGE') {
        try {
            await initEngine();
            const result = await runInference(imageBase64);
            self.postMessage({ status: 'success', data: result });
        } catch (err) {
            console.error('[Worker] OCR Error:', err);
            self.postMessage({ status: 'error', error: err.message });
        }
    } else if (action === 'LOAD_MODELS') {
        // Handle local model provisioning
        for (const [name, buffer] of Object.entries(files)) {
             await setModel(name, buffer);
        }
        await initEngine();
    }
};

async function fetchWithProgress(name, url) {
    // 1. Check Local Cache (IndexedDB)
    const cached = await getModel(name);
    if (cached) {
        console.log(`[Worker] Loaded from Cache: ${name}`);
        self.postMessage({ status: 'progress', file: name, percent: 100 });
        return cached;
    }

    // 2. Fetch from Cloud
    const response = await fetch(url);
    const total = parseInt(response.headers.get('content-length'), 10);
    let loaded = 0;

    const reader = response.body.getReader();
    const chunks = [];

    while(true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        self.postMessage({ 
            status: 'progress', 
            file: name, 
            percent: Math.round((loaded / total) * 100) 
        });
    }

    const buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        buffer.set(chunk, offset);
        offset += chunk.length;
    }

    // 3. Store in Cache
    await setModel(name, buffer.buffer);
    return buffer.buffer;
}

async function initEngine() {
    if (tokenizer && sessions.vision) return; // Already init

    console.log('[Worker] Starting Engine Initialization...');
    self.postMessage({ status: 'progress', message: 'Loading Tokenizer...', percent: 5 });

    // 1. Tokenizer
    tokenizer = await AutoTokenizer.from_pretrained(CONFIG.modelId);
    console.log('[Worker] Tokenizer loaded successfully.');

    // 2. Models
    const modelKeys = Object.keys(FILES);
    for (const key of modelKeys) {
        const fileName = FILES[key];
        const url = BASE_URL + fileName;
        
        console.log(`[Worker] Sourcing model: ${fileName}...`);
        const buffer = await fetchWithProgress(fileName, url);
        
        sessions[key] = await ort.InferenceSession.create(buffer, {
            executionProviders: ['webgpu']
        });
        console.log(`[Worker] Session ${key} ready. Inputs: ${sessions[key].inputNames}`);
    }

    console.log('[Worker] All 5 sessions ready.');
    self.postMessage({ status: 'progress', message: 'GPU Engines Ready.', percent: 100 });
}

async function runInference(imageBase64) {
    self.postMessage({ status: 'info', message: 'Preprocessing Image...' });
    
    // 1. Image Preprocessing
    const visionTensor = await prepareImageTensor(imageBase64);
    
    // 2. Vision Encoding
    self.postMessage({ status: 'info', message: 'Running Visual Encoder...' });
    // Robust I/O: Find the input name (likely 'pixel_values')
    const visionInputName = sessions.vision.inputNames[0];
    const visionFeeds = {};
    visionFeeds[visionInputName] = visionTensor;
    
    // Fix: Add grid_thw if the model requires it (for GLM models)
    if (sessions.vision.inputNames.includes('grid_thw')) {
        visionFeeds['grid_thw'] = new ort.Tensor('int64', BigInt64Array.from([1n, 24n, 24n]), [1, 3]);
    }
    
    const visionOutput = await sessions.vision.run(visionFeeds);
    
    // 3. Spatial Merge (2x2 downsampling)
    const mergedEmbeds = mergeVisionTokens(visionOutput.output);
    
    // 4. Tokenization
    const prompt = "请按下列JSON格式输出图中信息:\n{ \"vendor\": \"\", \"amount\": \"\", \"date\": \"\" }";
    const { input_ids } = await tokenizer(prompt);
    const textIds = Array.from(input_ids.data);
    
    // 5. Construct Full Input Embeddings
    const imageStartEmbed = await getEmbedding(CONFIG.tokens.start);
    const imageEndEmbed = await getEmbedding(CONFIG.tokens.end);
    const textEmbeds = await getEmbeddings(textIds);
    const sopEmbed = await getEmbedding(CONFIG.tokens.sop);

    const fullEmbeds = concatenateTensors([imageStartEmbed, mergedEmbeds, imageEndEmbed, textEmbeds, sopEmbed]);
    const seqLen = fullEmbeds.dims[1];

    // 6. Construct 3D Position IDs
    const posIds = generate3DPositionIds(seqLen, textIds.length);
    
    // 7. Prefill
    self.postMessage({ status: 'info', message: 'AI Prefill...' });
    const prefillOutput = await sessions.prefill.run({
        inputs_embeds: fullEmbeds,
        position_ids: posIds
    });
    
    // 8. Autoregressive Decode Loop
    let nextToken = sampleLogits(prefillOutput.logits);
    let pastKeyValues = prefillOutput.past_key_values;
    let generatedText = "";
    let step = 0;

    self.postMessage({ status: 'info', message: 'Generating Extraction...' });
    
    while (step < CONFIG.maxTokens) {
        const decoded = tokenizer.decode([Number(nextToken)], { skip_special_tokens: true });
        
        // Stop if we hit EOS or common end markers
        if (decoded.includes('</s>')) break;
        
        generatedText += decoded;
        
        // Stream the token back to UI
        self.postMessage({ status: 'stream', token: decoded, text: generatedText });

        // Next Step
        const nextEmbed = await getEmbedding(nextToken);
        const nextPos = generate3DPositionIds(1, 0, seqLen + step);
        
        const decodeOutput = await sessions.decode.run({
            inputs_embeds: nextEmbed,
            position_ids: nextPos,
            past_key_values: pastKeyValues
        });
        
        nextToken = sampleLogits(decodeOutput.logits);
        pastKeyValues = decodeOutput.past_key_values;
        step++;
    }

    // Attempt to parse JSON from the stream
    let result = { vendor: "Unknown", amount: "", date: "" };
    try {
        const jsonMatch = generatedText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            result = { ...result, ...parsed };
        }
    } catch (e) {
        console.warn('[Worker] JSON Parse failed, using regex fallback.');
        const amountMatch = generatedText.match(/(\d+[.,]\d{2})/);
        result.amount = amountMatch ? amountMatch[1] : "";
    }
    
    return result;
}

async function prepareImageTensor(base64) {
    const img = await createImageBitmap(await (await fetch(base64)).blob());
    const canvas = new OffscreenCanvas(336, 336);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 336, 336);
    const { data } = ctx.getImageData(0, 0, 336, 336);
    const floatData = new Float32Array(3 * 336 * 336);
    for (let i = 0; i < 336 * 336; i++) {
        floatData[i] = (data[i * 4] / 255 - 0.481) / 0.268;
        floatData[i + 336*336] = (data[i * 4 + 1] / 255 - 0.457) / 0.261;
        floatData[i + 2*336*336] = (data[i * 4 + 2] / 255 - 0.408) / 0.275;
    }
    return new ort.Tensor('float32', floatData, [1, 3, 336, 336]);
}

function mergeVisionTokens(tensor) {
    const [batch, seq, dim] = tensor.dims; 
    console.log(`[Worker] Vision Output Shape: [${batch}, ${seq}, ${dim}]`);
    
    if (seq === 144) {
        console.log('[Worker] Vision already merged (144 tokens).');
        return tensor;
    }

    // Manual Spatial Merge (2x2 pooling)
    // GLM Grid is 24x24 = 576. Language model expects 12x12 = 144.
    console.log('[Worker] Performing 2x2 Spatial Merge...');
    const data = tensor.data;
    const mergedData = new Float32Array(batch * 144 * dim);
    
    for (let b = 0; b < batch; b++) {
        for (let r = 0; r < 12; r++) {
            for (let c = 0; c < 12; c++) {
                const targetIdx = (b * 144 + r * 12 + c) * dim;
                // Sum 2x2 block from 24x24 grid
                for (let i = 0; i < 2; i++) {
                    for (let j = 0; j < 2; j++) {
                        const sourceR = r * 2 + i;
                        const sourceC = c * 2 + j;
                        const sourceIdx = (b * 576 + sourceR * 24 + sourceC) * dim;
                        for (let k = 0; k < dim; k++) {
                            mergedData[targetIdx + k] += data[sourceIdx + k] / 4.0;
                        }
                    }
                }
            }
        }
    }
    return new ort.Tensor('float32', mergedData, [batch, 144, dim]);
}

function generate3DPositionIds(seqLen, promptLen, offset = 0) {
    const data = new BigInt64Array(4 * seqLen);
    for (let i = 0; i < seqLen; i++) {
        const gIdx = BigInt(i + offset);
        data[i] = 0n;
        data[seqLen + i] = gIdx;
        if (gIdx >= 1n && gIdx <= 144n) {
            const idx = Number(gIdx) - 1;
            data[2 * seqLen + i] = BigInt(Math.floor(idx / 12));
            data[3 * seqLen + i] = BigInt(idx % 12);
        } else {
            data[2 * seqLen + i] = 0n;
            data[3 * seqLen + i] = 0n;
        }
    }
    return new ort.Tensor('int64', data, [4, 1, seqLen]);
}

async function getEmbedding(id) {
    const input = new ort.Tensor('int64', BigInt64Array.from([BigInt(id)]), [1, 1]);
    const { output } = await sessions.text_embeddings.run({ input });
    return output;
}

async function getEmbeddings(ids) {
    const input = new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const { output } = await sessions.text_embeddings.run({ input });
    return output;
}

function concatenateTensors(tensors) {
    const totalSeq = tensors.reduce((acc, t) => acc + t.dims[1], 0);
    const hiddenSize = tensors[0].dims[2];
    const data = new Float32Array(totalSeq * hiddenSize);
    let offset = 0;
    for (const t of tensors) {
        data.set(t.data, offset);
        offset += t.data.length;
    }
    return new ort.Tensor('float32', data, [1, totalSeq, hiddenSize]);
}

function sampleLogits(logits) {
    const data = logits.data;
    let maxIdx = 0, maxVal = -Infinity;
    for (let i = 0; i < data.length; i++) {
        if (data[i] > maxVal) { maxVal = data[i]; maxIdx = i; }
    }
    return BigInt(maxIdx);
}
