console.log('[Worker] Global Init: v1.1');
import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@3.0.0-alpha.19';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.mjs';

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
        image: 59280,
        sop: 59255
    }
};

// State management
let sessions = {};
let tokenizer = null;

self.onmessage = async (e) => {
    const { action, imageBase64 } = e.data;
    if (action === 'PROCESS_IMAGE') {
        try {
            await initEngine();
            const result = await runInference(imageBase64);
            self.postMessage({ status: 'success', data: result });
        } catch (err) {
            self.postMessage({ status: 'error', error: err.message });
        }
    }
};

async function initEngine() {
    if (tokenizer && sessions.language_model) return;
    
    console.log('[Worker] Starting Engine Initialization...');
    
    try {
        tokenizer = await AutoTokenizer.from_pretrained(CONFIG.modelId);
        console.log('[Worker] Tokenizer loaded successfully.');
    } catch (e) {
        console.error('[Worker] Tokenizer load failed:', e);
        throw e;
    }

    const baseUrl = `https://huggingface.co/${CONFIG.modelId}/resolve/main`;
    const components = [
        'vision_encoder_int8.onnx',
        'language_model_int8.onnx',
        'text_embeddings.onnx',
        'kv/prefill_int8.onnx',
        'kv/decode_int8.onnx'
    ];

    const totalFiles = components.length;
    let loadedFiles = 0;

    for (const file of components) {
        const key = file.replace('.onnx', '').replace('kv/', '');
        if (!sessions[key]) {
            console.log(`[Worker] Loading session: ${file}...`);
            self.postMessage({ 
                status: 'progress', 
                message: `Downloading Component: ${file} (${loadedFiles + 1}/${totalFiles})`,
                percent: Math.round((loadedFiles / totalFiles) * 100)
            });
            try {
                sessions[key] = await ort.InferenceSession.create(`${baseUrl}/${file}`, {
                    executionProviders: ['webgpu']
                });
                console.log(`[Worker] Session ${key} loaded on WebGPU.`);
            } catch (e) {
                console.error(`[Worker] Failed to load ${file}:`, e);
                throw e;
            }
            loadedFiles++;
            self.postMessage({ 
                status: 'progress', 
                percent: Math.round((loadedFiles / totalFiles) * 100)
            });
        } else {
            loadedFiles++;
        }
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
    const visionOutput = await sessions.vision_encoder_int8.run({ input: visionTensor });
    
    // 3. Spatial Merge (2x2 downsampling)
    const mergedEmbeds = mergeVisionTokens(visionOutput.output);
    
    // 4. Tokenization
    const prompt = "Extract receipt info: Vendor, Total Amount, Date.";
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

    while (step < CONFIG.maxTokens) {
        const decoded = tokenizer.decode([Number(nextToken)], { skip_special_tokens: true });
        if (decoded.includes('</s>')) break;
        generatedText += decoded;
        
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
        
        if (step % 5 === 0) {
            self.postMessage({ status: 'progress', message: `Extracted: ${generatedText}` });
        }
    }

    const amountMatch = generatedText.match(/(\d+[.,]\d{2})/);
    return { 
        vendor: "GLM: " + (generatedText.substring(0, 20)), 
        amount: amountMatch ? amountMatch[1].replace(',', '.') : "", 
        date: new Date().toISOString().split('T')[0] 
    };
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
    const [batch, seq, dim] = tensor.dims; // [1, 576, 1176]
    const data = tensor.data;
    const mergedData = new Float32Array(batch * 144 * dim * 4);
    // Simple 2x2 concat merge logic
    // This is a placeholder for the actual spatial merge math
    return new ort.Tensor('float32', new Float32Array(1 * 144 * 1536), [1, 144, 1536]);
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
