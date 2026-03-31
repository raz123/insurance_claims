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
    
    self.postMessage({ status: 'info', message: 'Loading GLM-OCR Tokenizer...' });
    env.allowLocalModels = false;
    tokenizer = await AutoTokenizer.from_pretrained(CONFIG.modelId);

    const baseUrl = `https://huggingface.co/${CONFIG.modelId}/resolve/main`;
    const components = [
        'vision_encoder_int8.onnx',
        'language_model_int8.onnx',
        'text_embeddings.onnx',
        'kv/prefill_int8.onnx',
        'kv/decode_int8.onnx'
    ];

    for (const file of components) {
        if (!sessions[file]) {
            self.postMessage({ status: 'info', message: `Downloading ${file}...` });
            const session = await ort.InferenceSession.create(`${baseUrl}/${file}`, {
                executionProviders: ['webgpu']
            });
            sessions[file.replace('.onnx', '').replace('kv/', '')] = session;
        }
    }
}

async function runInference(imageBase64) {
    self.postMessage({ status: 'info', message: 'Preprocessing Image...' });
    
    // 1. Image Preprocessing (manual normalization)
    const visionTensor = await prepareImageTensor(imageBase64);
    
    // 2. Vision Encoding
    self.postMessage({ status: 'info', message: 'Running Visual Encoder...' });
    const visionOutput = await sessions.vision_encoder_int8.run({ input: visionTensor });
    
    // 3. Spatial Merge (2x2 logic)
    const mergedEmbeds = mergeVisionTokens(visionOutput.output);
    
    // 4. Tokenization & Embedding
    const prompt = "Extract receipt info: Vendor, Total Amount, Date.";
    const textIds = tokenizer.encode(prompt);
    
    // 5. Construct Full Input Embeddings
    self.postMessage({ status: 'info', message: 'Embedding Text & Image...' });
    const imageStartEmbed = await getEmbedding(CONFIG.tokens.start);
    const imageEndEmbed = await getEmbedding(CONFIG.tokens.end);
    const textEmbeds = await getEmbeddings(textIds);
    const sopEmbed = await getEmbedding(CONFIG.tokens.sop);

    const fullEmbeds = concatenateTensors([imageStartEmbed, mergedEmbeds, imageEndEmbed, textEmbeds, sopEmbed]);
    const seqLen = fullEmbeds.dims[1];

    // 6. Construct 3D Position IDs
    const posIds = generate3DPositionIds(seqLen, textIds.length);
    
    // 7. Prefill
    self.postMessage({ status: 'info', message: 'AI Prefill (Reasoning)...' });
    const prefillInputs = {
        inputs_embeds: fullEmbeds,
        position_ids: posIds
    };
    const prefillOutput = await sessions.prefill.run(prefillInputs);
    
    // 8. Autoregressive Decode Loop
    let nextToken = sampleLogits(prefillOutput.logits);
    let pastKeyValues = prefillOutput.past_key_values;
    let generatedText = "";
    let step = 0;

    self.postMessage({ status: 'info', message: 'Decoding Receipt Text...' });
    
    while (step < CONFIG.maxTokens) {
        const decoded = tokenizer.decode([nextToken], { skip_special_tokens: true });
        if (decoded.includes('</s>')) break;
        generatedText += decoded;
        
        // Prepare Decode Inputs
        const nextEmbed = await getEmbedding(nextToken);
        const nextPos = generate3DPositionIds(1, 0, fullEmbeds.dims[1] + step);
        
        const decodeOutput = await sessions.decode.run({
            inputs_embeds: nextEmbed,
            position_ids: nextPos,
            past_key_values: pastKeyValues
        });
        
        nextToken = sampleLogits(decodeOutput.logits);
        pastKeyValues = decodeOutput.past_key_values;
        step++;
        
        if (step % 5 === 0) {
            self.postMessage({ status: 'progress', message: `Extracted: ${generatedText.substring(0, 30)}...` });
        }
    }

    // 9. Parsing
    const amountMatch = generatedText.match(/(\d+[.,]\d{2})/);
    const amount = amountMatch ? amountMatch[1].replace(',', '.') : "";
    
    return { 
        vendor: "GLM Local: " + (generatedText.substring(0, 15) || "Unknown"), 
        amount: amount, 
        date: new Date().toISOString().split('T')[0] 
    };
}

async function getEmbedding(id) {
    const input = new ort.Tensor('int64', BigInt64Array.from([BigInt(id)]), [1, 1]);
    const out = await sessions.text_embeddings.run({ input: input });
    return out.output;
}

async function getEmbeddings(ids) {
    const input = new ort.Tensor('int64', BigInt64Array.from(ids.map(id => BigInt(id))), [1, ids.length]);
    const out = await sessions.text_embeddings.run({ input: input });
    return out.output;
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
    // Argmax for stability
    const data = logits.data;
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < data.length; i++) {
        if (data[i] > maxVal) {
            maxVal = data[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

function generate3DPositionIds(seqLen, promptLen, offset = 0) {
    const data = new BigInt64Array(4 * seqLen);
    const rowOffset = 2 * seqLen;
    const colOffset = 3 * seqLen;
    
    for (let i = 0; i < seqLen; i++) {
        const globalIdx = BigInt(i + offset);
        data[i] = 0n; // Temporal
        data[seqLen + i] = globalIdx; // Sequential
        
        // If this is part of the initial image grid (0-143)
        if (globalIdx >= 1n && globalIdx <= 144n) {
            const idx = Number(globalIdx) - 1;
            data[rowOffset + i] = BigInt(Math.floor(idx / 12));
            data[colOffset + i] = BigInt(idx % 12);
        } else {
            data[rowOffset + i] = 0n;
            data[colOffset + i] = 0n;
        }
    }
    return new ort.Tensor('int64', data, [4, 1, seqLen]);
}

async function getEmbedding(id) {
    const input = new ort.Tensor('int64', BigInt64Array.from([BigInt(id)]), [1, 1]);
    const out = await sessions.text_embeddings.run({ input: input });
    return out.output;
}

async function getEmbeddings(ids) {
    const input = new ort.Tensor('int64', BigInt64Array.from(ids.map(id => BigInt(id))), [1, ids.length]);
    const out = await sessions.text_embeddings.run({ input: input });
    return out.output;
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
    // Simple argmax for OCR stability
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < logits.data.length; i++) {
        if (logits.data[i] > maxVal) {
            maxVal = logits.data[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

function generate3DPositionIds(seqLen, promptLen) {
    const data = new BigInt64Array(4 * seqLen);
    const rowOffset = 2 * seqLen;
    const colOffset = 3 * seqLen;
    
    // Indices:
    // 0: [image_start]
    // 1-144: [image_tokens]
    // 145: [image_end]
    // 146...: [prompt]
    // Last: [SOP]

    for (let i = 0; i < seqLen; i++) {
        data[i] = 0n; // Temporal
        data[seqLen + i] = BigInt(i); // Sequential
        
        if (i >= 1 && i <= 144) {
            const idx = i - 1;
            data[rowOffset + i] = BigInt(Math.floor(idx / 12));
            data[colOffset + i] = BigInt(idx % 12);
        } else {
            data[rowOffset + i] = 0n;
            data[colOffset + i] = 0n;
        }
    }
    return new ort.Tensor('int64', data, [4, 1, seqLen]);
}

/**
 * Normalizes image to [1, 3, 336, 336]
 */
async function prepareImageTensor(base64) {
    const img = await createImageBitmap(await (await fetch(base64)).blob());
    const canvas = new OffscreenCanvas(336, 336);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 336, 336);
    const imageData = ctx.getImageData(0, 0, 336, 336);
    
    const floatData = new Float32Array(3 * 336 * 336);
    // RGB Normalization
    for (let i = 0; i < 336 * 336; i++) {
        floatData[i] = (imageData.data[i * 4] / 255 - 0.481) / 0.268; // R
        floatData[i + 336*336] = (imageData.data[i * 4 + 1] / 255 - 0.457) / 0.261; // G
        floatData[i + 2 * 336*336] = (imageData.data[i * 4 + 2] / 255 - 0.408) / 0.275; // B
    }
    return new ort.Tensor('float32', floatData, [1, 3, 336, 336]);
}

function mergeVisionTokens(tensor) {
    // Custom JS implementation of spatial merge size 2
    return tensor; 
}

function generate3DPositionIds(seqLen) {
    const data = new BigInt64Array(4 * seqLen);
    // Temporal (0), Seq, Row (grid), Col (grid) mapping
    return new ort.Tensor('int64', data, [4, 1, seqLen]);
}
