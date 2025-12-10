// JS Backend for Expression Recognition
// This provides a pure JavaScript alternative to the WASM backend

import { 
    INPUT_DIM, 
    HIDDEN1_DIM, 
    HIDDEN2_DIM, 
    OUTPUT_DIM,
    LAYER1_WEIGHT, 
    LAYER1_BIAS,
    LAYER2_WEIGHT,
    LAYER2_BIAS,
    LAYER3_WEIGHT,
    LAYER3_BIAS,
    CLASS_NAMES
} from './model_weights.js';

import {
    EMOTION_TO_EMOJI,
    EMOJI_WIDTH,
    EMOJI_HEIGHT
} from './emoji_data.js';

// LeakyReLU activation function
function leakyReLU(x, negativeSlope = 0.01) {
    return x > 0 ? x : x * negativeSlope;
}

// Forward pass through the neural network
function forward(input) {
    // Layer 1: Linear + LeakyReLU
    const hidden1 = new Float32Array(HIDDEN1_DIM);
    for (let i = 0; i < HIDDEN1_DIM; i++) {
        let sum = LAYER1_BIAS[i];
        const rowOffset = i * INPUT_DIM;
        for (let j = 0; j < INPUT_DIM; j++) {
            sum += LAYER1_WEIGHT[rowOffset + j] * input[j];
        }
        hidden1[i] = leakyReLU(sum);
    }

    // Layer 2: Linear + LeakyReLU
    const hidden2 = new Float32Array(HIDDEN2_DIM);
    for (let i = 0; i < HIDDEN2_DIM; i++) {
        let sum = LAYER2_BIAS[i];
        const rowOffset = i * HIDDEN1_DIM;
        for (let j = 0; j < HIDDEN1_DIM; j++) {
            sum += LAYER2_WEIGHT[rowOffset + j] * hidden1[j];
        }
        hidden2[i] = leakyReLU(sum);
    }

    // Layer 3: Output Linear
    const output = new Float32Array(OUTPUT_DIM);
    for (let i = 0; i < OUTPUT_DIM; i++) {
        let sum = LAYER3_BIAS[i];
        const rowOffset = i * HIDDEN2_DIM;
        for (let j = 0; j < HIDDEN2_DIM; j++) {
            sum += LAYER3_WEIGHT[rowOffset + j] * hidden2[j];
        }
        output[i] = sum;
    }

    return output;
}

// Apply affine transformation to a point
function transformPoint(point, matrix) {
    return {
        x: matrix[0] * point.x + matrix[1] * point.y + matrix[2],
        y: matrix[3] * point.x + matrix[4] * point.y + matrix[5]
    };
}

// Calculate mean of points
function meanPoint(points) {
    let sumX = 0, sumY = 0;
    for (const p of points) {
        sumX += p.x;
        sumY += p.y;
    }
    return { x: sumX / points.length, y: sumY / points.length };
}

// Estimate partial affine transformation (similarity transform)
function estimateAffinePartial2D(srcPoints, dstPoints) {
    const srcCenter = meanPoint(srcPoints);
    const dstCenter = meanPoint(dstPoints);
    
    let numA = 0, numB = 0, den = 0;
    
    for (let i = 0; i < srcPoints.length; i++) {
        const sx = srcPoints[i].x - srcCenter.x;
        const sy = srcPoints[i].y - srcCenter.y;
        const dx = dstPoints[i].x - dstCenter.x;
        const dy = dstPoints[i].y - dstCenter.y;
        
        numA += sx * dx + sy * dy;
        numB += sx * dy - sy * dx;
        den += sx * sx + sy * sy;
    }
    
    if (den === 0) return null;
    
    const a = numA / den;
    const b = numB / den;
    
    const tx = dstCenter.x - (a * srcCenter.x - b * srcCenter.y);
    const ty = dstCenter.y - (b * srcCenter.x + a * srcCenter.y);
    
    return [a, -b, tx, b, a, ty];
}

// Calculate 3x3 determinant
function getDeterminant3x3(a, b, c, d, e, f, g, h, i) {
    return (a * (e * i - f * h) -
            b * (d * i - f * g) +
            c * (d * h - e * g));
}

// Estimate full affine transformation using 3 point pairs
function estimateAffineFull2D(ptsIn, ptsOut) {
    if (ptsIn.length !== 3 || ptsOut.length !== 3) return null;
    
    const x1 = ptsIn[0].x, y1 = ptsIn[0].y;
    const x2 = ptsIn[1].x, y2 = ptsIn[1].y;
    const x3 = ptsIn[2].x, y3 = ptsIn[2].y;
    
    const u1 = ptsOut[0].x, v1 = ptsOut[0].y;
    const u2 = ptsOut[1].x, v2 = ptsOut[1].y;
    const u3 = ptsOut[2].x, v3 = ptsOut[2].y;
    
    const det_A = getDeterminant3x3(
        x1, y1, 1,
        x2, y2, 1,
        x3, y3, 1);
    
    if (Math.abs(det_A) < 1e-9) {
        return null;
    }
    
    const det_a = getDeterminant3x3(u1, y1, 1, u2, y2, 1, u3, y3, 1);
    const det_b = getDeterminant3x3(x1, u1, 1, x2, u2, 1, x3, u3, 1);
    const det_c = getDeterminant3x3(x1, y1, u1, x2, y2, u2, x3, y3, u3);
    const det_d = getDeterminant3x3(v1, y1, 1, v2, y2, 1, v3, y3, 1);
    const det_e = getDeterminant3x3(x1, v1, 1, x2, v2, 1, x3, v3, 1);
    const det_f = getDeterminant3x3(x1, y1, v1, x2, y2, v2, x3, y3, v3);
    
    const a = det_a / det_A;
    const b = det_b / det_A;
    const c = det_c / det_A;
    const d = det_d / det_A;
    const e = det_e / det_A;
    const f = det_f / det_A;
    
    return [a, b, c, d, e, f];
}

// Apply affine transformation to a point with 2x3 matrix [a, b, c, d, e, f]
function applyAffineTransform(pt, matrix) {
    return {
        x: matrix[0] * pt.x + matrix[1] * pt.y + matrix[2],
        y: matrix[3] * pt.x + matrix[4] * pt.y + matrix[5]
    };
}

// Overlay emoji onto canvas using warp affine
function overlayWarpAffine(canvasData, canvasWidth, canvasHeight, srcData, srcWidth, srcHeight, ptsCanvas, ptsSrc) {
    // Calculate forward and inverse affine matrices
    const forwardMatrix = estimateAffineFull2D(ptsSrc, ptsCanvas);
    if (!forwardMatrix) return false;
    
    const inverseMatrix = estimateAffineFull2D(ptsCanvas, ptsSrc);
    if (!inverseMatrix) return false;
    
    // Calculate bounding box in canvas space
    const corners = [
        { x: 0, y: 0 },
        { x: srcWidth, y: 0 },
        { x: srcWidth, y: srcHeight },
        { x: 0, y: srcHeight }
    ];
    
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    for (const pt of corners) {
        const tf = applyAffineTransform(pt, forwardMatrix);
        if (tf.x < minX) minX = tf.x;
        if (tf.x > maxX) maxX = tf.x;
        if (tf.y < minY) minY = tf.y;
        if (tf.y > maxY) maxY = tf.y;
    }
    
    const startX = Math.max(0, Math.floor(minX));
    const endX = Math.min(canvasWidth, Math.ceil(maxX));
    const startY = Math.max(0, Math.floor(minY));
    const endY = Math.min(canvasHeight, Math.ceil(maxY));
    
    const [a, b, c, d, e, f] = inverseMatrix;
    
    // Iterate through canvas pixels and blend emoji
    for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
            // Apply inverse transform to find source pixel
            const srcX = Math.round(a * x + b * y + c);
            const srcY = Math.round(d * x + e * y + f);
            
            if (srcX >= 0 && srcX < srcWidth && srcY >= 0 && srcY < srcHeight) {
                const srcIdx = (srcY * srcWidth + srcX) * 4;
                const dstIdx = (y * canvasWidth + x) * 4;
                
                const alpha = srcData[srcIdx + 3];
                
                // Only copy if alpha is non-zero
                if (alpha > 0) {
                    canvasData[dstIdx] = srcData[srcIdx];         // R
                    canvasData[dstIdx + 1] = srcData[srcIdx + 1]; // G
                    canvasData[dstIdx + 2] = srcData[srcIdx + 2]; // B
                    canvasData[dstIdx + 3] = srcData[srcIdx + 3]; // A
                }
            }
        }
    }
    
    return true;
}

// Process frame with JS backend
export function processFrameJS(imageData, faces) {
    if (!faces || faces.length === 0) {
        return null;
    }
    
    const width = imageData.width;
    const height = imageData.height;
    
    // Constants for face alignment
    const LEFT_IRIS_INDICES = [468, 469, 470, 471, 472];
    const RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477];
    const LOWER_LIP_CENTER_INDEX = 17;
    const NORMALIZED_IMAGE_SIZE = 256;
    const EYE_DISTANCE = 96;
    
    // Store first face emotion for display
    let firstFaceEmotion = null;
    
    // Process each face
    for (let faceIndex = 0; faceIndex < faces.length; faceIndex++) {
        const landmarks = faces[faceIndex];
        
        // Convert normalized landmarks to pixel coordinates
        const landmarkPoints = landmarks.map(lm => ({
            x: lm.x * width,
            y: lm.y * height
        }));
        
        // Get iris centers
        const leftIrisPoints = LEFT_IRIS_INDICES.map(i => landmarkPoints[i]);
        const rightIrisPoints = RIGHT_IRIS_INDICES.map(i => landmarkPoints[i]);
        
        const leftIrisCenter = meanPoint(leftIrisPoints);
        const rightIrisCenter = meanPoint(rightIrisPoints);
        
        // Define source and destination points for alignment
        const srcPoints = [leftIrisCenter, rightIrisCenter];
        const dstPoints = [
            { x: NORMALIZED_IMAGE_SIZE / 2 - EYE_DISTANCE / 2, y: 50 },
            { x: NORMALIZED_IMAGE_SIZE / 2 + EYE_DISTANCE / 2, y: 50 }
        ];
        
        // Estimate affine transformation
        const affineMatrix = estimateAffinePartial2D(srcPoints, dstPoints);
        if (!affineMatrix) continue;
        
        // Transform all landmarks
        const normalizedPoints = landmarkPoints.map(p => transformPoint(p, affineMatrix));
        
        // Flatten normalized points to input array
        const input = new Float32Array(normalizedPoints.length * 2);
        for (let i = 0; i < normalizedPoints.length; i++) {
            input[i * 2] = normalizedPoints[i].x;
            input[i * 2 + 1] = normalizedPoints[i].y;
        }
        
        // Run inference
        const output = forward(input);
        
        // Find max class
        let maxIndex = 0;
        let maxValue = output[0];
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }
        
        const emotion = CLASS_NAMES[maxIndex];
        
        // Save first face emotion for display
        if (faceIndex === 0) {
            firstFaceEmotion = {
                emotion: emotion,
                confidence: maxValue,
                output: output
            };
        }
        
        // Overlay emoji on face
        const emojiData = EMOTION_TO_EMOJI[emotion.toLowerCase()];
        if (emojiData) {
            // Canvas points: left eye, right eye, lower lip center
            const canvasPoints = [
                leftIrisCenter,
                rightIrisCenter,
                landmarkPoints[LOWER_LIP_CENTER_INDEX]
            ];
            
            // Emoji source points
            const emojiPoints = emojiData.points;
            
            // Overlay the emoji
            overlayWarpAffine(
                imageData.data,
                width,
                height,
                emojiData.pixels,
                EMOJI_WIDTH,
                EMOJI_HEIGHT,
                canvasPoints,
                emojiPoints
            );
        }
    }
    
    // Return emotion for the first face
    return firstFaceEmotion;
}

// Initialize JS backend (for compatibility with WASM backend)
export function initJSBackend() {
    console.log('JS Backend initialized');
    console.log(`Input dimension: ${INPUT_DIM}`);
    console.log(`Hidden layers: ${HIDDEN1_DIM}, ${HIDDEN2_DIM}`);
    console.log(`Output classes: ${OUTPUT_DIM}`);
    console.log(`Classes: ${CLASS_NAMES.join(', ')}`);
    return true;
}
