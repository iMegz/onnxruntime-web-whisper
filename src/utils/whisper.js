import { WhisperFeatureExtractor } from "@xenova/transformers";
import { InferenceSession } from "onnxruntime-web";

export async function load_model(model_path = "./model_int8.onnx") {
    try {
        const sess = await InferenceSession.create(model_path);
        console.log(sess.inputNames);
        console.log(sess.outputNames);
        return sess;
    } catch (error) {
        console.error(`Failed to load model ${model_path}`);
        return null;
    }
}

/**
 * Extract feature from audio to use it in whipser model
 * @param {ArrayBuffer} buffer Audio buffer
 * @returns {Promise} A promise that resovles with input features
 */
export async function feature_extraction(buffer) {
    const featureConfig = {
        sampling_rate: 16000,
        n_fft: 400,
        feature_size: 80,
    };
    const featureExtractor = new WhisperFeatureExtractor(featureConfig);

    return featureExtractor(buffer);
}
