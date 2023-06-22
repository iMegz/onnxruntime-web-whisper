/*global BigInt */
/*global BigInt64Array */
import { AutoProcessor, WhisperTokenizer } from "@xenova/transformers";
import { InferenceSession, Tensor } from "onnxruntime-web";

console.warn = () => {};

function calc_perf(start) {
    const time = ~~(((performance.now() - start) / 1000) * 100) / 100;
    return time;
}

/**
 * @typedef {Object} modelConfig
 * @property {String} [encoder_path=""] Path to onnx encoder realtive to public
 * @property {String} [decoder_path=""] Path to onnx decoder realtive to public
 * @property {String} [tokenizer_config=""] Path to tokenizer_config.json and tokenizer.json folder relative to "public/models/"
 * @property {Boolean} [load_models=true] Load model automatically with class initialization
 */

/**
 * @typedef {Object} decodeStepInput
 * @property {Tensor} prev_decoder
 * @property {Number[] | BigInt64Array} input_ids
 * @property {Boolean} from_encoder
 */

class Whisper {
    #feature_extractor;
    #tokenizer;
    #encoder;
    #decoder;
    #config = {
        chunk_length: 30,
        feature_extractor_type: "WhisperFeatureExtractor",
        feature_size: 80,
        hop_length: 160,
        n_fft: 400,
        n_samples: 480000,
        nb_max_frames: 3000,
        padding_side: "right",
        padding_value: 0.0,
        processor_class: "WhisperProcessor",
        return_attention_mask: false,
        sampling_rate: 16000,
    };
    #loading_requirements;
    loaded = false;
    #temp_time;

    /**
     * Intitialize Whipser model object
     * @param {modelConfig} modelConfig
     */
    constructor({
        encoder_path = "",
        decoder_path = "",
        tokenizer_config = "",
        load_models = true,
    } = {}) {
        this.#loading_requirements = performance.now();

        this.#feature_extractor = AutoProcessor.from_pretrained(null, {
            config: this.#config,
        });
        this.#tokenizer = WhisperTokenizer.from_pretrained(tokenizer_config);

        if (load_models) {
            this.#encoder = InferenceSession.create(encoder_path);
            this.#decoder = InferenceSession.create(decoder_path);
        }
    }

    /**
     * Load model requirements
     */
    async load() {
        const loaders = [
            this.#encoder,
            this.#decoder,
            this.#feature_extractor,
            this.#tokenizer,
        ];

        try {
            await Promise.all(loaders);
            const time =
                ~~(
                    ((performance.now() - this.#loading_requirements) / 1000) *
                    100
                ) / 100;
            this.loaded = true;
            return `Model requirements loaded in ${time} s`;
        } catch (error) {
            throw Error(`Failed to load model requirements\n${error}`);
        }
    }

    /**
     * Apply feature extraction
     * @param {ArrayBuffer} audio_buffer Audio buffer sampled at 16KHz
     */
    async feature_extractor(audio_buffer) {
        this.#temp_time = performance.now();
        if (!this.loaded) await this.load();
        const extractor = await this.#feature_extractor;
        return await extractor(audio_buffer);
    }

    /**
     * Encode input features
     * @param {Tensor} input_features
     */
    async encode(input_features) {
        console.log(`Features extracted in ${calc_perf(this.#temp_time)}s`);
        this.#temp_time = performance.now();
        if (!this.loaded) await this.load();
        const enocder = await this.#encoder;
        return enocder.run({ input_features });
    }

    /**
     * Decode token ids
     * @param {Number[] | BigInt64Array} token_ids
     */
    async decode_tokens(token_ids) {
        console.log(`Decoded ${calc_perf(this.#temp_time)}s`);
        this.#temp_time = performance.now();

        if (!this.loaded) await this.load();
        const tokenizer = await this.#tokenizer;
        return tokenizer.decode(token_ids, { skip_special_tokens: true });
    }

    /**
     * Convert number array to BigInt64Array
     * @param {Number[]} array
     * @returns {BigInt64Array}
     */
    #toBigInt64Array(array) {
        return new BigInt64Array(
            array.map((i) => BigInt(i.toString().replace(",", "")))
        );
    }

    /**
     * Generate dummy input for first decoder step
     * @param {Number[] | BigInt64Array} input_ids
     * @param {Tensor} encoder_hidden_states
     */
    #dummy_input(input_ids, encoder_hidden_states) {
        const ENCODER_DIM = [1, 8, 1500, 64];
        const DECODER_DIM = [1, 8, 2, 64];

        const dummy_feed = {};

        for (let i = 0; i < 6; i++) {
            dummy_feed[`past_key_values.${i}.encoder.key`] = new Tensor(
                "float32",
                Array(768000).fill(0),
                ENCODER_DIM
            );
            dummy_feed[`past_key_values.${i}.encoder.value`] = new Tensor(
                "float32",
                Array(768000).fill(0),
                ENCODER_DIM
            );
            dummy_feed[`past_key_values.${i}.decoder.key`] = new Tensor(
                "float32",
                Array(1024).fill(0),
                DECODER_DIM
            );
            dummy_feed[`past_key_values.${i}.decoder.value`] = new Tensor(
                "float32",
                Array(1024).fill(0),
                DECODER_DIM
            );
        }
        dummy_feed["input_ids"] = input_ids;
        dummy_feed["encoder_hidden_states"] = encoder_hidden_states;
        dummy_feed["use_cache_branch"] = new Tensor("bool", [false], [1]);
        return dummy_feed;
    }

    /**
     * Decode one step resulting in one token
     * @param {Tensor} encoder_hidden_states
     * @param {decodeStepInput} decode_step_input
     */
    async #decode_step(
        encoder_hidden_states,
        {
            prev_decoder,
            input_ids = new Tensor(new BigInt64Array([1n, 50362n]), [1, 2]),
            from_encoder = false,
        } = {}
    ) {
        const decoder = await this.#decoder;
        if (from_encoder) {
            const input_feed = this.#dummy_input(
                input_ids,
                encoder_hidden_states
            );
            return decoder.run(input_feed);
        } else {
            const input_feed = {};

            Object.keys(prev_decoder).forEach((key) => {
                if (key !== "logits")
                    input_feed[key.replace("present", "past_key_values")] =
                        prev_decoder[key];
            });

            // input_feed["input_ids"] = new Tensor(
            //     new BigInt64Array(input_ids.map((id) => BigInt(id))),
            //     [1, input_ids.length]
            // );
            let temp_input_ids = this.#toBigInt64Array(input_ids);
            temp_input_ids = new Tensor(temp_input_ids, [1, input_ids.length]);
            input_feed["input_ids"] = temp_input_ids;
            input_feed["encoder_hidden_states"] = encoder_hidden_states;
            input_feed["use_cache_branch"] = new Tensor("bool", [false], [1]);
            return decoder.run(input_feed);
        }
    }

    /**
     * Get index of max value in second dimension
     * @param {Number[]} array
     * @param {Number} n
     * @returns {Number[]}
     */
    #argmax(array, n = 51864) {
        const result = [];

        for (let i = 0; i < array.length; i += n) {
            const subArray = array.slice(i, i + n);
            let maxIndex = 0;
            let maxValue = subArray[0];

            for (let j = 1; j < subArray.length; j++) {
                if (subArray[j] > maxValue) {
                    maxIndex = j;
                    maxValue = subArray[j];
                }
            }

            result.push(maxIndex);
        }

        return result;
    }

    /**
     * Get resulted tokens from decode step
     * @param {Tensor} decoder_out
     * @returns
     */
    #get_token_id(decoder_out) {
        const logits = decoder_out.logits.data;
        return this.#argmax(logits);
    }

    /**
     * Decode encoder_hidden_states into token_ids
     * @param {Tensor} encoder_hidden_states
     * @returns {Number[]}
     */
    async decode(encoder_hidden_states) {
        console.log(`Encoded in ${calc_perf(this.#temp_time)}s`);
        this.#temp_time = performance.now();

        if (!this.loaded) await this.load();
        const EOS_TOKEN_ID = 50256;
        let decoded_step = await this.#decode_step(encoder_hidden_states, {
            from_encoder: true,
        });
        let token_id = this.#get_token_id(decoded_step);
        const tokens_ids = [...token_id];

        while (tokens_ids.at(-1) !== EOS_TOKEN_ID) {
            const opt = { prev_decoder: decoded_step, input_ids: tokens_ids };
            decoded_step = await this.#decode_step(encoder_hidden_states, opt);
            token_id = this.#get_token_id(decoded_step);
            tokens_ids.push(token_id.at(-1));
        }

        return tokens_ids;
    }
}

export default Whisper;
