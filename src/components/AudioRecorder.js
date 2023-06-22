/*global BigInt */
/*global BigInt64Array */

import React, { useState, useEffect, useRef } from "react";
import Whisper from "../utils/whipser";
import { InferenceSession, Tensor } from "onnxruntime-web";
import { encode } from "node-wav";
const { WaveFile } = require("wavefile");

const kSampleRate = 16000;
const kMaxRecordingTime = 25 * 1000; // 25 seconds in milliseconds
let sampling_rate = 48000;

const context = new AudioContext({
    sampleRate: kSampleRate,
    channelCount: 1,
    echoCancellation: false,
    autoGainControl: true,
    noiseSuppression: true,
});

// const mask = Int32Array.from(
//     { length: 1 * 80 * 3000 },
//     () => 0
// );
// const attention_mask = new Tensor(
//     new Int32Array(mask),
//     [1, 80, 3000]
// );

// function argMax(array, n) {
//     const result = [];

//     for (let i = 0; i < array.length; i += n) {
//         const subArray = array.slice(i, i + n);
//         let maxIndex = 0;
//         let maxValue = subArray[0];

//         for (let j = 1; j < subArray.length; j++) {
//             if (subArray[j] > maxValue) {
//                 maxIndex = j;
//                 maxValue = subArray[j];
//             }
//         }

//         result.push(maxIndex);
//     }

//     return result;
// }

const AudioRecorder = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [recordedChunks, setRecordedChunks] = useState([]);
    const [recordedAudio, setRecordedAudio] = useState();

    const [modelLoaded, setModelLoaded] = useState(true);
    const [model, setModel] = useState();

    useEffect(() => {
        const config = {
            decoder_path: "./decoder_model_merged_quantized.onnx",
            encoder_path: "./encoder_model_quantized.onnx",
            tokenizer_config: "whisper-finetuned",
        };

        // const whisper = new Whisper(config);
        // whisper.load().then((result) => {
        //     console.log(result);
        //     setModel(whisper);
        //     setModelLoaded(true);
        // });
    }, []);

    let recorder;

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 44100 },
            });
            recorder = new MediaRecorder(stream);
            sampling_rate = stream.getAudioTracks()[0].getSettings().sampleRate;

            const chunks = [];
            recorder.addEventListener("dataavailable", (event) => {
                const data = event.data;
                console.log(data);

                if (data.size > 0) {
                    chunks.push(data);
                }
            });

            recorder.addEventListener("stop", () => {
                const recordedAudio = new Blob(chunks, {
                    type: "audio/wav; codecs=opu",
                    // type: "audio/ogg; codecs=opu",
                });
                setRecordedChunks(chunks);
                setIsRecording(false);
                setRecordedAudio(recordedAudio);
            });

            recorder.start();
            setMediaRecorder(recorder);
            setIsRecording(true);
            setTimeout(stopRecording, kMaxRecordingTime);
        } catch (error) {
            console.error("Error accessing media devices:", error);
        }
    };

    const stopRecording = () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
    };

    const handleRecordClick = () => {
        if (!isRecording) {
            setRecordedChunks([]);
            startRecording();
        }
    };

    const handleStopClick = () => {
        if (isRecording) {
            stopRecording();
        }
    };

    function calc_perf(start) {
        const time = ~~(((performance.now() - start) / 1000) * 100) / 100;
        return time;
    }

    const handleTranscribeClick = () => {
        const start = performance.now();
        recordedAudio.arrayBuffer().then((buffer) => {
            context.decodeAudioData(buffer).then((audioBuffer) => {
                const offlineContext = new OfflineAudioContext(
                    audioBuffer.numberOfChannels,
                    audioBuffer.length,
                    audioBuffer.sampleRate
                );
                const source = offlineContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(offlineContext.destination);
                source.start();
                const renderedBuffer = offlineContext
                    .startRendering()
                    .then((renderedBuffer) => {
                        const audio = renderedBuffer.getChannelData(0);
                        console.log(`Audio processed in ${calc_perf(start)}s`);
                        console.log(audio);
                    });
            });
            // Function to convert Blob to ArrayBuffer
            // const uint8Array = new Uint8Array(buffer);
            // const wav = new WaveFile();
            // wav.fromBuffer(uint8Array);
            // const audioData = new Float32Array(wav.getSamples(true));
            // model.feature_extractor(audioData).then(({ input_features }) => {
            //     model.encode(input_features).then(({ last_hidden_state }) => {
            //         model.decode(last_hidden_state).then((tokens) => {
            //             model.decode_tokens(tokens).then((result) => {
            //                 console.log(result);
            //             });
            //         });
            //     });
            // });
            //     const processor = AutoProcessor.from_pretrained(null, {
            //         config,
            //     });
            //     processor.then((extractor) => {
            //         extractor(buffer).then(({ input_features }) => {
            //             const input_ids = new Tensor(
            //                 new BigInt64Array([1n, 50362n]),
            //                 [1, 2]
            //             );
            //             encoder
            //                 .run({ input_features })
            //                 .then(({ last_hidden_state }) => {
            //                     const encoder_hidden_states = last_hidden_state;
            //                     const feed = { encoder_hidden_states, input_ids };
            //                     decoder.run(feed).then(({ logits }) => {
            //                         const tokens = argMax(logits.data, 51864);
            //                         console.log(tokens);
            //                         WhisperTokenizer.from_pretrained(
            //                             "whisper-finetuned"
            //                         ).then((result) => {
            //                             // console.log(result.decode(tokens));
            //                             console.log(result.encode("test"));
            //                         });
            //                     });
            //                 });
            //         });
            //     });
        });

        // recordedChunks[0].arrayBuffer().then((result) => console.log(result));
        // feature_extraction(recordedChunks).then((result) => {
        //     console.log(result);
        // });
    };
    const ref = useRef();
    function displayButtons() {
        if (modelLoaded) {
            return (
                <>
                    <button onClick={handleRecordClick} disabled={isRecording}>
                        Record
                    </button>
                    <button onClick={handleStopClick} disabled={!isRecording}>
                        Stop
                    </button>
                    <button
                        onClick={handleTranscribeClick}
                        disabled={recordedChunks.length === 0}
                    >
                        Transcribe
                    </button>
                </>
            );
        } else return <h1>Loading Model...</h1>;
    }
    function handle(e) {
        e.target.files[0].arrayBuffer().then((buffer) => {
            // Assuming `audioData` is the captured audio data as a Uint8Array

            const uint8Array = new Uint8Array(buffer);
            console.log(buffer);
            const wav = new WaveFile();
            wav.fromBuffer(uint8Array);
            const audioData = new Float32Array(wav.getSamples(true));
            console.log(audioData);
            // model.feature_extractor(audioData).then(({ input_features }) => {
            //     model.encode(input_features).then(({ last_hidden_state }) => {
            //         model.decode(last_hidden_state).then((tokens) => {
            //             model.decode_tokens(tokens).then((result) => {
            //                 console.log(result);
            //             });
            //         });
            //     });
            // });
        });
    }
    return (
        <div>
            <input type="file" ref={ref} onInput={handle} />
            {displayButtons()}
        </div>
    );
};

export default AudioRecorder;

// useEffect(() => {
//     return () => {
//         if (recorder) {
//             recorder.removeEventListener("dataavailable");
//             recorder.removeEventListener("stop");
//         }
//     };
// }, [mediaRecorder]);

// session.run(feeds).then((result) => {
//     // const last_hidden_state = result.last_hidden_state;

//     // const res = mean_pooling(
//     //     last_hidden_state,
//     //     attention_mask
//     // );

//     console.log(result);
//     console.log(test);
// });

// console.log(feeds);
// const promises = [
//     WhisperTokenizer.from_pretrained("whisper-finetuned"),
//     session.run(feeds),
// ];
// Promise.all(promises).then(([tokenizer, result])=>{
//     tokenizer(result.last_hidden_state)
// })

//output = result.last_hidden_state

/*
Decoder only
const decoder_bas = "decoder_model_quantized.onnx";
                    InferenceSession.create(decoder_bas).then((sess) => {
                        const feed2 = {
                            encoder_hidden_states: last_hidden_state,
                            input_ids: new Tensor(
                                new BigInt64Array([1n, 50362n]),
                                [1, 2]
                            ),
                        };
                        sess.run(feed2).then((res) => {
                            const max = model.get_token_id(res);

                            model
                                .decode_tokens(max)
                                .then((text) => console.log({ max, text }));
                            // model.decode_tokens([50257, 632]).then((result) => {
                            //     console.log(result);
                            // });
                        });
                    });
*/
