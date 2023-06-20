// /*global BigInt */
/*global BigInt64Array */

import React, { useState, useEffect } from "react";
import { load_model } from "../utils/whisper";
import {
    AutoProcessor,
    WhisperTokenizer,
    mean_pooling,
} from "@xenova/transformers";
import { Tensor } from "onnxruntime-web";
import config from "../utils/preprocessor";

const kMaxRecordingTime = 25 * 1000; // 25 seconds in milliseconds

const AudioRecorder = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [recordedChunks, setRecordedChunks] = useState([]);
    const [recordedAudio, setRecordedAudio] = useState();
    const [modelLoaded, setModelLoaded] = useState(false);
    const [session, setSession] = useState();

    useEffect(() => {
        load_model().then((sess) => {
            setModelLoaded(true);
            setSession(sess);
        });
    }, []);

    let recorder;

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: true,
            });
            recorder = new MediaRecorder(stream);

            const chunks = [];
            recorder.addEventListener("dataavailable", (event) => {
                if (event.data.size > 0) {
                    chunks.push(event.data);
                }
            });

            recorder.addEventListener("stop", () => {
                const recordedAudio = new Blob(chunks, {
                    type: "audio/wav",
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

    const handleTranscribeClick = () => {
        recordedAudio.arrayBuffer().then((buffer) => {
            const processor = AutoProcessor.from_pretrained(null, {
                config,
            });

            processor.then((extractor) => {
                extractor(buffer).then(({ input_features }) => {
                    const decoder_input_ids = new Tensor(
                        new BigInt64Array([50257n, 50257n]),
                        [1, 2]
                    );
                    const mask = Int32Array.from(
                        { length: 1 * 80 * 3000 },
                        () => 0
                    );
                    const attention_mask = new Tensor(
                        new Int32Array(mask),
                        [1, 80, 3000]
                    );
                    const feeds = {
                        input_features,
                        decoder_input_ids,
                    };
                    console.log(session);
                    // session.run(feeds).then((result) => {
                    //     const last_hidden_state = result.last_hidden_state;

                    //     const res = mean_pooling(
                    //         last_hidden_state,
                    //         attention_mask
                    //     );
                    //     console.log(res);
                    // });

                    // WhisperTokenizer.from_pretrained("whisper-finetuned").then(
                    //     (result) => {
                    //         console.log(Object.keys(result.model.config));
                    //     }
                    // );

                    // console.log(feeds);
                    // const promises = [
                    //     WhisperTokenizer.from_pretrained("whisper-finetuned"),
                    //     session.run(feeds),
                    // ];
                    // Promise.all(promises).then(([tokenizer, result])=>{
                    //     tokenizer(result.last_hidden_state)
                    // })

                    //output = result.last_hidden_state
                });
            });
        });

        // recordedChunks[0].arrayBuffer().then((result) => console.log(result));
        // feature_extraction(recordedChunks).then((result) => {
        //     console.log(result);
        // });
    };
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
    return <div>{displayButtons()}</div>;
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
