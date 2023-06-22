import { useEffect, useState } from "react";
import Buttons from "./Buttons";
import Whisper from "../utils/whipser";

const OUTPUT_SAMPLE_RATE = 16000;
const INPUT_SAMPLE_RATE = 44100;

function Mic() {
    const [isRec, setIsRec] = useState(false);
    const [record, setRecord] = useState();
    const [recorder, setRecorder] = useState(null);
    const [model, setModel] = useState();
    const [modelLoaded, setModelLoaded] = useState(false);

    const context = new AudioContext({ sampleRate: INPUT_SAMPLE_RATE });

    useEffect(() => {
        const config = {
            decoder_path: "./decoder_model_merged_quantized.onnx",
            encoder_path: "./encoder_model_quantized.onnx",
            tokenizer_config: "whisper-finetuned",
        };
        const whisper = new Whisper(config);
        whisper.load().then((result) => {
            console.log(result);
            setModel(whisper);
            setModelLoaded(true);
        });
    }, []);

    async function startRecording() {
        try {
            const opt = {
                audio: { sampleRate: INPUT_SAMPLE_RATE, channelCount: 1 },
            };
            const stream = await navigator.mediaDevices.getUserMedia(opt);
            await context.resume();

            const mediaRecorder = new MediaRecorder(stream);
            const chunks = [];

            mediaRecorder.addEventListener("dataavailable", ({ data }) => {
                if (data.size > 0) chunks.push(data);
            });

            mediaRecorder.addEventListener("stop", async () => {
                const opt = { type: "audio/wav" };
                const recordedAudio = new Blob(chunks, opt);
                const buffer = await recordedAudio.arrayBuffer();
                const audioBuffer = await context.decodeAudioData(buffer);

                const old_sr = audioBuffer.sampleRate;
                const old_length = audioBuffer.length;

                const new_length = (old_length * OUTPUT_SAMPLE_RATE) / old_sr;

                const offlineContext = new OfflineAudioContext(
                    1,
                    new_length,
                    OUTPUT_SAMPLE_RATE
                );
                const source = offlineContext.createBufferSource();
                source.buffer = audioBuffer;

                offlineContext.oncomplete = ({ renderedBuffer }) => {
                    const audioData = renderedBuffer.getChannelData(0);
                    model
                        .feature_extractor(audioData)
                        .then(({ input_features }) => {
                            model
                                .encode(input_features)
                                .then(({ last_hidden_state }) => {
                                    model
                                        .decode(last_hidden_state)
                                        .then((tokens) => {
                                            model
                                                .decode_tokens(tokens)
                                                .then((result) => {
                                                    console.log(result);
                                                });
                                        });
                                });
                        });
                };

                source.connect(offlineContext.destination);
                source.start(0);
                offlineContext.startRendering();

                setIsRec(false);
                setRecord(recordedAudio);
            });

            mediaRecorder.start();
            setRecorder(mediaRecorder);
            setIsRec(true);
        } catch (error) {
            console.log("Media device access failed", error);
        }
    }

    function stopRecording() {
        if (recorder && recorder.state === "recording") {
            recorder.stop();
        }
    }

    // Btn click handlers
    function hRecClick() {
        startRecording();
    }
    function hStopClick() {
        stopRecording();
    }
    function hTransClick() {
        console.log(record);
    }
    return (
        <div>
            <h1>Microphone Testing</h1>
            <Buttons
                hRecClick={hRecClick}
                hStopClick={hStopClick}
                hTransClick={hTransClick}
                isRec={isRec}
            />
        </div>
    );
}

export default Mic;
