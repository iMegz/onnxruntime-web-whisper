import "./App.css";
import AudioRecorder from "./components/AudioRecorder";
import Mic from "./components/Mic";
// import Whipser from "./components/Whipser";

// const MODEL_PATH = "./model_int8.onnx";
// const MODEL_NAME = "openai/whisper-base.en";

// // Load onnx model
// async function loadModel(modelPath = MODEL_PATH) {
//     try {
//         const options = { executionProviders: ["wasm"] };
//         const session = await InferenceSession.create(modelPath, options);

//         log("Model loaded");
//         return session;
//     } catch (e) {
//         console.log(e);
//     }
// }

// // Tokenizer
// async function tokenize(text = "") {
//     const tokenizer = await WhisperTokenizer.from_pretrained(MODEL_NAME);
//     return await tokenizer(text);
// }

// // Feature Extractor
// async function extractFeatures(buffer = []) {
//     const config = { sampling_rate: 16000, n_fft: 2048, feature_size: 1024 };
//     const extractor = new WhisperFeatureExtractor(config);
//     const wav = await decode(buffer);
//     const data = wav.channelData;
//     return extractor(data);
// }

function App() {
    // const ref = useRef();
    // loadModel().then((result) => log(result));

    // function transcibe() {
    //     const file = ref.current.files[0];
    //     const reader = new FileReader();
    //     reader.onload = function (e) {
    //         const buffer = e.target.result;
    //         log(buffer);
    //         // extractFeatures(buffer).then((result) => {
    //         //     console.log(result);
    //         // });
    //     };

    //     // Read the file as an ArrayBuffer
    //     reader.readAsArrayBuffer(file);
    // }

    return (
        <div className="App">
            {/* <h1>Hello</h1>
            <AudioRecorder /> */}
            <Mic />
        </div>
    );
}

export default App;
