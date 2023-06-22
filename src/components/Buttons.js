function Buttons({ hRecClick, hStopClick, hTransClick, isRec }) {
    return (
        <div>
            <button onClick={hRecClick} disabled={isRec}>
                Record
            </button>
            <button onClick={hStopClick} disabled={!isRec}>
                Stop
            </button>
            <button onClick={hTransClick}>Transcribe</button>
        </div>
    );
}

export default Buttons;
