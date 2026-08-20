/**
 * LiveTranscriber — drop-in mic client for the live speech-to-text API.
 *
 * Usage:
 *   const t = new LiveTranscriber({
 *     sessionUrl: "http://localhost:8000/session",
 *     wsUrl: "ws://localhost:8000/ws/stream",
 *   });
 *   t.addEventListener("status", e => console.log(e.detail.status));   // "connecting" | "listening" | "idle"
 *   t.addEventListener("level", e => console.log(e.detail.rms));        // 0..1, for a waveform/level meter
 *   t.addEventListener("rawsegment", e => console.log(e.detail));       // {text, low_confidence, low_energy, ...}
 *   t.addEventListener("corrected", e => console.log(e.detail.text));   // the short, cleaned transcript
 *   await t.start();   // asks for mic permission and begins streaming
 *   t.stop();          // stops the mic and closes the session
 *
 * No build step / dependencies. Works as a plain <script> include or an ES module.
 * Handles the browser-audio mismatch for you: mic audio normally comes in at
 * 44.1/48kHz float32, but the server expects 16kHz mono int16 PCM — this
 * resamples in an AudioWorklet before anything is sent.
 */

const RESAMPLER_WORKLET_CODE = `
class ResamplerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetSampleRate = (options.processorOptions && options.processorOptions.targetSampleRate) || 16000;
    this.ratio = sampleRate / this.targetSampleRate; // 'sampleRate' is a global in AudioWorkletGlobalScope
    this.pending = new Float32Array(0);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;
    const channelData = input[0]; // mono (first channel)

    const combined = new Float32Array(this.pending.length + channelData.length);
    combined.set(this.pending, 0);
    combined.set(channelData, this.pending.length);

    const outLength = Math.max(0, Math.floor(combined.length / this.ratio));
    const out = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const srcIndex = i * this.ratio;
      const i0 = Math.floor(srcIndex);
      const i1 = Math.min(i0 + 1, combined.length - 1);
      const frac = srcIndex - i0;
      const sample = combined[i0] * (1 - frac) + combined[i1] * frac;
      out[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32768)));
    }

    const consumedSrcLength = Math.floor(outLength * this.ratio);
    this.pending = combined.slice(consumedSrcLength);

    if (out.length > 0) {
      this.port.postMessage(out.buffer, [out.buffer]);
    }
    return true;
  }
}
registerProcessor('resampler-processor', ResamplerProcessor);
`;

class LiveTranscriber extends EventTarget {
  constructor({ sessionUrl, wsUrl, targetSampleRate = 16000, sendBufferMs = 200 }) {
    super();
    this.sessionUrl = sessionUrl;
    this.wsUrl = wsUrl;
    this.targetSampleRate = targetSampleRate;
    this.sendBufferMs = sendBufferMs;

    this._ws = null;
    this._audioCtx = null;
    this._workletNode = null;
    this._stream = null;
    this._sendBuffer = [];
    this._sendBufferSamples = 0;
    this.sessionId = null;
    this.status = "idle";
  }

  async start() {
    if (this.status !== "idle") return;
    this._setStatus("connecting");

    // 1. create a session
    const resp = await fetch(this.sessionUrl, { method: "POST" });
    if (!resp.ok) throw new Error(`session create failed: ${resp.status}`);
    const data = await resp.json();
    this.sessionId = data.session_id;

    // 2. open the websocket before we start capturing audio
    this._ws = new WebSocket(`${this.wsUrl}/${this.sessionId}`);
    this._ws.binaryType = "arraybuffer";
    await new Promise((resolve, reject) => {
      this._ws.onopen = () => resolve();
      this._ws.onerror = (e) => reject(e);
    });
    this._ws.onmessage = (evt) => this._handleMessage(evt);
    this._ws.onclose = () => this._setStatus("idle");

    // 3. mic capture + resample pipeline
    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    this._audioCtx = new AudioContext();
    const workletUrl = URL.createObjectURL(
      new Blob([RESAMPLER_WORKLET_CODE], { type: "application/javascript" })
    );
    await this._audioCtx.audioWorklet.addModule(workletUrl);

    const source = this._audioCtx.createMediaStreamSource(this._stream);
    this._workletNode = new AudioWorkletNode(this._audioCtx, "resampler-processor", {
      processorOptions: { targetSampleRate: this.targetSampleRate },
    });
    this._workletNode.port.onmessage = (evt) => this._onAudioChunk(evt.data);
    source.connect(this._workletNode);
    // intentionally NOT connected to audioCtx.destination — we don't want to play the mic back

    this._setStatus("listening");
    this.dispatchEvent(new CustomEvent("sessionstart", { detail: { sessionId: this.sessionId } }));
  }

  stop() {
    this._flushSendBuffer();
    if (this._workletNode) this._workletNode.disconnect();
    if (this._stream) this._stream.getTracks().forEach((t) => t.stop());
    if (this._audioCtx) this._audioCtx.close();
    if (this._ws && this._ws.readyState === WebSocket.OPEN) this._ws.close();
    this._setStatus("idle");
  }

  _onAudioChunk(int16Buffer) {
    const int16 = new Int16Array(int16Buffer);

    // simple RMS level, handy for a waveform / mic-level indicator
    let sumSq = 0;
    for (let i = 0; i < int16.length; i++) sumSq += int16[i] * int16[i];
    const rms = int16.length ? Math.sqrt(sumSq / int16.length) / 32768 : 0;
    this.dispatchEvent(new CustomEvent("level", { detail: { rms } }));

    this._sendBuffer.push(int16);
    this._sendBufferSamples += int16.length;

    const bufferTargetSamples = (this.targetSampleRate * this.sendBufferMs) / 1000;
    if (this._sendBufferSamples >= bufferTargetSamples) {
      this._flushSendBuffer();
    }
  }

  _flushSendBuffer() {
    if (this._sendBuffer.length === 0) return;
    const merged = new Int16Array(this._sendBufferSamples);
    let offset = 0;
    for (const chunk of this._sendBuffer) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    this._sendBuffer = [];
    this._sendBufferSamples = 0;
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(merged.buffer);
    }
  }

  _handleMessage(evt) {
    let msg;
    try {
      msg = JSON.parse(evt.data);
    } catch {
      return;
    }
    if (msg.type === "raw_segment") {
      this.dispatchEvent(new CustomEvent("rawsegment", { detail: msg }));
    } else if (msg.type === "corrected") {
      this.dispatchEvent(new CustomEvent("corrected", { detail: msg }));
    }
  }

  _setStatus(status) {
    this.status = status;
    this.dispatchEvent(new CustomEvent("status", { detail: { status } }));
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { LiveTranscriber };
}
