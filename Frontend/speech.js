"use strict";

(function () {
  const mic = document.getElementById("micBtn");
  const panel = document.getElementById("voicePanel");
  const status = document.getElementById("voiceStatus");
  const timer = document.getElementById("voiceTimer");
  const stop = document.getElementById("stopRecordingBtn");
  const cancel = document.getElementById("cancelRecordingBtn");
  const canvas = document.getElementById("voiceWaveform");
  const voiceSelect = document.getElementById("voiceSelect");
  const voiceFeedback = document.getElementById("voiceSelectionStatus");
  const preview = document.getElementById("previewVoiceBtn");
  const stopPreview = document.getElementById("stopPreviewBtn");
  document.getElementById("configurationPanels").appendChild(document.querySelector(".voice-settings"));
  const synth = window.speechSynthesis;
  const canSpeak = Boolean(synth && window.SpeechSynthesisUtterance);
  const preferenceKey = "truecheck-read-aloud-voice";
  let hooks;
  let allowed = false;
  let configured = false;
  let generation = 0;
  let recording = null;
  let speaker = null;
  let voices = [];
  let selectedVoice = "";
  try { selectedVoice = localStorage.getItem(preferenceKey) || ""; } catch { /* Optional preference. */ }

  function voiceKey(voice) {
    return JSON.stringify([voice.voiceURI, voice.name, voice.lang]);
  }

  function readVoice() {
    return voices.find(voice => voiceKey(voice) === selectedVoice) ||
      voices.find(voice => /^en(?:-|$)/i.test(voice.lang) && voice.default) ||
      voices.find(voice => /^en(?:-|$)/i.test(voice.lang)) || voices.find(voice => voice.default) || voices[0];
  }

  function refreshVoices() {
    voices = canSpeak ? synth.getVoices() : [];
    voiceSelect.replaceChildren(new Option("Default English voice", ""));
    voices.forEach(voice => voiceSelect.add(new Option(voice.name + " · " + voice.lang, voiceKey(voice))));
    const savedAvailable = voices.some(voice => voiceKey(voice) === selectedVoice);
    voiceSelect.value = savedAvailable ? selectedVoice : "";
    voiceSelect.disabled = !canSpeak;
    preview.disabled = !canSpeak || Boolean(recording);
    voiceFeedback.textContent = !canSpeak ? "Read aloud is not supported by this browser." :
      selectedVoice && !savedAvailable ? "The saved voice is unavailable. The default voice will be used." :
      "Voice selection applies to the next playback.";
  }

  function stopSpeaking() {
    const previous = speaker;
    speaker = null;
    if (previous) {
      previous.onState("idle");
      synth.cancel();
    }
    stopPreview.disabled = true;
  }

  function speak(text, onState, onError) {
    stopSpeaking();
    if (!canSpeak || recording || !text.trim()) return;
    const current = { chunks: text.match(/[\s\S]{1,240}(?:\s|$)|[\s\S]{1,240}/g) || [], index: 0, onState };
    speaker = current;
    function next() {
      if (speaker !== current) return;
      if (current.index === current.chunks.length) { stopSpeaking(); return; }
      const utterance = new SpeechSynthesisUtterance(current.chunks[current.index++].trim());
      current.utterance = utterance;
      const voice = readVoice();
      if (voice) { utterance.voice = voice; utterance.lang = voice.lang; }
      else utterance.lang = "en";
      utterance.onstart = () => { if (speaker === current) onState("playing"); };
      utterance.onpause = () => { if (speaker === current) onState("paused"); };
      utterance.onresume = () => { if (speaker === current) onState("playing"); };
      utterance.onend = next;
      utterance.onerror = event => {
        if (speaker !== current) return;
        stopSpeaking();
        if (!["canceled", "interrupted"].includes(event.error)) onError("Read aloud could not play. Choose another voice in Settings and retry.");
      };
      onState("loading");
      try { synth.speak(utterance); }
      catch { stopSpeaking(); onError("Read aloud could not start. Choose another voice and retry."); }
    }
    next();
  }

  function bindRead(button, text, feedback) {
    let state = "idle";
    const stopButton = document.createElement("button");
    stopButton.type = "button";
    stopButton.textContent = "Stop reading";
    stopButton.hidden = true;
    button.after(stopButton);
    button.disabled = !canSpeak;
    button.title = canSpeak ? "Read the prediction and explanation aloud" : "Read aloud is not supported by this browser";
    button.setAttribute("aria-label", button.title);
    function setState(value) {
      state = value;
      button.textContent = { idle: "Read aloud", loading: "Preparing voice…", playing: "Pause reading", paused: "Resume reading" }[value];
      button.disabled = !canSpeak || value === "loading";
      button.setAttribute("aria-label", button.textContent);
      stopButton.hidden = value === "idle";
    }
    button.addEventListener("click", () => {
      if (recording) { feedback.textContent = "Finish or cancel the recording before playing audio."; return; }
      feedback.textContent = "";
      if (state === "playing") { synth.pause(); setState("paused"); }
      else if (state === "paused") { synth.resume(); setState("playing"); }
      else speak(text, setState, message => { feedback.textContent = message; });
    });
    stopButton.addEventListener("click", stopSpeaking);
  }

  function releaseCapture(current) {
    clearInterval(current.interval);
    clearTimeout(current.limitTimer);
    cancelAnimationFrame(current.frame);
    if (current.stream) current.stream.getTracks().forEach(track => track.stop());
    if (current.audioContext && current.audioContext.state !== "closed") current.audioContext.close().catch(() => {});
    current.stream = null;
    current.audioContext = null;
  }

  function cancelRecording(message = "") {
    generation += 1;
    const previous = recording;
    recording = null;
    if (previous) {
      previous.controller?.abort();
      if (previous.recorder) {
        previous.recorder.onstop = null;
        previous.recorder.ondataavailable = null;
        previous.recorder.onerror = null;
        if (previous.recorder.state !== "inactive") { try { previous.recorder.stop(); } catch { /* Already stopped. */ } }
      }
      releaseCapture(previous);
      previous.chunks.length = 0;
      hooks?.onBusy(false);
    }
    panel.hidden = true;
    mic.setAttribute("aria-pressed", "false");
    update(allowed, configured);
    preview.disabled = !canSpeak;
    if (message) hooks?.onFeedback(message);
  }

  async function transcribe(current) {
    if (recording !== current) return;
    status.textContent = "Transcribing recording…";
    stop.hidden = true;
    canvas.hidden = true;
    if (document.activeElement === stop || document.activeElement === mic) cancel.focus();
    const blob = new Blob(current.chunks, { type: current.recorder.mimeType });
    current.chunks.length = 0;
    if (!blob.size) { cancelRecording("The recording is empty. Try recording again."); return; }
    if (blob.size > 5 * 1024 * 1024) { cancelRecording("The recording is too large. Record a shorter claim."); return; }
    current.controller = new AbortController();
    const timeout = setTimeout(() => current.controller.abort(), 70000);
    try {
      const form = new FormData();
      form.append("audio", blob, blob.type.includes("mp4") ? "recording.m4a" : "recording.webm");
      const response = await window.TrueCheckSettings.request("/api/transcriptions", {
        method: "POST", body: form, signal: current.controller.signal
      });
      const result = await response.json();
      if (recording !== current) return;
      if (response.status === 422 && result.detail === "No speech was detected. Record your claim again.") {
        cancelRecording("No speech detected. Try recording again or type your claim.");
        return;
      }
      if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Transcription failed. Try again or type the claim.");
      if (typeof result.text !== "string" || !result.text.trim() || result.text.length > 4000) throw new Error("No usable transcript was returned. Record a shorter claim.");
      const text = result.text.trim();
      cancelRecording();
      hooks.onResult(text);
    } catch (error) {
      if (recording !== current) return;
      cancelRecording(error.name === "AbortError" ? "Transcription timed out. Your draft was kept." :
        error instanceof TypeError ? "The server could not be reached. Your draft was kept." : error.message);
    } finally { clearTimeout(timeout); }
  }

  function finishRecording() {
    const current = recording;
    if (!current || !current.recorder || current.recorder.state !== "recording") return;
    stop.disabled = true;
    status.textContent = "Preparing recording…";
    try { current.recorder.stop(); }
    catch { cancelRecording("The recording could not finish. Try again."); return; }
    releaseCapture(current);
  }

  function drawWaveform(current) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    try {
      const audioContext = new AudioContext();
      current.audioContext = audioContext;
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 128;
      audioContext.createMediaStreamSource(current.stream).connect(analyser);
      audioContext.resume().catch(() => {});
      const values = new Uint8Array(analyser.frequencyBinCount);
      const context = canvas.getContext("2d");
      if (!context) return;
      function draw() {
        if (recording !== current || !current.stream) return;
        analyser.getByteTimeDomainData(values);
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.strokeStyle = "#1d4ed8";
        context.lineWidth = 2;
        context.beginPath();
        values.forEach((value, index) => {
          const x = index * canvas.width / (values.length - 1);
          const y = value / 255 * canvas.height;
          if (index) context.lineTo(x, y); else context.moveTo(x, y);
        });
        context.stroke();
        current.frame = requestAnimationFrame(draw);
      }
      draw();
    } catch { /* Recording still works when waveform analysis is unavailable. */ }
  }

  async function startRecording() {
    if (!allowed || !configured || !hooks?.canRecord()) return;
    if (!window.isSecureContext) { hooks.onFeedback("Microphone recording requires localhost or HTTPS."); return; }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { hooks.onFeedback("Voice recording is not supported by this browser."); return; }
    stopSpeaking();
    const current = { id: ++generation, chunks: [], bytes: 0 };
    recording = current;
    hooks.onBusy(true);
    hooks.onFeedback("");
    panel.hidden = false;
    status.textContent = "Waiting for microphone permission…";
    timer.textContent = "";
    canvas.hidden = true;
    stop.hidden = true;
    preview.disabled = true;
    mic.disabled = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      if (recording !== current || generation !== current.id) { stream.getTracks().forEach(track => track.stop()); return; }
      current.stream = stream;
      const mimeType = ["audio/webm;codecs=opus", "audio/mp4", "audio/webm"].find(type => MediaRecorder.isTypeSupported(type));
      if (!mimeType) throw new Error("format");
      const recorder = new MediaRecorder(stream, { mimeType });
      current.recorder = recorder;
      recorder.ondataavailable = event => {
        if (recording !== current || !event.data.size) return;
        current.bytes += event.data.size;
        if (current.bytes > 5 * 1024 * 1024) { cancelRecording("The recording is too large. Record a shorter claim."); return; }
        current.chunks.push(event.data);
      };
      recorder.onstop = () => { releaseCapture(current); void transcribe(current); };
      recorder.onerror = () => { if (recording === current) cancelRecording("The microphone recording failed. Your draft was kept."); };
      stream.getAudioTracks().forEach(track => track.addEventListener("ended", () => {
        if (recording === current && current.stream) finishRecording();
      }));
      recorder.start(250);
      current.started = performance.now();
      function tick() {
        const seconds = Math.min(30, Math.floor((performance.now() - current.started) / 1000));
        timer.textContent = "0:" + String(seconds).padStart(2, "0") + " / 0:30";
      }
      tick();
      current.interval = setInterval(tick, 250);
      current.limitTimer = setTimeout(finishRecording, 30000);
      status.textContent = "Recording claim";
      stop.hidden = false;
      stop.disabled = false;
      canvas.hidden = false;
      mic.disabled = false;
      mic.setAttribute("aria-pressed", "true");
      mic.setAttribute("aria-label", "Stop recording");
      mic.title = "Stop recording";
      drawWaveform(current);
    } catch (error) {
      if (recording !== current) return;
      const message = error.name === "NotAllowedError" ? "Microphone permission was denied. Allow access in your browser and retry." :
        error.name === "NotFoundError" ? "No microphone was found. Connect one and retry." :
        error.message === "format" ? "This browser cannot record a supported audio format." : "The microphone could not start. Check that it is available and retry.";
      cancelRecording(message);
    }
  }

  function update(ready, available) {
    allowed = ready;
    configured = available;
    if (recording) return;
    const supported = Boolean(window.isSecureContext && navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
    mic.disabled = !allowed || !configured || !supported;
    mic.title = !supported ? "Microphone recording requires a supported browser on localhost or HTTPS" :
      !configured ? "Configure Voice input in Settings" : !allowed ? "Voice input is unavailable while another operation is running" : "Record a claim";
    mic.setAttribute("aria-label", mic.title);
  }

  mic.addEventListener("click", () => recording ? finishRecording() : void startRecording());
  stop.addEventListener("click", finishRecording);
  cancel.addEventListener("click", () => cancelRecording("Recording cancelled. Your draft was kept."));
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && recording) { event.preventDefault(); cancelRecording("Recording cancelled. Your draft was kept."); }
  });
  voiceSelect.addEventListener("change", () => {
    stopSpeaking();
    selectedVoice = voiceSelect.value;
    try { localStorage.setItem(preferenceKey, selectedVoice); voiceFeedback.textContent = "Voice preference saved."; }
    catch { voiceFeedback.textContent = "Voice selected for this page. Browser storage is unavailable."; }
  });
  preview.addEventListener("click", () => speak("Welcome to TrueCheck. Enter a claim to fact check.", state => {
    preview.disabled = state !== "idle";
    stopPreview.disabled = state === "idle";
  }, message => { voiceFeedback.textContent = message; }));
  stopPreview.addEventListener("click", stopSpeaking);
  if (canSpeak) synth.addEventListener("voiceschanged", refreshVoices);
  window.addEventListener("pagehide", () => { cancelRecording(); stopSpeaking(); });
  window.addEventListener("storage", event => {
    if (event.key === preferenceKey || event.key === null) { selectedVoice = event.newValue || ""; refreshVoices(); }
  });
  refreshVoices();
  window.TrueCheckSpeech = Object.freeze({
    init(callbacks) { hooks = callbacks; }, update, bindRead, stopSpeaking,
    cancelRecording, reset() { cancelRecording(); stopSpeaking(); }
  });
})();
