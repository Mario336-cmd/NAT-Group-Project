"use strict";

(function () {
  const panels = document.getElementById("configurationPanels");
  const feedback = document.getElementById("configurationFeedback");
  const clearAll = document.getElementById("clearAllConfiguration");
  const providerNames = { openai: "OpenAI", anthropic: "Anthropic" };
  const purposeNames = { explanation: "Explanations", extraction: "Image/PDF text extraction", transcription: "Voice input" };
  const tilePurposes = { 1: "explanation", 2: "extraction", 3: "transcription" };
  const modelDocumentation = {
    openai: "https://developers.openai.com/api/docs/models",
    anthropic: "https://platform.claude.com/docs/en/models/overview"
  };
  let csrfToken = null;
  let loading = null;
  let operation = false;
  let applicationBusy = false;
  let epoch = 0;
  const cards = new Map();

  function node(tag, className, text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text) item.textContent = text;
    return item;
  }

  function field(form, labelText, input) {
    const label = node("label", "configuration-field", labelText);
    label.htmlFor = input.id;
    form.append(label, input);
    return input;
  }

  function select(id, options) {
    const input = document.createElement("select");
    input.id = id;
    options.forEach(([value, label]) => {
      const option = node("option", "", label);
      option.value = value;
      input.appendChild(option);
    });
    return input;
  }

  function controls() {
    const disabled = operation || applicationBusy || !csrfToken;
    clearAll.disabled = disabled;
    cards.forEach(card => {
      card.form.querySelectorAll("input, select, button").forEach(input => { input.disabled = disabled; });
      card.clear.disabled = disabled || !card.configured;
      card.reveal.disabled = disabled || !card.key.value;
    });
  }

  function setKeyVisibility(card, visible) {
    card.key.type = visible ? "text" : "password";
    const label = (visible ? "Hide" : "Show") + " API key for configuration " + card.slot;
    card.reveal.setAttribute("aria-label", label);
    card.reveal.setAttribute("aria-pressed", String(visible));
    card.reveal.title = visible ? "Hide API key" : "Show API key";
    card.revealSlash.setAttribute("visibility", visible ? "visible" : "hidden");
  }

  function clearKey(card) {
    card.key.value = "";
    card.file.value = "";
    setKeyVisibility(card, false);
  }

  function scrub() {
    epoch += 1;
    cards.forEach(card => {
      clearKey(card);
      updateStatus(card);
    });
  }

  function updateStatus(card) {
    const saved = card.saved;
    const baseline = saved || { provider: card.defaultProvider, model: card.defaultModel };
    const dirty = card.provider.value !== baseline.provider ||
      card.model.value.trim() !== baseline.model || Boolean(card.key.value.trim());
    card.current.textContent = saved ? "Currently used: " + providerNames[saved.provider] + " · " +
      saved.model + " · " + purposeNames[saved.purpose] : "Not configured";
    card.status.textContent = dirty ? (saved ? "Unsaved changes. Saved settings remain in use." : "Unsaved changes.") : "";
    card.key.placeholder = saved && saved.provider === card.provider.value ?
      "Leave blank to keep the saved key" : "Paste an API key or load a .txt file";
    card.form.dataset.unsaved = String(dirty);
    card.modelDocs.href = card.purpose === "transcription" ? "https://developers.openai.com/api/docs/guides/speech-to-text" : modelDocumentation[card.provider.value];
    card.modelDocs.textContent = providerNames[card.provider.value] + " documentation";
    card.reveal.disabled = operation || applicationBusy || !csrfToken || !card.key.value;
  }

  function applyState(state, updateFields = true) {
    if (!state || typeof state.csrf_token !== "string" || !/^[A-Za-z0-9_-]{43}$/.test(state.csrf_token) ||
        !Array.isArray(state.slots) || state.slots.length !== 3) {
      throw new Error("Restart the updated backend and refresh this page to load the three API configurations.");
    }
    csrfToken = state.csrf_token;
    if (state.persistence_error) feedback.textContent = state.persistence_error;
    if (updateFields) epoch += 1;
    for (const slot of [1, 2, 3]) {
      const item = state.slots.find(item => item.slot === slot);
      if (!item || typeof item.configured !== "boolean" || typeof item.model !== "string" ||
          item.purpose !== tilePurposes[slot] || (item.configured && (!providerNames[item.provider] || (slot === 3 && item.provider !== "openai")))) {
        throw new Error("The settings response is incomplete.");
      }
      const card = cards.get(slot);
      card.configured = item.configured;
      card.saved = item.configured ? {provider: item.provider, purpose: item.purpose, model: item.model} : null;
      if (updateFields) {
        card.provider.value = item.provider || card.defaultProvider;
        card.model.value = item.model || card.defaultModel;
        clearKey(card);
      }
      updateStatus(card);
      card.form.dataset.configured = String(item.configured);
    }
    controls();
  }

  async function readResponse(response) {
    let value;
    try { value = await response.json(); }
    catch { throw new Error("The server response could not be read. Please retry."); }
    if (!response.ok) {
      throw new Error(typeof value.detail === "string" ? value.detail : "The request could not be completed.");
    }
    return value;
  }

  async function refresh() {
    if (loading) return loading;
    loading = (async () => {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 8000);
      try {
        const response = await fetch("/api/configuration", {
          credentials: "same-origin", mode: "same-origin", cache: "no-store", signal: controller.signal
        });
        const state = await readResponse(response);
        applyState(state);
        return state;
      } catch (error) {
        csrfToken = null;
        controls();
        throw error;
      } finally { window.clearTimeout(timeout); }
    })();
    try { return await loading; }
    finally { loading = null; }
  }

  async function request(url, options = {}) {
    if (typeof url !== "string" || !url.startsWith("/api/")) throw new Error("Invalid application endpoint.");
    if (!csrfToken) await refresh();
    const method = (options.method || "GET").toUpperCase();
    const headers = new Headers(options.headers || {});
    if (!["GET", "HEAD"].includes(method)) headers.set("X-CSRF-Token", csrfToken);
    const response = await fetch(url, { ...options, headers, credentials: "same-origin", mode: "same-origin", cache: "no-store" });
    if (response.status === 403) {
      csrfToken = null;
      scrub();
      try { await refresh(); }
      catch { /* The original request remains failed if the server is unavailable. */ }
    }
    return response;
  }

  async function save(card, event) {
    event.preventDefault();
    if (operation || applicationBusy) return;
    setKeyVisibility(card, false);
    operation = true;
    controls();
    feedback.textContent = "";
    card.status.textContent = "Saving configuration…";
    const supplied = {
      provider: card.provider.value,
      purpose: card.purpose,
      model: card.model.value.trim()
    };
    if (card.key.value.trim()) supplied.api_key = card.key.value.trim();
    try {
      const response = await request("/api/configuration/" + card.slot, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(supplied)
      });
      const state = await readResponse(response);
      applyState(state, false);
      epoch += 1;
      clearKey(card);
      card.model.value = supplied.model;
      updateStatus(card);
      feedback.textContent = "Configuration " + card.slot + (state.persistent ?
        " saved in Windows Credential Manager for this browser." : " saved for this session.");
      window.dispatchEvent(new Event("truecheck:configuration"));
    } catch (error) {
      card.status.textContent = "Configuration was not saved.";
      feedback.textContent = error.name === "AbortError" ? "The request timed out. Please retry." : error.message;
    } finally {
      // Remove the request's copy once it is no longer needed.
      delete supplied.api_key;
      operation = false;
      controls();
    }
  }

  async function clear(slot) {
    if (operation || applicationBusy) return;
    operation = true;
    controls();
    if (slot) {
      epoch += 1;
      clearKey(cards.get(slot));
    } else scrub();
    try {
      const response = await request("/api/configuration" + (slot ? "/" + slot : ""), { method: "DELETE" });
      const value = await readResponse(response);
      if (slot) {
        applyState(value, false);
        const card = cards.get(slot);
        card.provider.value = card.defaultProvider;
        card.model.value = card.defaultModel;
        updateStatus(card);
      }
      else {
        csrfToken = null;
        await refresh();
      }
      feedback.textContent = slot ? "Configuration " + slot + " cleared." : "All API configurations cleared.";
      window.dispatchEvent(new Event("truecheck:configuration"));
    } catch (error) { feedback.textContent = error.message; }
    finally { operation = false; controls(); }
  }

  async function loadFile(card) {
    const file = card.file.files[0];
    card.fileVersion = (card.fileVersion || 0) + 1;
    const version = card.fileVersion;
    const generation = epoch;
    const provider = card.provider.value;
    card.file.value = "";
    if (!file) return;
    setKeyVisibility(card, false);
    try {
      if (!/\.txt$/i.test(file.name) || file.size < 1 || file.size > 8192) {
        throw new Error("Choose a .txt file containing one API key, no larger than 8 KiB.");
      }
      const content = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
      const matches = [...new Set(content.match(/\bsk-[A-Za-z0-9_-]{17,509}\b/g) || [])];
      if (matches.length !== 1) throw new Error("The file must contain exactly one API key.");
      if (matches[0].startsWith("sk-ant-") !== (provider === "anthropic")) {
        throw new Error("The file's key format does not match the selected provider.");
      }
      if (generation !== epoch || card.fileVersion !== version || card.provider.value !== provider || operation || applicationBusy) return;
      card.key.value = matches[0];
      setKeyVisibility(card, false);
      updateStatus(card);
      card.status.textContent = "Key loaded locally. Choose Save configuration to apply it.";
      feedback.textContent = "";
    } catch (error) {
      if (generation !== epoch || card.fileVersion !== version) return;
      clearKey(card);
      updateStatus(card);
      feedback.textContent = error instanceof TypeError ? "The key file must use UTF-8 text." : error.message;
    }
  }

  for (const slot of [1, 2, 3]) {
    const form = node("form", "configuration-card");
    form.autocomplete = "off";
    const purpose = tilePurposes[slot];
    form.setAttribute("aria-label", purposeNames[purpose]);
    form.appendChild(node("h2", "", purposeNames[purpose]));
    const current = node("p", "configuration-current", "Checking session settings…");
    current.setAttribute("role", "status");
    const status = node("p", "configuration-status");
    status.setAttribute("role", "status");
    form.append(current, status);
    const provider = field(form, "Provider", select("provider" + slot, slot === 3 ? [["openai", "OpenAI"]] : [["openai", "OpenAI"], ["anthropic", "Anthropic"]]));
    provider.value = slot === 2 ? "anthropic" : "openai";
    const model = document.createElement("input");
    model.type = "text";
    model.id = "model" + slot;
    model.required = true;
    model.maxLength = 128;
    model.spellcheck = false;
    model.placeholder = "Enter a provider model name";
    if (slot === 3) model.value = "gpt-transcribe";
    model.setAttribute("aria-describedby", "modelHelp" + slot);
    field(form, "Model name", model);
    const modelHelp = node("p", "configuration-model-help");
    modelHelp.id = "modelHelp" + slot;
    const modelDocs = node("a");
    modelDocs.href = modelDocumentation[provider.value];
    modelDocs.textContent = providerNames[provider.value] + " documentation";
    modelDocs.target = "_blank";
    modelDocs.rel = "noopener noreferrer";
    modelHelp.append(node("span", "", "Copy the exact API model name from "), modelDocs, node("span", "", "."));
    form.appendChild(modelHelp);
    const key = document.createElement("input");
    key.type = "password";
    key.id = "apiKey" + slot;
    key.maxLength = 512;
    key.autocomplete = "off";
    key.spellcheck = false;
    field(form, "API key", key);
    const keyControl = node("div", "configuration-key-control");
    const reveal = node("button", "configuration-key-toggle");
    reveal.type = "button";
    reveal.id = "revealKey" + slot;
    reveal.setAttribute("aria-controls", key.id);
    const svgNamespace = "http://www.w3.org/2000/svg";
    const eye = document.createElementNS(svgNamespace, "svg");
    eye.setAttribute("viewBox", "0 0 24 24");
    eye.setAttribute("width", "20");
    eye.setAttribute("height", "20");
    eye.setAttribute("fill", "none");
    eye.setAttribute("stroke", "currentColor");
    eye.setAttribute("stroke-width", "1.8");
    eye.setAttribute("stroke-linecap", "round");
    eye.setAttribute("stroke-linejoin", "round");
    eye.setAttribute("aria-hidden", "true");
    const outline = document.createElementNS(svgNamespace, "path");
    outline.setAttribute("d", "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z");
    const pupil = document.createElementNS(svgNamespace, "circle");
    pupil.setAttribute("cx", "12");
    pupil.setAttribute("cy", "12");
    pupil.setAttribute("r", "3");
    const revealSlash = document.createElementNS(svgNamespace, "path");
    revealSlash.setAttribute("d", "M3 3l18 18");
    eye.append(outline, pupil, revealSlash);
    reveal.appendChild(eye);
    keyControl.append(key, reveal);
    form.appendChild(keyControl);
    const file = document.createElement("input");
    file.type = "file";
    file.accept = ".txt,text/plain";
    file.id = "keyFile" + slot;
    file.hidden = true;
    const load = node("button", "configuration-secondary", "Load key from .txt");
    load.type = "button";
    load.addEventListener("click", () => file.click());
    const actions = node("div", "configuration-actions");
    const submit = node("button", "configuration-primary", "Save configuration");
    submit.type = "submit";
    const clearButton = node("button", "configuration-secondary", "Clear");
    clearButton.type = "button";
    clearButton.addEventListener("click", () => clear(slot));
    actions.append(submit, clearButton);
    form.append(load, file, actions);
    panels.appendChild(form);
    const card = { slot, form, provider, purpose, model, modelDocs, key, file, current, status, reveal, revealSlash, clear: clearButton,
      configured: false, saved: null, defaultProvider: provider.value, defaultModel: slot === 3 ? "gpt-transcribe" : "" };
    cards.set(slot, card);
    setKeyVisibility(card, false);
    reveal.addEventListener("click", () => {
      if (!reveal.disabled) setKeyVisibility(card, key.type === "password");
    });
    form.addEventListener("submit", event => save(card, event));
    file.addEventListener("change", () => loadFile(card));
    provider.addEventListener("change", () => { epoch += 1; clearKey(card); updateStatus(card); });
    model.addEventListener("input", () => updateStatus(card));
    key.addEventListener("input", () => {
      epoch += 1;
      if (!key.value) setKeyVisibility(card, false);
      updateStatus(card);
    });
  }
  clearAll.addEventListener("click", () => clear(null));
  window.addEventListener("truecheck:busy", event => { applicationBusy = event.detail.busy; controls(); });
  window.addEventListener("pagehide", scrub);
  window.addEventListener("pageshow", event => { if (event.persisted) refresh().catch(() => {}); });
  window.TrueCheckSettings = Object.freeze({ request, refresh, scrub });
  controls();
})();
