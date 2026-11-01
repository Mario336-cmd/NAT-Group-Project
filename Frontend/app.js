"use strict";

const input = document.getElementById("claimInput");
const sendBtn = document.getElementById("sendBtn");
const topKSelect = document.getElementById("topKSelect");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const chatScroll = document.getElementById("chatScroll");
const chatList = document.getElementById("chatList");
const sidebar = document.getElementById("sidebar");
const mobileMenu = document.getElementById("mobileMenu");
const sidebarToggle = document.getElementById("sidebarToggle");
const sidebarExpandBtn = document.getElementById("sidebarExpandBtn");
const sidebarBackdrop = document.getElementById("sidebarBackdrop");
const inputFeedback = document.getElementById("inputFeedback");
const serverState = document.getElementById("serverState");
const serverStatus = document.getElementById("serverStatus");
const retryBtn = document.getElementById("retryBtn");
const modelProgress = document.getElementById("modelProgress");
const composerWrap = document.querySelector(".composer-wrap");
const attachBtn = document.getElementById("attachBtn");
const fileInput = document.getElementById("fileInput");
const extractionDialog = document.getElementById("extractionDialog");
const extractionFilename = document.getElementById("extractionFilename");
const extractionProgress = document.getElementById("extractionProgress");
const extractionReview = document.getElementById("extractionReview");
const extractionText = document.getElementById("extractionText");
const extractionFeedback = document.getElementById("extractionFeedback");
const useClaimBtn = document.getElementById("useClaimBtn");
const historyList = document.getElementById("historyList");
const historyEmpty = document.getElementById("historyEmpty");
const historyCount = document.getElementById("historyCount");
const historyFeedback = document.getElementById("historyFeedback");
const historyStorageNotice = document.getElementById("historyStorageNotice");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const deleteHistoryDialog = document.getElementById("deleteHistoryDialog");
const cancelDeleteHistoryBtn = document.getElementById("cancelDeleteHistoryBtn");
const confirmDeleteHistoryBtn = document.getElementById("confirmDeleteHistoryBtn");

const historyStore = window.TrueCheckHistory.createStore({ validateResult });
const transientRecords = new Map();
let activeChat = historyStore.selectedChat() || historyStore.createChat();
const expandedChats = new Set([activeChat.id]);
let currentView = "chat";
let openedRecordId = null;
let pendingDeletion = null;
let deletionFocus = null;
let historySaveError = "";
let serverReady = false;
let extractionAvailable = false;
let extractionProviderName = "your configured provider";
let transcriptionAvailable = false;
let voiceBusy = false;
let checkingServer = false;
let pendingServerCheck = false;
let healthPollTimer = null;
let pageActive = true;
let retryingSetup = false;
let busy = false;
let busyKind = null;
let conversation = 0;
let activeRequest = null;
let scrollRevision = 0;
let sidebarCollapsed = false;
try { sidebarCollapsed = localStorage.getItem("truecheck-sidebar-collapsed") === "true"; }
catch { /* The sidebar still works without preference storage. */ }

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
  sendBtn.disabled = busy || voiceBusy || !serverReady || !input.value.trim();
}

function setMenuOpen(open) {
  sidebar.classList.toggle("open", open);
  sidebarBackdrop.hidden = !open;
  mobileMenu.setAttribute("aria-expanded", String(open));
  mobileMenu.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  // The closed mobile drawer must not leave invisible links in the tab order.
  const mobile = window.matchMedia("(max-width: 800px)").matches;
  sidebar.inert = mobile ? !open : sidebarCollapsed;
  document.querySelector("main").inert = mobile && open;
}

function setSidebarCollapsed(collapsed, persist = true) {
  const focusInside = sidebar.contains(document.activeElement);
  const focusExpand = document.activeElement === sidebarExpandBtn;
  sidebarCollapsed = collapsed;
  document.querySelector(".app-shell").classList.toggle("sidebar-collapsed", collapsed);
  sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
  sidebarExpandBtn.hidden = !collapsed;
  sidebarExpandBtn.setAttribute("aria-expanded", String(!collapsed));
  const mobile = window.matchMedia("(max-width: 800px)").matches;
  sidebar.inert = mobile ? !sidebar.classList.contains("open") : collapsed;
  if (!mobile && (focusInside || focusExpand)) {
    (collapsed ? sidebarExpandBtn : sidebarToggle).focus({ preventScroll: true });
  }
  if (persist) {
    try { localStorage.setItem("truecheck-sidebar-collapsed", String(collapsed)); }
    catch { /* Preference storage is optional. */ }
  }
}

function showView(view, moveFocus = true) {
  if (view !== currentView) window.TrueCheckSpeech.reset();
  currentView = view;
  if (view !== "settings") window.TrueCheckSettings.scrub();
  document.querySelectorAll(".view").forEach(section => {
    const active = section.id === view + "View";
    section.classList.toggle("active", active);
    section.hidden = !active;
  });
  document.querySelectorAll(".nav-item[data-view]").forEach(button => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  setMenuOpen(false);
  updateChatSelection();
  if (moveFocus) {
    if (view === "chat") input.focus();
    else document.getElementById(view + "Heading").focus();
  }
  if (view === "settings") window.TrueCheckSettings.refresh().catch(() => {
    document.getElementById("configurationFeedback").textContent = "Settings could not be loaded. Check the server and retry.";
  });
  updateClaimControls();
}

function fillInput(text) {
  if (busy || voiceBusy || !serverReady) return;
  openChat(activeChat);
  input.value = text;
  inputFeedback.textContent = "";
  resizeInput();
  input.focus();
}

function chatDate(key) {
  const [year, month, day] = key.split("-").map(Number);
  return new Date(year, month - 1, day, 12).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
}

function allRecords() {
  const records = new Map(historyStore.list().map(record => [record.id, record]));
  transientRecords.forEach((record, id) => { if (!records.has(id)) records.set(id, record); });
  return Array.from(records.values()).sort((a, b) => b.saved_at.localeCompare(a.saved_at));
}

function recordById(id) {
  return historyStore.get(id) || transientRecords.get(id) || null;
}

function recordsForChat(id) {
  return allRecords().filter(record => record.chat.id === id).reverse();
}

function updateChatSelection() {
  chatList.querySelectorAll("[data-chat-id]").forEach(button => {
    const selected = currentView === "chat" && button.dataset.chatId === activeChat.id;
    button.classList.toggle("active", selected);
    if (selected) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  chatList.querySelectorAll("[data-record-id]").forEach(button => {
    const selected = currentView === "chat" && button.dataset.recordId === openedRecordId;
    button.classList.toggle("active", selected);
    if (selected) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  });
}

function renderChats() {
  chatList.replaceChildren();
  const groups = new Map();
  allRecords().forEach(record => {
    const key = record.chat.id;
    if (!groups.has(key)) groups.set(key, { chat: record.chat, records: [] });
    groups.get(key).records.push(record);
  });
  const ordered = Array.from(groups.values()).sort((a, b) =>
    a.chat.created_at.localeCompare(b.chat.created_at) || a.chat.id.localeCompare(b.chat.id));
  const days = new Map();
  for (const group of ordered) {
    if (!days.has(group.chat.date)) days.set(group.chat.date, []);
    days.get(group.chat.date).push(group);
  }
  for (const date of Array.from(days.keys()).sort().reverse()) {
    const day = element("section", "chat-day");
    day.dataset.chatDay = date;
    const dayHeading = element("h3", "chat-day-title", chatDate(date));
    dayHeading.id = "chat-day-" + date;
    day.setAttribute("aria-labelledby", dayHeading.id);
    day.appendChild(dayHeading);
    days.get(date).forEach(({ chat, records: savedRecords }, index) => {
      const key = chat.id;
      const title = "Chat " + (index + 1);
      const description = title + ", " + chatDate(date);
      const group = element("div", "chat-group");
      const heading = element("div", "chat-group-heading");
      const toggle = element("button", "chat-expand", expandedChats.has(key) ? "▾" : "▸");
      toggle.type = "button";
      toggle.dataset.chatToggle = key;
      toggle.dataset.historyAction = "toggle";
      toggle.disabled = busy;
      toggle.setAttribute("aria-label", (expandedChats.has(key) ? "Collapse" : "Expand") + " verifications for " + description);
      toggle.setAttribute("aria-expanded", String(expandedChats.has(key)));
      toggle.setAttribute("aria-controls", "chat-entries-" + key);
      toggle.addEventListener("click", () => {
        if (busy) return;
        if (expandedChats.has(key)) expandedChats.delete(key);
        else expandedChats.add(key);
        renderChats();
        chatList.querySelector('[data-chat-toggle="' + key + '"]').focus();
      });
      const open = historyButton(title, () => openChat(chat), true);
      open.className = "chat-name";
      open.dataset.chatId = key;
      open.title = "Open " + description;
      open.setAttribute("aria-label", "Open " + description);
      heading.append(toggle, open);
      const entries = element("div", "chat-entries");
      entries.id = "chat-entries-" + key;
      entries.hidden = !expandedChats.has(key);
      const records = savedRecords.slice().reverse();
      records.forEach(record => {
        const button = historyButton(record.result.claim, () => openSavedCheck(record.id), true);
        button.className = "chat-verification";
        button.dataset.recordId = record.id;
        button.title = record.result.claim;
        button.setAttribute("aria-label", "Go to verification: " + record.result.claim + ", " + savedDate(record.saved_at));
        entries.appendChild(button);
      });
      group.append(heading, entries);
      day.appendChild(group);
    });
    chatList.appendChild(day);
  }
  updateChatSelection();
}

function savedDate(value) {
  return new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

async function exportCheck(record, format, feedback, button) {
  if (button?.disabled) return;
  const label = button?.textContent;
  if (button) {
    button.disabled = true;
    button.textContent = "Preparing...";
    button.setAttribute("aria-busy", "true");
  }
  feedback.textContent = "Preparing " + (format === "pdf" ? "PDF" : "Word document") + "...";
  try {
    const filename = await window.TrueCheckExport.download(record, format);
    feedback.textContent = "Download requested: " + filename;
  } catch (error) {
    feedback.textContent = error instanceof Error ? error.message : "The download could not be started. Please try again.";
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = label;
      button.removeAttribute("aria-busy");
    }
  }
}

function historyButton(label, callback, locked = false) {
  const button = element("button", "", label);
  button.type = "button";
  if (locked) {
    button.dataset.historyAction = "true";
    button.disabled = busy;
  }
  button.addEventListener("click", callback);
  return button;
}

function renderHistory() {
  const records = historyStore.list();
  const storage = historyStore.status();
  historyCount.textContent = records.length + (records.length === 1 ? " saved check" : " saved checks");
  historyStorageNotice.textContent = historySaveError || storage.message;
  historyStorageNotice.classList.toggle("storage-warning", Boolean(historySaveError) || !storage.persistent);
  historyEmpty.hidden = Boolean(records.length);
  historyList.hidden = !records.length;
  clearHistoryBtn.disabled = busy || (!records.length && !transientRecords.size && storage.persistent && !historySaveError);
  historyList.replaceChildren();
  records.forEach(record => {
    const row = element("article", "history-row");
    const details = element("div", "history-details");
    const claim = element("h2", "history-claim", record.result.claim);
    const label = record.result.prediction ? record.result.prediction.label :
      record.result.status === "screening_unavailable" ? "Screening unavailable" :
      record.result.status === "rejected" ? "Input screening" : "No prediction";
    const summary = label + " · Top " + record.result.top_k + " · " + record.result.evidence.length + " passages · " + savedDate(record.saved_at);
    details.append(claim, element("p", "history-summary", summary));
    const actions = element("div", "history-actions");
    const open = historyButton("Open", () => openSavedCheck(record.id), true);
    open.setAttribute("aria-label", "Open saved check: " + record.result.claim);
    const downloadPdf = historyButton("PDF", () => exportCheck(record, "pdf", historyFeedback, downloadPdf));
    downloadPdf.setAttribute("aria-label", "Download PDF: " + record.result.claim);
    const downloadWord = historyButton("Word", () => exportCheck(record, "docx", historyFeedback, downloadWord));
    downloadWord.setAttribute("aria-label", "Download Word: " + record.result.claim);
    const remove = historyButton("Delete", () => requestHistoryDeletion(record.id, remove), true);
    remove.classList.add("history-delete");
    remove.setAttribute("aria-label", "Delete saved check: " + record.result.claim);
    actions.append(open, downloadPdf, downloadWord, remove);
    row.append(details, actions);
    historyList.appendChild(row);
  });
  renderChats();
}

function renderChat(chat) {
  scrollRevision += 1;
  const records = recordsForChat(chat.id);
  activeChat = chat;
  historyStore.selectChat(records.length ? chat.id : null);
  messages.replaceChildren();
  welcome.hidden = Boolean(records.length);
  records.forEach(record => {
    const thread = element("article", "verification-thread");
    thread.id = "verification-" + record.id;
    const prompt = addUserMessage(record.result.claim, thread);
    prompt.id = "prompt-" + record.id;
    const { answer } = assistantMessage("Saved TrueCheck response", thread);
    renderResult(answer, record.result, record);
    if (transientRecords.has(record.id)) {
      answer.appendChild(element("p", "history-result-warning", "This check is available only in this page session. Download it to keep a copy."));
    } else if (!historyStore.status().persistent) {
      answer.appendChild(element("p", "history-result-warning", historyStore.status().message));
    }
    messages.appendChild(thread);
  });
  if (openedRecordId && !records.some(record => record.id === openedRecordId)) openedRecordId = null;
  renderChats();
}

function openChat(chat, targetId = null) {
  if (busy) return;
  window.TrueCheckSpeech.reset();
  expandedChats.add(chat.id);
  openedRecordId = targetId;
  renderChat(chat);
  showView("chat", false);
  if (targetId) {
    const prompt = document.getElementById("prompt-" + targetId);
    if (!prompt) return;
    prompt.tabIndex = -1;
    prompt.focus({ preventScroll: true });
    chatScroll.scrollTop += prompt.getBoundingClientRect().top - chatScroll.getBoundingClientRect().top - 12;
    prompt.classList.add("verification-highlight");
    window.setTimeout(() => prompt.classList.remove("verification-highlight"), 2200);
  } else {
    input.focus({ preventScroll: true });
    scrollToLatest();
  }
}

function openSavedCheck(id) {
  if (busy) return;
  const record = recordById(id);
  if (!record) {
    historyFeedback.textContent = "This saved check is no longer available.";
    inputFeedback.textContent = historyFeedback.textContent;
    renderHistory();
    return;
  }
  openChat(record.chat, id);
}

function requestHistoryDeletion(id, trigger) {
  if (busy) return;
  const record = id === null ? null : historyStore.get(id);
  if (id !== null && !record) return;
  pendingDeletion = { id };
  deletionFocus = trigger;
  document.getElementById("deleteHistoryHeading").textContent = id === null ? "Clear saved history?" : "Delete saved check?";
  document.getElementById("deleteHistoryDescription").textContent = id === null ?
    "Remove all saved checks from this browser and clear the displayed results? Your composer draft will stay." :
    "Remove this saved check from this browser? " + record.result.claim;
  confirmDeleteHistoryBtn.textContent = id === null ? "Clear history" : "Delete";
  deleteHistoryDialog.showModal();
  cancelDeleteHistoryBtn.focus();
}

function closeHistoryDeletion() {
  pendingDeletion = null;
  deleteHistoryDialog.close();
  if (deletionFocus?.isConnected && !deletionFocus.disabled) deletionFocus.focus();
  else document.getElementById("historyHeading").focus();
  deletionFocus = null;
}

function addUserMessage(text, container = messages) {
  welcome.hidden = true;
  const row = element("div", "message user");
  row.setAttribute("aria-label", "Your claim");
  row.appendChild(element("div", "bubble", text));
  container.appendChild(row);
  return row;
}

async function copyText(text, button, feedback) {
  button.disabled = true;
  feedback.textContent = "";
  try {
    let copied = false;
    if (window.isSecureContext && navigator.clipboard) {
      try {
        await navigator.clipboard.writeText(text);
        copied = true;
      } catch {
        // Some browsers block clipboard access even on localhost. Try selection copy.
      }
    }
    if (!copied) {
      const previousFocus = document.activeElement;
      const selection = element("textarea", "copy-fallback");
      selection.value = text;
      selection.setAttribute("readonly", "");
      document.body.appendChild(selection);
      try {
        selection.focus();
        selection.select();
        copied = document.execCommand("copy");
      } finally {
        selection.remove();
        if (previousFocus instanceof HTMLElement && previousFocus.isConnected) previousFocus.focus();
      }
    }
    feedback.textContent = copied ? "Response copied." : "Copy was blocked. Select the response text and copy it manually.";
  } catch {
    feedback.textContent = "Copy was blocked. Select the response text and copy it manually.";
  } finally {
    button.disabled = false;
  }
}

function assistantMessage(label, container = messages) {
  const row = element("div", "message assistant");
  row.setAttribute("aria-label", label);
  const avatar = element("img", "message-avatar");
  avatar.src = "/assets/logo.svg";
  avatar.alt = "";
  avatar.width = 30;
  avatar.height = 30;
  const bubble = element("div", "bubble");
  const answer = element("div", "answer");
  bubble.appendChild(answer);
  row.append(avatar, bubble);
  container.appendChild(row);
  return { row, answer };
}

function addResponseActions(answer, text, record = null, spoken = text) {
  const actions = element("div", "answer-actions");
  const copyButton = element("button", "copy-btn", "Copy response");
  copyButton.type = "button";
  const speechButton = element("button", "speak-btn", "Read aloud");
  speechButton.type = "button";
  actions.append(copyButton, speechButton);
  const feedback = element("div", "action-feedback");
  feedback.setAttribute("role", "status");
  window.TrueCheckSpeech.bindRead(speechButton, spoken, feedback);
  copyButton.addEventListener("click", () => copyText(text, copyButton, feedback));
  if (record) {
    const pdf = historyButton("Download PDF", () => exportCheck(record, "pdf", feedback, pdf));
    const word = historyButton("Download Word", () => exportCheck(record, "docx", feedback, word));
    actions.insertBefore(pdf, speechButton);
    actions.insertBefore(word, speechButton);
  }
  answer.append(actions, feedback);
}

function scrollToLatest() {
  const revision = ++scrollRevision;
  chatScroll.scrollTop = chatScroll.scrollHeight;
  window.requestAnimationFrame(() => {
    if (revision === scrollRevision) chatScroll.scrollTop = chatScroll.scrollHeight;
  });
}

function updateUploadControls() {
  attachBtn.disabled = busy || voiceBusy || !serverReady || !extractionAvailable;
  fileInput.disabled = busy || voiceBusy || !serverReady || !extractionAvailable;
}

function updateClaimControls() {
  input.disabled = voiceBusy || !serverReady;
  topKSelect.disabled = busy || voiceBusy || !serverReady;
  document.querySelectorAll(".suggestions button").forEach(button => { button.disabled = busy || voiceBusy || !serverReady; });
  useClaimBtn.disabled = busy || voiceBusy || !serverReady || extractionReview.hidden;
  composerWrap.dataset.initializing = String(!serverReady);
  updateUploadControls();
  window.TrueCheckSpeech.update(serverReady && !busy && !voiceBusy && currentView === "chat", transcriptionAvailable);
  resizeInput();
}

function setBusy(value, kind = "verification") {
  busy = value;
  busyKind = value ? kind : null;
  window.dispatchEvent(new CustomEvent("truecheck:busy", { detail: { busy: value || voiceBusy } }));
  document.querySelectorAll("[data-history-action]").forEach(button => { button.disabled = value; });
  if (!value) {
    clearHistoryBtn.disabled = !historyStore.list().length && !transientRecords.size && historyStore.status().persistent && !historySaveError;
  }
  extractionText.disabled = value;
  updateClaimControls();
  document.getElementById("claimForm").setAttribute("aria-busy", String(value));
  extractionDialog.setAttribute("aria-busy", String(value && kind === "extraction"));
  const sendLabel = value ? (kind === "extraction" ? "Extracting uploaded text" : "Checking claim") : "Send claim";
  sendBtn.setAttribute("aria-label", sendLabel);
  sendBtn.title = sendLabel;
  resizeInput();
}

function clearExtractionReview() {
  extractionFilename.textContent = "";
  extractionText.value = "";
  extractionFeedback.textContent = "";
  extractionProgress.hidden = true;
  extractionReview.hidden = true;
  useClaimBtn.disabled = true;
  fileInput.value = "";
  if (extractionDialog.open) extractionDialog.close();
}

function cancelExtraction() {
  const wasExtracting = busyKind === "extraction";
  if (wasExtracting) {
    conversation += 1;
    if (activeRequest) activeRequest.abort();
    activeRequest = null;
    setBusy(false);
  }
  clearExtractionReview();
  if (wasExtracting) inputFeedback.textContent = "Upload extraction cancelled. No claim was submitted.";
  input.focus();
}

async function extractFile(file) {
  if (busy || voiceBusy) return;
  if (!extractionAvailable) {
    inputFeedback.textContent = "Configure upload text extraction in Settings and retry.";
    return;
  }
  const supportedTypes = ["image/png", "image/jpeg", "image/webp", "application/pdf"];
  const namedType = /\.(png|jpe?g|webp|pdf)$/i.test(file.name);
  if (!supportedTypes.includes(file.type) && !((!file.type || file.type === "application/octet-stream") && namedType)) {
    inputFeedback.textContent = "Choose one PNG, JPEG, WebP image or PDF.";
    return;
  }
  if (!file.size || file.size > 10 * 1024 * 1024) {
    inputFeedback.textContent = file.size ? "The file exceeds 10 MiB. Choose a smaller file." : "The file is empty. Choose another file.";
    return;
  }
  inputFeedback.textContent = "";
  clearExtractionReview();
  const uploadProvider = extractionProviderName;
  document.getElementById("extractionDisclosure").textContent = "Extracting with " + uploadProvider + "…";
  extractionFilename.textContent = file.name;
  extractionProgress.hidden = false;
  extractionDialog.showModal();
  setBusy(true, "extraction");
  const generation = conversation;
  const controller = new AbortController();
  activeRequest = controller;
  let timedOut = false;
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 180000);
  try {
    const upload = new FormData();
    upload.append("file", file);
    const response = await window.TrueCheckSettings.request("/api/extract", { method: "POST", body: upload, signal: controller.signal });
    let result;
    try {
      result = await response.json();
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new Error("The extraction response could not be read. Please try the upload again.");
    }
    if (generation !== conversation) return;
    if (!response.ok) {
      throw new Error(typeof result?.detail === "string" ? result.detail : "Text extraction failed. Please try another file or type the claim yourself.");
    }
    if (!result || typeof result !== "object" || result.status !== "ready" || typeof result.filename !== "string" ||
        !supportedTypes.includes(result.media_type) || !Number.isInteger(result.pages) || result.pages < 1 || result.pages > 10 ||
        typeof result.text !== "string" || !result.text.trim() || result.text.length > 20000 ||
        typeof result.model !== "string" || typeof result.notice !== "string") {
      throw new Error("The server returned an incomplete extraction. Please try another file or type the claim yourself.");
    }
    extractionFilename.textContent = result.filename + " · " + result.pages + (result.pages === 1 ? " page" : " pages");
    document.getElementById("extractionDisclosure").textContent = "Extracted with " + uploadProvider + ".";
    extractionText.value = result.text;
    extractionReview.hidden = false;
  } catch (error) {
    if (generation !== conversation) return;
    document.getElementById("extractionDisclosure").textContent = "Text extraction with " + uploadProvider + ".";
    extractionFeedback.textContent = error.name === "AbortError" ?
      (timedOut ? "Extraction timed out. Your draft is unchanged. Try a smaller file or type the claim yourself." : "Extraction was cancelled. Your draft is unchanged.") :
      (error instanceof TypeError ? "The server connection was interrupted. Your draft is unchanged. Try the upload again." : error.message);
  } finally {
    window.clearTimeout(timeout);
    if (generation === conversation) {
      activeRequest = null;
      extractionProgress.hidden = true;
      setBusy(false);
      if (!extractionReview.hidden) extractionText.focus();
    }
  }
}

function wikipediaUrl(value) {
  try {
    const url = new URL(value);
    if ((url.protocol === "https:" || url.protocol === "http:") &&
        (url.hostname === "wikipedia.org" || url.hostname.endsWith(".wikipedia.org")) &&
        !url.username && !url.password) return url.href;
  } catch {
    // An invalid or non-Wikipedia URL is displayed without a link.
  }
  return null;
}

function sourceLink(source, text) {
  const url = wikipediaUrl(source.url);
  if (!url) return element("span", "source-label", text);
  const link = element("a", "source-link", text);
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function validateResult(result) {
  if (!result || typeof result !== "object" ||
      !["prediction", "no_evidence", "rejected", "screening_unavailable"].includes(result.status) ||
      !Number.isInteger(result.top_k) || result.top_k < 1 || result.top_k > 10 ||
      typeof result.claim !== "string" || !result.claim.trim() || result.claim.length > 4000 ||
      !Array.isArray(result.evidence) || result.evidence.length > result.top_k ||
      typeof result.notice !== "string" || !result.explanation ||
      !["ready", "unavailable", "error"].includes(result.explanation.status) ||
      typeof result.explanation.summary !== "string" ||
      !Array.isArray(result.explanation.citations) ||
      result.explanation.citations.some(id => typeof id !== "string")) {
    throw new Error("The server returned an unexpected response. Please retry.");
  }
  const ids = new Set();
  for (const source of result.evidence) {
    if (!source || typeof source.id !== "string" || !/^E[1-9]\d*$/.test(source.id) || ids.has(source.id) ||
        typeof source.page !== "string" || typeof source.text !== "string" ||
        typeof source.url !== "string") {
      throw new Error("The server returned incomplete evidence. Please retry.");
    }
    ids.add(source.id);
  }
  if (result.explanation.citations.some(id => !ids.has(id))) {
    throw new Error("The server returned an unknown citation. Please retry.");
  }
  const screening = result.screening;
  if (screening != null && (typeof screening !== "object" ||
      !["accepted", "invalid_input", "out_of_scope", "unavailable"].includes(screening.status) ||
      !(screening.reason === null || typeof screening.reason === "string"))) {
    throw new Error("The server returned invalid screening data. Please retry.");
  }
  if (result.status === "prediction") {
    if (!result.evidence.length || !result.prediction || !["SUPPORTS", "REFUTES"].includes(result.prediction.label) ||
        typeof result.prediction.confidence !== "number" || !Number.isFinite(result.prediction.confidence) ||
        result.prediction.confidence < 0 || result.prediction.confidence > 1 ||
        (screening && (screening.status !== "accepted" || result.explanation.status !== "ready"))) {
      throw new Error("The server returned an incomplete model prediction. Please retry.");
    }
  } else if (result.status === "no_evidence" && (result.prediction !== null || result.evidence.length)) {
    throw new Error("The server returned an inconsistent response. Please retry.");
  } else if (["rejected", "screening_unavailable"].includes(result.status)) {
    const permitted = result.status === "rejected" ? ["invalid_input", "out_of_scope"] : ["unavailable"];
    if (!screening || !permitted.includes(screening.status) || !screening.reason?.trim() ||
        result.prediction !== null || result.explanation.status === "ready" ||
        result.explanation.summary || result.explanation.citations.length) {
      throw new Error("The server returned an inconsistent screening result. Please retry.");
    }
  }
  return result;
}

function renderResult(answer, result, record = null) {
  answer.replaceChildren();
  if (record) {
    answer.appendChild(element("p", "saved-result-date", "Completed " + savedDate(record.saved_at)));
  }
  const copy = ["Claim: " + result.claim];
  const head = element("div", "answer-head");
  const blocked = ["rejected", "screening_unavailable"].includes(result.status);
  const screeningMessage = blocked ? result.screening.reason : "";
  const sourceNumbers = new Map(result.evidence.map((source, index) => [source.id, index + 1]));
  const citationLabel = id => "[" + (sourceNumbers.get(id) || id) + "]";
  const readableSummary = result.explanation.summary.replace(/\[(E[1-9]\d*(?:,\s*E[1-9]\d*)*)\]/g, (marker, group) => {
    const ids = group.split(/,\s*/);
    return ids.every(id => sourceNumbers.has(id)) ? "[" + ids.map(id => sourceNumbers.get(id)).join(", ") + "]" : marker;
  });
  head.appendChild(element("strong", "", blocked ? "Input screening" : "Model result"));
  const prediction = result.prediction;
  if (blocked) {
    const label = result.status === "screening_unavailable" ? "Screening unavailable" :
      result.screening.status === "invalid_input" ? "Check your input" : "Enter a factual claim";
    head.appendChild(element("span", "preview-badge", label));
    answer.appendChild(head);
    answer.appendChild(element("p", result.status === "screening_unavailable" ? "explanation-error" : "screening-message", screeningMessage));
    copy.push(label, screeningMessage);
  } else if (prediction) {
    const label = prediction.label === "SUPPORTS" ? "Supports" : "Refutes";
    head.appendChild(element("span", "verdict model-prediction", label));
    const confidence = "Model confidence: " + (prediction.confidence * 100).toFixed(1) + "%";
    answer.appendChild(head);
    answer.appendChild(element("p", "confidence", confidence));
    copy.push("Model result: " + label, confidence);
    if (record && !result.screening) {
      const legacyNote = "Saved before input screening was added.";
      answer.appendChild(element("p", "confidence-note", legacyNote));
      copy.push(legacyNote);
    }
  } else {
    head.appendChild(element("span", "preview-badge", "No result"));
    answer.appendChild(head);
    const empty = "No passages were found for this claim.";
    answer.appendChild(element("p", "", empty));
    copy.push(empty);
  }
  const explanation = result.explanation;
  const showExplanation = !blocked && result.status !== "no_evidence";
  if (showExplanation) answer.appendChild(element("h3", "", "What the evidence says"));
  if (showExplanation && explanation.status === "ready") {
    for (const paragraph of explanation.summary.split(/\n\s*\n/).filter(text => text.trim())) {
      const node = element("p", "explanation-summary");
      const markers = /\[(E[1-9]\d*(?:,\s*E[1-9]\d*)*)\]/g;
      let end = 0;
      for (const match of paragraph.matchAll(markers)) {
        node.appendChild(document.createTextNode(paragraph.slice(end, match.index)));
        const references = match[1].split(/,\s*/);
        references.forEach((id, index) => {
          if (index) node.appendChild(document.createTextNode(" "));
          const source = result.evidence.find(item => item.id === id);
          node.appendChild(source ? sourceLink(source, citationLabel(id)) : document.createTextNode("[" + id + "]"));
        });
        end = match.index + match[0].length;
      }
      node.appendChild(document.createTextNode(paragraph.slice(end)));
      answer.appendChild(node);
    }
    copy.push("What the evidence says", readableSummary);
    if (explanation.citations.length) {
      const citations = element("div", "citation-list");
      citations.appendChild(element("span", "", "Sources:"));
      for (const id of explanation.citations) {
        const source = result.evidence.find(item => item.id === id);
        citations.appendChild(sourceLink(source, citationLabel(id)));
      }
      answer.appendChild(citations);
      copy.push("Sources: " + explanation.citations.map(citationLabel).join(" "));
    }
  } else if (showExplanation) {
    const title = explanation.status === "unavailable" ? "Explanation unavailable." : "Explanation could not be generated.";
    const reason = typeof explanation.reason === "string" ? explanation.reason : "Please try again later.";
    answer.appendChild(element("p", "explanation-error", title + " " + reason));
    copy.push(title + " " + reason);
  }

  if (result.evidence.length) {
    const evidence = element("section", "evidence");
    const count = result.evidence.length;
    const heading = (blocked ? "Passages reviewed" : "Evidence used") + " · " + count + (count === 1 ? " passage" : " passages");
    evidence.setAttribute("aria-label", blocked ? "Passages reviewed" : "Evidence used");
    evidence.appendChild(element("h3", "evidence-title", heading));
    copy.push(heading);
    result.evidence.forEach(source => {
      const item = element("div", "source");
      const mark = element("div", "source-mark");
      mark.setAttribute("aria-hidden", "true");
      const content = element("div", "source-content");
      const label = citationLabel(source.id) + " " + source.page.replaceAll("_", " ");
      content.appendChild(sourceLink(source, label));
      content.appendChild(element("p", "source-text", source.text));
      item.append(mark, content);
      evidence.appendChild(item);
      const url = wikipediaUrl(source.url);
      copy.push(label + "\n" + source.text + (url ? "\n" + url : ""));
    });
    answer.appendChild(evidence);
  }
  if (prediction) {
    const confidenceNote = "Confidence is a model score, not a guarantee of accuracy.";
    answer.appendChild(element("p", "confidence-note result-footnote", confidenceNote));
    copy.push(confidenceNote);
  }
  const spoken = ["Your claim: " + result.claim,
    blocked ? screeningMessage : prediction ? "The model predicts " + prediction.label.toLowerCase() + "." : "No evidence was returned, so no prediction was made.",
    !showExplanation ? "" : explanation.status === "ready" ? explanation.summary : "Explanation unavailable. " + (explanation.reason || "Please retry later.")
  ].filter(Boolean).join("\n\n").replace(/\[E[1-9]\d*(?:,\s*E[1-9]\d*)*\]/g, "").replace(/https?:\/\/\S+/gi, "");
  addResponseActions(answer, copy.join("\n\n"), record, spoken);
}

function verificationErrorMessage(status, result) {
  if (typeof result?.detail === "string" && result.detail.trim()) return result.detail;
  if (status === 422 && Array.isArray(result?.detail)) {
    const errors = result.detail.filter(error => error && Array.isArray(error.loc));
    const forField = (error, field) => error.loc[0] === "body" && error.loc[1] === field;
    if (errors.some(error => forField(error, "top_k") && error.type === "extra_forbidden")) {
      return "The backend is out of date. Restart the server and refresh this page.";
    }
    if (errors.some(error => forField(error, "top_k"))) {
      return "Choose an evidence passage count from 1 to 10.";
    }
    if (errors.some(error => forField(error, "claim"))) {
      return "Please enter a valid claim of 4,000 characters or fewer.";
    }
  }
  if (status === 422 || Array.isArray(result?.detail)) {
    return "The request was rejected. Please check your input and try again.";
  }
  return "The verification request failed. Please retry.";
}

async function submitClaim(event) {
  if (event) event.preventDefault();
  if (busy || voiceBusy || !serverReady) return;
  const text = input.value.trim();
  if (!text) {
    inputFeedback.textContent = "Enter a claim first.";
    input.focus();
    return;
  }
  if (text.length > 4000) {
    inputFeedback.textContent = "Please keep the claim to 4,000 characters or fewer.";
    input.focus();
    return;
  }
  const topK = Number(topKSelect.value);
  if (!Number.isInteger(topK) || topK < 1 || topK > 10) {
    inputFeedback.textContent = "Choose an evidence passage count from 1 to 10.";
    topKSelect.focus();
    return;
  }
  inputFeedback.textContent = "";
  window.TrueCheckSpeech.stopSpeaking();
  openedRecordId = null;
  updateChatSelection();
  const thread = element("article", "verification-thread");
  messages.appendChild(thread);
  addUserMessage(text, thread);
  input.value = "";
  setBusy(true);
  const generation = conversation;
  const submittedChat = { ...activeChat };
  const controller = new AbortController();
  activeRequest = controller;
  let timedOut = false;
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 180000);
  const { answer } = assistantMessage("TrueCheck response", thread);
  const loading = element("p", "loading-message", "Finding evidence and checking your claim…");
  loading.setAttribute("role", "status");
  const progress = element("div", "verification-progress");
  const cancel = element("button", "verification-cancel", "Cancel");
  cancel.type = "button";
  cancel.setAttribute("aria-label", "Cancel verification");
  cancel.addEventListener("click", () => {
    if (generation !== conversation || activeRequest !== controller) return;
    conversation += 1;
    window.clearTimeout(timeout);
    controller.abort();
    activeRequest = null;
    setBusy(false);
    answer.replaceChildren();
    answer.appendChild(element("h3", "", "Verification cancelled"));
    const notice = element("p", "loading-note", "No result was saved. The server may finish work already started.");
    notice.setAttribute("role", "status");
    answer.appendChild(notice);
    if (!input.value.trim()) input.value = text;
    inputFeedback.textContent = "Verification cancelled. You can edit your claim and try again.";
    resizeInput();
    input.focus();
    scrollToLatest();
    checkServer();
  });
  progress.append(loading, cancel);
  answer.appendChild(progress);
  scrollToLatest();
  try {
    const response = await window.TrueCheckSettings.request("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim: text, top_k: topK }),
      signal: controller.signal
    });
    let result;
    try {
      result = await response.json();
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new Error("The server response could not be read. Please retry.");
    }
    if (!response.ok) {
      throw new Error(verificationErrorMessage(response.status, result));
    }
    if (generation !== conversation) return;
    validateResult(result);
    let record;
    try {
      record = historyStore.save(result, submittedChat).record;
      historySaveError = "";
    } catch {
      record = { id: "unsaved-" + Date.now(), saved_at: new Date().toISOString(), chat: submittedChat, result };
      transientRecords.set(record.id, record);
      if (transientRecords.size > window.TrueCheckHistory.MAX_ENTRIES) transientRecords.delete(transientRecords.keys().next().value);
      historySaveError = "This check could not be added to history. You can download it from the result below.";
    }
    openedRecordId = record.id;
    renderHistory();
    expandedChats.add(record.chat.id);
    renderChat(record.chat);
    checkServer();
  } catch (error) {
    if (generation !== conversation) return;
    answer.replaceChildren();
    const cancelled = error.name === "AbortError";
    const heading = cancelled ? (timedOut ? "Request timed out" : "Request cancelled") : "Verification could not be completed";
    const detail = cancelled ?
      (timedOut ? "The local model load or request took too long. No result was received. Please retry; the local models may now be loaded." : "No result was received for this request.") :
      (error instanceof TypeError ? "The server connection was interrupted. Check the server and retry." : error.message);
    answer.appendChild(element("h3", "", heading));
    answer.appendChild(element("p", "request-error", detail));
    addResponseActions(answer, "Claim: " + text + "\n\n" + heading + "\n" + detail);
    if (!input.value.trim()) input.value = text;
    checkServer();
  } finally {
    window.clearTimeout(timeout);
    if (generation === conversation) {
      activeRequest = null;
      setBusy(false);
      scrollToLatest();
    }
  }
  input.focus();
}

function validSetupProgress(components, progress) {
  const states = ["waiting", "loading", "preparing", "ready", "error"];
  return Boolean(components && progress && ["retriever", "classifier"].every(name => {
    const item = progress[name];
    return states.includes(components[name]) && item && Number.isInteger(item.completed) &&
      Number.isInteger(item.total) && item.total > 0 && item.completed >= 0 && item.completed <= item.total &&
      Number.isInteger(item.percent) && item.percent === Math.floor(item.completed * 100 / item.total) &&
      (components[name] !== "ready" || item.completed === item.total);
  }));
}

function showModelProgress(components = {}, progress = {}) {
  for (const name of ["retriever", "classifier"]) {
    const row = document.getElementById(name + "Progress");
    const status = document.getElementById(name + "Status");
    const track = row.querySelector('[role="progressbar"]');
    const state = components[name] || row.dataset.state || "waiting";
    const item = progress[name];
    const percent = item ? item.percent : Number(track.getAttribute("aria-valuenow") || 0);
    row.dataset.state = state;
    const label = state === "ready" ? "✓ 100%" : (state === "error" ? "Failed" : "Loading") + " · " + percent + "%";
    if (status.textContent !== label) status.textContent = label;
    track.setAttribute("aria-valuenow", String(percent));
    track.setAttribute("aria-valuetext", label);
    track.firstElementChild.style.width = percent + "%";
  }
  modelProgress.hidden = false;
}

function setupMessage(text) {
  if (serverStatus.textContent !== text) serverStatus.textContent = text;
}

async function retryModelSetup() {
  if (retryingSetup) return;
  retryingSetup = true;
  retryBtn.disabled = true;
  retryBtn.setAttribute("aria-busy", "true");
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 10000);
  try {
    const response = await window.TrueCheckSettings.request("/api/initialize", { method: "POST", signal: controller.signal });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Setup could not be restarted. Please retry.");
    if (result.status === "loading" && validSetupProgress(result.components, result.progress)) {
      serverState.dataset.state = "loading";
      setupMessage("Loading fact checker");
      showModelProgress(result.components, result.progress);
    }
    await checkServer();
  } catch (error) {
    serverReady = false;
    serverState.hidden = false;
    serverState.dataset.state = "error";
    setupMessage(error.name === "AbortError" ? "Setup did not respond. Please retry." : error instanceof TypeError ? "Local server unavailable. Start the server and retry." : error.message);
    retryBtn.hidden = false;
    updateClaimControls();
  } finally {
    window.clearTimeout(timeout);
    retryingSetup = false;
    retryBtn.disabled = false;
    retryBtn.removeAttribute("aria-busy");
  }
}

async function checkServer() {
  window.clearTimeout(healthPollTimer);
  healthPollTimer = null;
  if (!pageActive) return;
  if (checkingServer) { pendingServerCheck = true; return; }
  checkingServer = true;
  if (!serverReady && serverState.dataset.state !== "loading") setupMessage("Checking local setup…");
  retryBtn.hidden = true;
  let modelsLoading = false;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 5000);
  try {
    const response = await window.TrueCheckSettings.request("/api/health", { signal: controller.signal });
    if (!response.ok) throw new Error("Server health request failed");
    const health = await response.json();
    if (health.status !== "ok" || typeof health.verification_available !== "boolean") {
      throw new Error("Unexpected server health response");
    }
    if (!validSetupProgress(health.components, health.progress)) {
      throw new Error("Restart the updated backend to show local setup progress.");
    }
    serverReady = health.verification_available && health.pipeline_status === "ready" &&
      health.components.retriever === "ready" && health.components.classifier === "ready";
    showModelProgress(health.components, health.progress);
    extractionAvailable = health.extraction_available === true;
    transcriptionAvailable = health.transcription_available === true;
    const extractionProvider = {openai: "OpenAI", anthropic: "Anthropic"}[health.extraction_provider];
    extractionProviderName = extractionProvider || "your configured provider";
    if (!extractionDialog.open) {
      document.getElementById("extractionDisclosure").textContent = extractionAvailable ?
        "Text extraction with " + extractionProviderName + "." : "Configure text extraction in Settings.";
    }
    attachBtn.title = extractionAvailable ? "Upload one PNG, JPEG, WebP image or PDF (up to 10 MiB, PDFs up to 10 pages)" : "Configure upload extraction in Settings";
    attachBtn.setAttribute("aria-label", extractionAvailable ? "Upload an image or PDF for text extraction" : "Configure upload extraction in Settings");
    serverState.dataset.state = serverReady ? "ready" : "error";
    if (!health.verification_available) {
      serverState.dataset.state = "error";
      serverState.hidden = false;
      setupMessage(typeof health.message === "string" ? health.message : "Verification is unavailable. Check the server setup.");
      retryBtn.hidden = false;
    } else if (health.pipeline_status === "error") {
      serverState.hidden = false;
      serverState.dataset.state = "error";
      setupMessage(typeof health.message === "string" ? health.message : "Local setup could not finish. Choose Retry to try again.");
      retryBtn.hidden = false;
    } else if (["not_loaded", "loading"].includes(health.pipeline_status)) {
      modelsLoading = true;
      serverState.dataset.state = "loading";
      serverState.hidden = false;
      setupMessage("Loading fact checker");
    } else {
      serverState.hidden = serverReady;
      setupMessage(serverReady ? "" : "Local setup is not ready. Choose Retry to try again.");
      retryBtn.hidden = serverReady;
    }
  } catch (error) {
    serverReady = false;
    extractionAvailable = false;
    transcriptionAvailable = false;
    serverState.dataset.state = "error";
    serverState.hidden = false;
    setupMessage(error.message === "Restart the updated backend to show local setup progress." ? error.message : "Local server unavailable. Start the server and retry.");
    showModelProgress();
    retryBtn.hidden = false;
  } finally {
    window.clearTimeout(timeout);
    checkingServer = false;
    updateClaimControls();
    if (pendingServerCheck) {
      pendingServerCheck = false;
      if (pageActive && !document.hidden) checkServer();
    } else if (modelsLoading && pageActive && !document.hidden) {
      healthPollTimer = window.setTimeout(checkServer, 500);
    }
  }
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    window.clearTimeout(healthPollTimer);
    healthPollTimer = null;
  } else {
    checkServer();
  }
});
window.addEventListener("pagehide", () => {
  pageActive = false;
  pendingServerCheck = false;
  window.clearTimeout(healthPollTimer);
  healthPollTimer = null;
});
window.addEventListener("pageshow", event => {
  if (event.persisted) {
    pageActive = true;
    checkServer();
  }
});

document.getElementById("claimForm").addEventListener("submit", submitClaim);
window.addEventListener("truecheck:configuration", checkServer);
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    submitClaim();
  }
});
input.addEventListener("input", () => {
  inputFeedback.textContent = "";
  resizeInput();
});
document.querySelectorAll(".suggestions button").forEach(button => {
  button.addEventListener("click", () => fillInput(button.dataset.prompt));
});
document.querySelectorAll(".nav-item[data-view]").forEach(button => {
  button.addEventListener("click", () => showView(button.dataset.view));
});
document.querySelectorAll("[data-home]").forEach(button => {
  button.addEventListener("click", () => {
    window.TrueCheckSpeech.reset();
    showView("chat", false);
    welcome.hidden = false;
    scrollRevision += 1;
    chatScroll.scrollTop = 0;
    const heading = welcome.querySelector("h1");
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
  });
});
document.getElementById("newChatBtn").addEventListener("click", () => {
  window.TrueCheckSpeech.reset();
  const wasBusy = busy;
  conversation += 1;
  if (activeRequest) activeRequest.abort();
  activeRequest = null;
  setBusy(false);
  clearExtractionReview();
  openedRecordId = null;
  renderChat(historyStore.createChat());
  input.value = "";
  inputFeedback.textContent = wasBusy ? "Previous request cancelled. Any late response will be ignored." : "";
  showView("chat");
  resizeInput();
  scrollToLatest();
});
attachBtn.addEventListener("click", () => {
  if (busy || voiceBusy || !serverReady || !extractionAvailable) return;
  fileInput.value = "";
  fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 1) {
    inputFeedback.textContent = "Choose one file at a time.";
    return;
  }
  const file = fileInput.files[0];
  if (file) extractFile(file);
});
document.getElementById("closeExtractionBtn").addEventListener("click", cancelExtraction);
document.getElementById("cancelExtractionBtn").addEventListener("click", cancelExtraction);
extractionDialog.addEventListener("cancel", event => {
  event.preventDefault();
  cancelExtraction();
});
extractionText.addEventListener("input", () => { extractionFeedback.textContent = ""; });
useClaimBtn.addEventListener("click", () => {
  if (busy || !serverReady || extractionReview.hidden) return;
  const hasSelection = extractionText.selectionStart < extractionText.selectionEnd;
  const claim = (hasSelection ? extractionText.value.slice(extractionText.selectionStart, extractionText.selectionEnd) : extractionText.value).trim();
  if (!claim || claim.length > 4000) {
    extractionFeedback.textContent = !claim ? "Enter or select one factual claim before continuing." :
      "This text has " + claim.length.toLocaleString() + " characters. Select one claim or edit it down to 4,000 characters or fewer before using it.";
    extractionText.focus();
    return;
  }
  clearExtractionReview();
  fillInput(claim);
  inputFeedback.textContent = "Extracted text added to the composer. Review it, then send when ready.";
});
sidebarToggle.addEventListener("click", () => setSidebarCollapsed(!sidebarCollapsed));
sidebarExpandBtn.addEventListener("click", () => setSidebarCollapsed(false));
window.addEventListener("storage", event => {
  if (event.key === "truecheck-sidebar-collapsed" || event.key === null) {
    setSidebarCollapsed(event.key === null ? false : event.newValue === "true", false);
  }
});
mobileMenu.addEventListener("click", () => {
  const open = !sidebar.classList.contains("open");
  setMenuOpen(open);
  if (open) document.getElementById("newChatBtn").focus();
});
sidebarBackdrop.addEventListener("click", () => {
  setMenuOpen(false);
  mobileMenu.focus();
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && sidebar.classList.contains("open")) {
    setMenuOpen(false);
    mobileMenu.focus();
  }
});
window.matchMedia("(max-width: 800px)").addEventListener("change", () => setMenuOpen(false));
retryBtn.addEventListener("click", retryModelSetup);

clearHistoryBtn.addEventListener("click", () => requestHistoryDeletion(null, clearHistoryBtn));
cancelDeleteHistoryBtn.addEventListener("click", closeHistoryDeletion);
deleteHistoryDialog.addEventListener("cancel", event => {
  event.preventDefault();
  closeHistoryDeletion();
});
confirmDeleteHistoryBtn.addEventListener("click", () => {
  if (!pendingDeletion || busy) return;
  const { id } = pendingDeletion;
  const hadActiveRecords = recordsForChat(activeChat.id).length > 0;
  try {
    if (id === null) historyStore.clear();
    else historyStore.remove(id);
    if (id === null) transientRecords.clear();
    else transientRecords.delete(id);
    historySaveError = "";
    if (id === null || openedRecordId === id) {
      openedRecordId = null;
    }
    if (id === null || (hadActiveRecords && !recordsForChat(activeChat.id).length)) activeChat = historyStore.createChat();
    renderChat(activeChat);
    renderHistory();
    historyFeedback.textContent = historyStore.status().persistent ?
      (id === null ? "Saved history cleared." : "Saved check deleted.") :
      "Removed from this page. " + historyStore.status().message;
  } catch {
    historyFeedback.textContent = "History could not be updated. Please try again.";
  }
  closeHistoryDeletion();
});
window.addEventListener("storage", event => {
  if (event.key !== null && event.key !== window.TrueCheckHistory.STORAGE_KEY) return;
  const hadActiveRecords = recordsForChat(activeChat.id).length > 0;
  historyStore.reload();
  renderHistory();
  if (!busy) {
    const previousTop = chatScroll.scrollTop;
    const wasAtBottom = chatScroll.scrollHeight - chatScroll.clientHeight - previousTop < 32;
    if (hadActiveRecords && !recordsForChat(activeChat.id).length) activeChat = historyStore.createChat();
    renderChat(activeChat);
    if (wasAtBottom) scrollToLatest();
    else chatScroll.scrollTop = previousTop;
  }
  historyFeedback.textContent = "Saved history was updated in another tab.";
});

window.TrueCheckSpeech.init({
  canRecord() {
    if (input.value.length >= 4000) {
      inputFeedback.textContent = "Make room in the claim box before adding a recording. Claims can contain up to 4,000 characters.";
      return false;
    }
    return serverReady && !busy && !voiceBusy && currentView === "chat";
  },
  onBusy(value) {
    voiceBusy = value;
    window.dispatchEvent(new CustomEvent("truecheck:busy", { detail: { busy: busy || value } }));
    document.getElementById("claimForm").setAttribute("aria-busy", String(busy || value));
    updateClaimControls();
  },
  onFeedback(message) { inputFeedback.textContent = message; },
  onResult(text) {
    const draft = input.value.trimEnd();
    const combined = draft ? draft + " " + text : text;
    if (combined.length > 4000) {
      inputFeedback.textContent = "The transcript and draft exceed 4,000 characters. Your draft was kept. Record a shorter claim.";
      input.focus();
      return;
    }
    input.value = combined;
    inputFeedback.textContent = "Transcript added. Review and edit your claim before sending.";
    resizeInput();
    input.focus();
  }
});
renderHistory();
renderChat(activeChat);
setMenuOpen(false);
setSidebarCollapsed(sidebarCollapsed, false);
updateClaimControls();
scrollToLatest();
window.addEventListener("load", () => {
  if (currentView === "chat" && !openedRecordId) scrollToLatest();
}, { once: true });
checkServer();
