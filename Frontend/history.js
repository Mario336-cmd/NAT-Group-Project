"use strict";

(function (global) {
  const STORAGE_KEY = "truecheck.history";
  const MAX_ENTRIES = 50;
  const VERSION = 2;
  const ACTIVE_CHAT_KEY = "truecheck.activeChat";
  // localStorage implementations commonly account for UTF-16 code units.
  const MAX_STORAGE_BYTES = 2 * 1024 * 1024;
  let nextId = 0;

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function createStore(options) {
    if (!options || typeof options.validateResult !== "function") {
      throw new TypeError("History requires a result validator.");
    }
    const validateResult = options.validateResult;
    let records = [];
    let writeBlocked = false;
    let state = { persistent: false, message: "Browser history has not been read yet." };
    const pendingSaves = new Map();
    const pendingRemovals = new Set();
    let pendingClear = false;

    function storage() {
      const target = global.localStorage;
      if (!target || typeof target.getItem !== "function" ||
          typeof target.setItem !== "function" || typeof target.removeItem !== "function") {
        throw new Error("Browser storage is unavailable.");
      }
      return target;
    }

    function completedResult(value) {
      const result = clone(value);
      if (!result || typeof result !== "object" || Array.isArray(result) ||
          !["prediction", "no_evidence", "rejected", "screening_unavailable"].includes(result.status) || validateResult(result) === false) {
        throw new TypeError("Only completed, valid verification results can be saved.");
      }
      return result;
    }

    function envelope(entries) {
      return JSON.stringify({ version: VERSION, entries: entries });
    }

    function requireRecordFits(record) {
      if (envelope([record]).length * 2 > MAX_STORAGE_BYTES) {
        throw new RangeError("This result is too large to save in browser history. Previous checks were kept. You can still export this result.");
      }
    }

    function bounded(entries) {
      const limited = entries.slice(0, MAX_ENTRIES);
      while (limited.length > 1 && envelope(limited).length * 2 > MAX_STORAGE_BYTES) {
        limited.pop();
      }
      return limited;
    }

    function trimPending() {
      const ids = new Set(records.map(record => record.id));
      for (const id of pendingSaves.keys()) {
        if (!ids.has(id)) pendingSaves.delete(id);
      }
    }

    function unsaved(message) {
      state = { persistent: false, message: message };
    }

    function saved() {
      state = { persistent: true, message: "" };
    }

    function validDate(value) {
      if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)) {
        return false;
      }
      const time = Date.parse(value);
      return Number.isFinite(time) && new Date(time).toISOString() === value;
    }

    function localDay(value) {
      const date = new Date(value);
      return [date.getFullYear(), String(date.getMonth() + 1).padStart(2, "0"), String(date.getDate()).padStart(2, "0")].join("-");
    }

    function completedChat(value) {
      if (!value || typeof value !== "object" || Array.isArray(value) ||
          typeof value.id !== "string" || !/^[A-Za-z0-9_-]{1,128}$/.test(value.id) ||
          !validDate(value.created_at) || typeof value.date !== "string" ||
          !/^\d{4}-\d{2}-\d{2}$/.test(value.date) ||
          !validDate(value.date + "T12:00:00.000Z")) {
        throw new TypeError("Invalid chat metadata.");
      }
      return { id: value.id, created_at: value.created_at, date: value.date };
    }

    function createChat() {
      const now = new Date();
      return { id: "chat-" + newId(), created_at: now.toISOString(), date: localDay(now) };
    }

    function decode(raw) {
      if (raw === null) return { entries: [], warning: "", fatal: false };
      if (typeof raw !== "string" || raw.length * 2 > MAX_STORAGE_BYTES) {
        return { entries: [], fatal: true, warning:
          "Saved history is too large or unreadable. Current checks remain in this tab. Clear history to reset browser storage." };
      }
      let value;
      try {
        value = JSON.parse(raw);
      } catch {
        return { entries: [], fatal: true, warning:
          "Saved history could not be read. Current checks remain in this tab. Clear history to reset browser storage." };
      }
      if (!value || typeof value !== "object" || Array.isArray(value) || ![1, VERSION].includes(value.version) ||
          !Array.isArray(value.entries)) {
        return { entries: [], fatal: true, warning:
          "Saved history uses an unsupported format. Current checks remain in this tab. Clear history to reset browser storage." };
      }
      const entries = [];
      const ids = new Set();
      const chats = new Map();
      let invalid = value.entries.length > MAX_ENTRIES;
      for (const entry of value.entries.slice(0, MAX_ENTRIES)) {
        try {
          if (!entry || typeof entry !== "object" || Array.isArray(entry) ||
              typeof entry.id !== "string" || !/^[A-Za-z0-9_-]{1,128}$/.test(entry.id) ||
              ids.has(entry.id) || !validDate(entry.saved_at)) {
            throw new TypeError("Invalid saved record.");
          }
          const result = completedResult(entry.result);
          const chat = value.version === 1 ? {
            id: "legacy-" + localDay(entry.saved_at),
            created_at: entry.saved_at,
            date: localDay(entry.saved_at)
          } : completedChat(entry.chat);
          const previous = chats.get(chat.id);
          if (value.version === 1 && previous && previous.created_at > chat.created_at) {
            previous.created_at = chat.created_at;
          } else if (value.version === VERSION && previous &&
              (previous.created_at !== chat.created_at || previous.date !== chat.date)) {
            throw new TypeError("Conflicting chat metadata.");
          }
          if (!previous) chats.set(chat.id, chat);
          ids.add(entry.id);
          entries.push({ id: entry.id, saved_at: entry.saved_at, chat: chats.get(chat.id), result: result });
        } catch {
          invalid = true;
        }
      }
      entries.sort((a, b) => b.saved_at.localeCompare(a.saved_at));
      return { entries: entries, fatal: false, warning: invalid ?
        "Some saved checks were invalid and were hidden. Current valid checks remain available. Clear history to reset browser storage." : "" };
    }

    function merge(entries, keepCurrent) {
      const merged = new Map();
      // Pending saves win ties in saved_at and retain their newest-first order.
      for (const record of Array.from(pendingSaves.values()).reverse()) merged.set(record.id, record);
      if (!pendingClear) {
        if (keepCurrent) {
          for (const record of records) {
            if (!merged.has(record.id)) merged.set(record.id, record);
          }
        }
        for (const record of entries) {
          if (!merged.has(record.id)) merged.set(record.id, record);
        }
      }
      for (const id of pendingRemovals) merged.delete(id);
      records = bounded(Array.from(merged.values()).sort((a, b) => b.saved_at.localeCompare(a.saved_at)));
      trimPending();
    }

    function reload() {
      let decoded;
      try {
        decoded = decode(storage().getItem(STORAGE_KEY));
      } catch {
        writeBlocked = true;
        unsaved("Browser history could not be read. Current checks remain available in this tab; storage has not been overwritten.");
        return clone(records);
      }
      if (decoded.fatal) {
        writeBlocked = true;
        unsaved(decoded.warning);
        return clone(records);
      }
      merge(decoded.entries, Boolean(decoded.warning));
      writeBlocked = Boolean(decoded.warning);
      if (decoded.warning) unsaved(decoded.warning);
      else if (pendingClear || pendingSaves.size || pendingRemovals.size) {
        unsaved("Some history changes are available only in this tab. A later save or clear can retry browser storage.");
      } else saved();
      return clone(records);
    }

    function persist() {
      if (writeBlocked) return false;
      const serialised = envelope(records);
      if (serialised.length * 2 > MAX_STORAGE_BYTES) {
        unsaved("This result is too large for browser history. It remains available in this tab until reload; previously saved browser history has not been overwritten.");
        return false;
      }
      try {
        storage().setItem(STORAGE_KEY, serialised);
      } catch {
        unsaved("History could not be saved in this browser, possibly because storage is blocked or full. Current checks and changes remain available only in this tab.");
        return false;
      }
      pendingSaves.clear();
      pendingRemovals.clear();
      pendingClear = false;
      saved();
      return true;
    }

    function newId() {
      try {
        const id = global.crypto.randomUUID();
        if (typeof id === "string" && /^[A-Za-z0-9_-]{1,128}$/.test(id) &&
            !records.some(record => record.id === id)) return id;
      } catch {
        // Older browsers and restricted contexts can lack crypto.randomUUID.
      }
      let id;
      do {
        nextId += 1;
        id = "check-" + Date.now().toString(36) + "-" + nextId.toString(36) + "-" + Math.random().toString(36).slice(2, 12);
      } while (records.some(record => record.id === id));
      return id;
    }

    function save(result, chat) {
      const record = { id: newId(), saved_at: new Date().toISOString(), chat: completedChat(chat), result: completedResult(result) };
      // Reject one oversized result before evicting any older visible checks.
      requireRecordFits(record);
      // Storage events are asynchronous: rebase before writing so a stale tab
      // does not drop another tab's newer check or restore a deleted check.
      reload();
      // Reuse canonical metadata if another tab already saved this conversation.
      const existingChat = records.find(item => item.chat.id === record.chat.id);
      if (existingChat) record.chat = clone(existingChat.chat);
      if (records.some(item => item.id === record.id)) record.id = newId();
      requireRecordFits(record);
      records = bounded([record].concat(records));
      pendingSaves.set(record.id, record);
      pendingRemovals.delete(record.id);
      trimPending();
      const persisted = persist();
      return { record: clone(record), persisted: persisted };
    }

    function remove(id) {
      reload();
      const exists = records.some(record => record.id === id);
      if (!exists) return { removed: false, persisted: state.persistent };
      records = records.filter(record => record.id !== id);
      pendingSaves.delete(id);
      pendingRemovals.add(id);
      return { removed: true, persisted: persist() };
    }

    function clear() {
      records = [];
      pendingSaves.clear();
      pendingRemovals.clear();
      pendingClear = true;
      try {
        storage().removeItem(STORAGE_KEY);
      } catch {
        unsaved("History was cleared from this tab, but browser storage could not be cleared. Previously saved checks may return after reload; retry Clear history when storage is available.");
        return { persisted: false };
      }
      pendingClear = false;
      writeBlocked = false;
      saved();
      return { persisted: true };
    }

    reload();
    return {
      createChat: createChat,
      selectedChat: function () {
        try {
          const id = global.sessionStorage.getItem(ACTIVE_CHAT_KEY);
          const selected = records.find(record => record.chat.id === id);
          return selected ? clone(selected.chat) : null;
        } catch {
          return null;
        }
      },
      selectChat: function (id) {
        try {
          // Session storage remembers navigation per tab, never empty conversations.
          if (id) global.sessionStorage.setItem(ACTIVE_CHAT_KEY, id);
          else global.sessionStorage.removeItem(ACTIVE_CHAT_KEY);
        } catch {
          // History still works if navigation state cannot survive a refresh.
        }
      },
      list: function () { return clone(records); },
      get: function (id) {
        const record = records.find(item => item.id === id);
        return record ? clone(record) : null;
      },
      save: save,
      remove: remove,
      clear: clear,
      reload: reload,
      status: function () { return { persistent: state.persistent, message: state.message }; }
    };
  }

  global.TrueCheckHistory = { createStore: createStore, STORAGE_KEY: STORAGE_KEY, MAX_ENTRIES: MAX_ENTRIES };
})(window);
