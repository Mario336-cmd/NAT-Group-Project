(function () {
  "use strict";

  const REVOKE_DELAY_MS = 10000;
  const TYPES = {
    pdf: "application/pdf",
    docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  };

  async function download(record, format) {
    if (!record || typeof record.id !== "string" || !record.result ||
        typeof record.saved_at !== "string" || !Number.isFinite(Date.parse(record.saved_at))) {
      throw new Error("This saved verification cannot be exported because its data is incomplete.");
    }
    if (!Object.hasOwn(TYPES, format)) throw new Error("Choose PDF or Word as the report format.");
    if (!window.URL || !window.URL.createObjectURL || !window.URL.revokeObjectURL || !window.Blob) {
      throw new Error("Report downloads are not supported in this browser.");
    }
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 45000);
    let response;
    let bytes;
    try {
      response = await window.TrueCheckSettings.request("/api/export/" + format, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: record.id, saved_at: record.saved_at, result: record.result }),
        signal: controller.signal
      });
      if (!response.ok) {
        if (response.status === 404) throw new Error("Restart the updated backend to enable PDF and Word downloads.");
        let error;
        try { error = await response.json(); } catch { /* Use a safe fallback. */ }
        throw new Error(typeof error?.detail === "string" ? error.detail : "The report could not be generated. Please retry.");
      }
      const type = (response.headers.get("content-type") || "").split(";")[0].trim();
      if (type !== TYPES[format]) throw new Error("The server returned an unexpected report format. Please retry.");
      bytes = new Uint8Array(await response.arrayBuffer());
      const valid = format === "pdf" ? bytes[0] === 37 && bytes[1] === 80 && bytes[2] === 68 && bytes[3] === 70 :
        bytes[0] === 80 && bytes[1] === 75 && bytes[2] === 3 && bytes[3] === 4;
      if (!valid) throw new Error("The server returned an invalid report file. Please retry.");
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The report download timed out. Please retry.");
      if (error instanceof TypeError) throw new Error("The server connection was interrupted. Check the server and retry.");
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
    const stamp = new Date(record.saved_at).toISOString().replace(/[:.]/g, "-");
    const id = record.id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64) || "result";
    const filename = "truecheck-" + stamp + "-" + id + "." + format;
    let url;
    let link;
    try {
      url = window.URL.createObjectURL(new window.Blob([bytes], { type: TYPES[format] }));
      link = window.document.createElement("a");
      link.href = url;
      link.download = filename;
      link.hidden = true;
      window.document.body.appendChild(link);
      link.click();
      const downloadURL = url;
      window.setTimeout(() => window.URL.revokeObjectURL(downloadURL), REVOKE_DELAY_MS);
      return filename;
    } catch {
      if (url) window.URL.revokeObjectURL(url);
      throw new Error("The report could not be downloaded. Please try again in a browser that allows downloads.");
    } finally {
      if (link) link.remove();
    }
  }

  window.TrueCheckExport = Object.freeze({ download });
}());
