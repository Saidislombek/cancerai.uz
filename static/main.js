const el = (id) => document.getElementById(id);

const CLASS_DESCRIPTIONS = {
  HSIL:
    "HSIL — здесь будет текст с описанием клинического значения данного фенотипа. " +
    "Пока используется временное описание, которое ты позже заменишь на финальное.",
  LSIL:
    "LSIL — временное пояснение. Здесь можно кратко описать характер изменений клеток " +
    "и примерные рекомендации по дальнейшему обследованию.",
  NILM:
    "NILM — результат без выраженных патологических изменений. " +
    "Точный текст-описание мы позже подставим сюда.",
  SCC:
    "SCC — предполагаемый инвазивный процесс. Сейчас это только шаблон текста, " +
    "который позже будет заменён на согласованное медицинское описание.",
};

const FIXED_CLASS_ORDER = ["HSIL", "LSIL", "NILM", "SCC"];

let selectedFile = null;
let selectedUrl = null;

function setStatus(kind, msg) {
  const box = el("status");
  if (!box) return;

  box.classList.remove("hidden");
  box.className = "rounded-lg border p-3 text-sm";

  if (kind === "error") box.classList.add("bg-red-50", "border-red-200", "text-red-900");
  if (kind === "ok") box.classList.add("bg-green-50", "border-green-200", "text-green-900");
  if (kind === "info") box.classList.add("bg-gray-50", "border-gray-200", "text-gray-900");

  box.textContent = msg;
}

function clearStatus() {
  const box = el("status");
  if (!box) return;
  box.classList.add("hidden");
  box.textContent = "";
}

/**
 * Полностью скрывает блок результатов и чистит данные в DOM.
 * Важно: используем при "Очистить кэш модели", чтобы старый результат исчез.
 */
function clearResultsUI() {
  const results = el("results");
  if (results) results.classList.add("hidden");

  const metricsTbody = el("metricsTbody");
  if (metricsTbody) metricsTbody.innerHTML = "";

  const probsTbody = el("probsTbody");
  if (probsTbody) probsTbody.innerHTML = "";

  const interp = el("interpretation");
  if (interp) interp.textContent = "";

  const img = el("resultImg");
  if (img) img.removeAttribute("src");
}

function setFile(file) {
  selectedFile = file || null;

  const row = el("fileRow");
  const name = el("fileName");
  const runBtn = el("runBtn");

  if (!selectedFile) {
    if (row) row.classList.add("hidden");
    if (name) name.textContent = "";
    if (runBtn) runBtn.disabled = true;

    if (selectedUrl) URL.revokeObjectURL(selectedUrl);
    selectedUrl = null;

    // при сбросе файла — прячем результаты
    clearResultsUI();
    return;
  }

  if (selectedUrl) URL.revokeObjectURL(selectedUrl);
  selectedUrl = URL.createObjectURL(selectedFile);

  if (row) row.classList.remove("hidden");
  if (name) name.textContent = selectedFile.name;
  if (runBtn) runBtn.disabled = false;
}

function setupUploader() {
  const dropzone = el("dropzone");
  const browseBtn = el("browseBtn");
  const fileInput = el("fileInput");
  const clearFileBtn = el("clearFileBtn");

  if (browseBtn && fileInput) {
    browseBtn.addEventListener("click", () => fileInput.click());
  }

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      clearStatus();
      const f = fileInput.files?.[0];
      setFile(f);
    });
  }

  if (clearFileBtn && fileInput) {
    clearFileBtn.addEventListener("click", () => {
      fileInput.value = "";
      setFile(null);
      clearStatus();
    });
  }

  if (!dropzone || !fileInput) return;

  const prevent = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  ["dragenter", "dragover", "dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, prevent);
  });

  dropzone.addEventListener("dragover", () => {
    dropzone.classList.add("border-teal-700", "bg-indigo-50");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("border-teal-700", "bg-indigo-50");
  });

  dropzone.addEventListener("drop", (e) => {
    dropzone.classList.remove("border-teal-700", "bg-indigo-50");
    clearStatus();

    const f = e.dataTransfer?.files?.[0];
    if (!f) return;

    // записываем в input (чтобы было “честно”)
    const dt = new DataTransfer();
    dt.items.add(f);
    fileInput.files = dt.files;

    setFile(f);
  });
}

async function loadModels() {
  const sel = el("modelSelect");
  if (!sel) return;

  sel.innerHTML = `<option value="">Загрузка...</option>`;
  try {
    const r = await fetch("/api/models");
    const data = await r.json();

    sel.innerHTML = "";
    for (const m of data.models || []) {
      const opt = document.createElement("option");
      opt.value = m.model_id;
      opt.textContent = m.display_name;
      sel.appendChild(opt);
    }

    // По умолчанию: swin_s, иначе первый доступный
    const hasSwin = Array.from(sel.options).some((o) => o.value === "swin_s");
    sel.value = hasSwin ? "swin_s" : (sel.options[0]?.value || "");

  } catch (e) {
    sel.innerHTML = `<option value="">Не удалось загрузить список моделей</option>`;
    setStatus("error", "Не удалось получить список моделей (/api/models). Проверь backend.");
  }
}

function renderResults(payload) {
  const results = el("results");
  if (!results) return;

  // 1) Метрики
  const metricsTbody = el("metricsTbody");
  if (metricsTbody) {
    metricsTbody.innerHTML = "";

    const elapsedSec = (Number(payload.latency_ms || 0) / 1000).toFixed(3) + " сек";
    const conf = (Number(payload.predicted_probability || 0) * 100).toFixed(2) + " %";
    const pred = payload.predicted_label || "-";

    const rows = [
      ["1", "Время на прогноз", elapsedSec],
      ["2", "Точность прогнозирования", conf],
      ["3", "Предсказанный класс", pred],
    ];

    for (const [n, k, v] of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="border-2 border-black px-3 py-2 text-[16px] text-center">${n}</td>
        <td class="border-2 border-black px-3 py-2 text-[16px] text-center">${k}</td>
        <td class="border-2 border-black px-3 py-2 text-[16px] text-center">${v}</td>
      `;
      metricsTbody.appendChild(tr);
    }
  }

  // 2) Вероятности по классам
  const probsTbody = el("probsTbody");
  if (probsTbody) {
    probsTbody.innerHTML = "";

    const probs = payload.probabilities || {};
    const keys = Object.keys(probs);

    const ordered =
      FIXED_CLASS_ORDER.filter((k) => k in probs).concat(
        keys.filter((k) => !FIXED_CLASS_ORDER.includes(k)).sort()
      );

    ordered.forEach((label, idx) => {
      const p = Number(probs[label] || 0) * 100;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="border-2 border-black px-3 py-2 text-[16px] text-center">${idx}</td>
        <td class="border-2 border-black px-3 py-2 text-[16px] text-center">${label}</td>
        <td class="border-2 border-black px-3 py-2 text-[16px] text-center">${p.toFixed(2)}</td>
      `;
      probsTbody.appendChild(tr);
    });
  }

  // 3) Интерпретация
  const pred = payload.predicted_label || "";
  const interp = el("interpretation");
  if (interp) {
    interp.textContent = CLASS_DESCRIPTIONS[pred] || "Описание для данного класса пока не добавлено.";
  }

  // 4) Картинка (берём из выбранного файла)
  const img = el("resultImg");
  if (img) {
    if (selectedUrl) img.src = selectedUrl;
    else img.removeAttribute("src");
  }

  results.classList.remove("hidden");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runInference() {
  clearStatus();

  const modelSel = el("modelSelect");
  const modelId = modelSel?.value || "";
  if (!modelId) return setStatus("error", "Не выбран алгоритм (model_id).");

  if (!selectedFile) return setStatus("error", "Пожалуйста, сначала загрузите изображение.");

  const runBtn = el("runBtn");
  if (runBtn) runBtn.disabled = true;

  setStatus("info", "Модель выполняет прогноз...");

  const fd = new FormData();
  fd.append("model_id", modelId);
  fd.append("file", selectedFile);

  try {
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await r.json();

    if (!r.ok) {
      setStatus("error", data.error || "Ошибка при выполнении прогноза.");
      return;
    }

    setStatus("ok", "Прогноз выполнен успешно.");
    renderResults(data);

  } catch (e) {
    setStatus("error", "Сетевая ошибка. Проверьте, что backend запущен и доступен.");
  } finally {
    if (runBtn) runBtn.disabled = false;
  }
}

/**
 * ОБНОВЛЕНО:
 * При "Очистить кэш модели" теперь сбрасываем:
 * - выбранный файл (строку с именем файла + крестик)
 * - fileInput.value
 * - selectedFile/selectedUrl
 * - результаты прогноза
 */
async function clearCache() {
  const clearBtn = el("clearCacheBtn");
  const fileInput = el("fileInput");

  // 1) Сразу сбрасываем загруженное фото в UI (красный квадрат на скрине)
  if (fileInput) fileInput.value = "";
  setFile(null); // прячет fileRow, делает runBtn disabled, revoke URL, скрывает результаты

  // 2) Идём чистить кэш на backend
  clearStatus();
  setStatus("info", "Очищаем кэш модели...");

  if (clearBtn) clearBtn.disabled = true;

  try {
    const r = await fetch("/api/admin/clear-cache", { method: "POST" });
    const data = await r.json();

    if (!r.ok) {
      setStatus("error", data.error || "Не удалось очистить кэш модели.");
      return;
    }

    setStatus("ok", data.message || "Кэш очищен. Загруженный файл и результаты сброшены.");
  } catch (e) {
    setStatus("error", "Не удалось очистить кэш модели (ошибка сети).");
  } finally {
    if (clearBtn) clearBtn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  const year = el("year");
  if (year) year.textContent = new Date().getFullYear();

  setupUploader();
  await loadModels();

  const runBtn = el("runBtn");
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.addEventListener("click", runInference);
  }

  const clearCacheBtn = el("clearCacheBtn");
  if (clearCacheBtn) clearCacheBtn.addEventListener("click", clearCache);
});
