const el = (id) => document.getElementById(id);

const CLASS_DESCRIPTIONS = {
  NILM:
    "NILM (Negative for Intraepithelial Lesion or Malignancy) — цитологический результат ПАП-теста, " +
    "означающий отсутствие признаков внутриэпителиального поражения и злокачественности: " +
    "не выявлены раковые клетки или другие атипичные клетки эпителия шейки матки. " +
    "Возможны сопутствующие реактивные/воспалительные изменения без признаков предрака.",

  LSIL:
    "LSIL (Low-grade Squamous Intraepithelial Lesion) — низкодифференцированное плоскоклеточное " +
    "внутриэпителиальное поражение: лёгкие (умеренно выраженные) цитологические атипии плоского эпителия, " +
    "часто ассоциированные с ВПЧ-инфекцией; обычно соответствует CIN 1 (лёгкая дисплазия). " +
    "Многие случаи регрессируют самостоятельно, но требуют наблюдения по клиническим рекомендациям.",

  HSIL:
    "HSIL (High-grade Squamous Intraepithelial Lesion) — высокодифференцированное плоскоклеточное " +
    "внутриэпителиальное поражение: выраженные цитологические атипии плоского эпителия, " +
    "соответствующие более высокому риску значимого предрака; обычно соответствует CIN 2–CIN 3 " +
    "(умеренная/тяжёлая дисплазия, включая carcinoma in situ). " +
    "Результат требует прицельной верификации (как правило, кольпоскопия/биопсия) по протоколам.",

  SCC:
    "SCC (Squamous Cell Carcinoma) — плоскоклеточный рак: злокачественная опухоль, " +
    "происходящая из плоских клеток эпителия шейки матки. " +
    "В цитологическом заключении категория 'SCC' означает признаки, соответствующие раку, " +
    "и требует срочного уточнения диагноза в специализированном порядке.",
};

const FIXED_CLASS_ORDER = ["HSIL", "LSIL", "NILM", "SCC"];

let selectedFile = null;
let selectedUrl = null;

function setStatus(kind, msg) {
  const box = el("status");
  if (!box) return;

  box.classList.remove("d-none", "alert-success", "alert-danger", "alert-secondary");
  box.classList.add("alert");

  if (kind === "error") box.classList.add("alert-danger");
  else if (kind === "ok") box.classList.add("alert-success");
  else box.classList.add("alert-secondary");

  box.textContent = msg;
}

function clearStatus() {
  const box = el("status");
  if (!box) return;
  box.classList.add("d-none");
  box.textContent = "";
}

function clearResultsUI() {
  const results = el("results");
  if (results) results.classList.add("d-none");

  const placeholder = el("resultsPlaceholder");
  if (placeholder) placeholder.classList.remove("d-none");

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
    if (row) row.classList.add("d-none");
    if (name) name.textContent = "";
    if (runBtn) runBtn.disabled = true;

    if (selectedUrl) URL.revokeObjectURL(selectedUrl);
    selectedUrl = null;

    clearResultsUI();
    return;
  }

  if (selectedUrl) URL.revokeObjectURL(selectedUrl);
  selectedUrl = URL.createObjectURL(selectedFile);

  if (row) row.classList.remove("d-none");
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
    dropzone.classList.add("border-primary");
    dropzone.classList.remove("bg-light");
    dropzone.classList.add("bg-white");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("border-primary");
    dropzone.classList.remove("bg-white");
    dropzone.classList.add("bg-light");
  });

  dropzone.addEventListener("drop", (e) => {
    dropzone.classList.remove("border-primary");
    dropzone.classList.remove("bg-white");
    dropzone.classList.add("bg-light");
    clearStatus();

    const f = e.dataTransfer?.files?.[0];
    if (!f) return;

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
    if (!r.ok) throw new Error("bad response");

    const data = await r.json();
    const models = data.models || [];

    sel.innerHTML = "";

    if (models.length === 0) {
      sel.innerHTML = `<option value="">Список моделей пуст</option>`;
      setStatus("error", "Backend вернул пустой список моделей (/api/models).");
      return;
    }

    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.model_id;
      opt.textContent = m.display_name;
      sel.appendChild(opt);
    }

    const preferred = "cc_vit_sts_tested";
    const hasPreferred = Array.from(sel.options).some((o) => o.value === preferred);
    sel.value = hasPreferred ? preferred : (sel.options[0]?.value || "");

  } catch (e) {
    sel.innerHTML = `<option value="">Не удалось загрузить список моделей</option>`;
    setStatus("error", "Не удалось получить список моделей (/api/models). Проверь backend.");
  }
}

function renderResults(payload) {
  const results = el("results");
  if (!results) return;

  const placeholder = el("resultsPlaceholder");
  if (placeholder) placeholder.classList.add("d-none");

  // 1) Метрики
  const metricsTbody = el("metricsTbody");
  if (metricsTbody) {
    metricsTbody.innerHTML = "";

    const elapsedSec = (Number(payload.latency_ms || 0) / 1000).toFixed(3) + " сек";
    const conf = (Number(payload.predicted_probability || 0) * 100).toFixed(2) + " %";
    const pred = payload.predicted_label || "-";

    const rows = [
      ["1", "Время на прогноз", elapsedSec],
      ["2", "Уверенность", conf],
      ["3", "Предсказанный класс", pred],
    ];

    for (const [n, k, v] of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="text-center">${n}</td>
        <td class="text-center">${k}</td>
        <td class="text-center">${v}</td>
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

    const ordered = FIXED_CLASS_ORDER.filter((k) => k in probs).concat(
      keys.filter((k) => !FIXED_CLASS_ORDER.includes(k)).sort()
    );

    ordered.forEach((label, idx) => {
      const p = Number(probs[label] || 0) * 100;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="text-center">${idx}</td>
        <td class="text-center">${label}</td>
        <td class="text-center">${p.toFixed(2)}</td>
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

  // 4) Картинка
  const img = el("resultImg");
  if (img) {
    if (selectedUrl) img.src = selectedUrl;
    else img.removeAttribute("src");
  }

  results.classList.remove("d-none");
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

async function clearCache() {
  const clearBtn = el("clearCacheBtn");
  const fileInput = el("fileInput");

  if (fileInput) fileInput.value = "";
  setFile(null);

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

  // если это не страница прогноза — тихо выходим
  if (!el("modelSelect") || !el("fileInput") || !el("runBtn")) return;

  setupUploader();
  await loadModels();

  const runBtn = el("runBtn");
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.addEventListener("click", runInference);
  }

  const clearCacheBtn = el("clearCacheBtn");
  if (clearCacheBtn) clearCacheBtn.addEventListener("click", clearCache);

  // старт: нет файла — прячем результаты
  clearResultsUI();
});
