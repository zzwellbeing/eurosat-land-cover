const element = (id) => document.getElementById(id);
const input = element("image");
const image = element("preview");
const button = element("predict");
const reset = element("reset");
const result = element("result");
const allowed = new Set(["image/jpeg", "image/png", "image/webp"]);
const percent = (value) => (value * 100).toFixed(1) + "%";
let file = null;
let previewUrl = null;
let ready = false;
let loading = false;

function setStatus(message, error = false) {
  element("status").textContent = message;
  element("status").classList.toggle("error", error);
}

function updateButtons() {
  button.disabled = !file || !ready || loading;
  reset.disabled = !file || loading;
  input.disabled = loading;
}

function clearSelection() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  file = null;
  image.removeAttribute("src");
  image.hidden = true;
  element("placeholder").hidden = false;
  element("filename").textContent = "请选择一张影像";
  result.hidden = true;
  updateButtons();
}

input.addEventListener("change", () => {
  const candidate = input.files[0];
  clearSelection();
  if (!candidate) return;
  if (!allowed.has(candidate.type) || candidate.size > 5 * 1024 * 1024) {
    input.value = "";
    setStatus("请选择不超过 5 MB 的 JPG、PNG 或 WebP 图片。", true);
    return;
  }
  file = candidate;
  previewUrl = URL.createObjectURL(file);
  image.src = previewUrl;
  image.hidden = false;
  element("placeholder").hidden = true;
  element("filename").textContent = file.name;
  setStatus("已选择图片，可以开始预测。");
  updateButtons();
});

reset.addEventListener("click", () => {
  input.value = "";
  clearSelection();
  setStatus("请选择新的影像。");
  input.click();
});

button.addEventListener("click", async () => {
  loading = true;
  result.hidden = true;
  updateButtons();
  setStatus("正在运行 PyTorch 推理…");
  const form = new FormData();
  form.append("image", file);
  try {
    const response = await fetch("/api/predict", {method: "POST", body: form});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "预测失败，请重试。");
    const top = body.top3[0];
    element("top-label").textContent = top.label;
    element("top-score").textContent = percent(top.probability);
    element("timing").textContent = body.inference_ms.toFixed(2) + " ms";
    element("ranks").replaceChildren(...body.top3.map((item) => {
      const row = document.createElement("li");
      const title = document.createElement("span");
      const value = document.createElement("span");
      const meter = document.createElement("progress");
      title.textContent = item.label;
      value.textContent = percent(item.probability);
      meter.max = 1;
      meter.value = item.probability;
      meter.setAttribute("aria-label", item.label + "预测概率");
      row.append(title, value, meter);
      return row;
    }));
    result.hidden = false;
    setStatus("预测完成。");
  } catch (error) {
    setStatus(error.message || "无法连接推理服务，请重试。", true);
  } finally {
    loading = false;
    updateButtons();
  }
});

try {
  const response = await fetch("/api/model");
  if (!response.ok) throw new Error("模型服务未就绪");
  const model = await response.json();
  ready = model.status === "ready";
  element("model").textContent = model.name + " · " + model.experiment + " · Epoch " + model.epoch + " · 已就绪";
} catch {
  element("model").textContent = "模型服务未连接，请检查后端。";
}
updateButtons();
