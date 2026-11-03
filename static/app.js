(() => {
  const MAX_SIDE = 1024;
  const MAX_BYTES = 1024 * 1024;

  const shrink = (file) =>
    new Promise((resolve) => {
      if (!file.type.startsWith("image/") || file.size <= MAX_BYTES) return resolve(file);
      const img = new Image();
      const url = URL.createObjectURL(file);
      const done = (result) => {
        URL.revokeObjectURL(url);
        resolve(result);
      };
      img.onload = () => {
        const scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
        canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          (blob) => done(blob ? new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" }) : file),
          "image/jpeg",
          0.85
        );
      };
      img.onerror = () => done(file);
      img.src = url;
    });

  const setFiles = (input, file) => {
    if (typeof DataTransfer === "undefined") return false;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    return true;
  };

  const initUpload = (form) => {
    const input = form.querySelector('input[type="file"]');
    const dropzone = form.querySelector(".dropzone");
    const preview = form.querySelector("[data-preview]");
    const prompt = form.querySelector("[data-prompt]");
    const filename = form.querySelector("[data-filename]");
    const reset = form.querySelector("[data-reset]");
    const submit = form.querySelector('[type="submit"]');
    if (!input || !dropzone || !preview || !prompt || !filename || !submit) return;

    const home = form.dataset.home && new URL(form.dataset.home, location.href).pathname;
    if (home && location.pathname !== home) history.replaceState(null, "", home);

    const pickLabel = submit.textContent;
    const readyLabel = submit.dataset.readyLabel || pickLabel;

    const show = async (file) => {
      if (preview.src) URL.revokeObjectURL(preview.src);
      if (!file) {
        preview.removeAttribute("src");
        preview.hidden = filename.hidden = true;
        prompt.hidden = false;
        if (reset) reset.hidden = true;
        submit.textContent = pickLabel;
        return;
      }
      const small = await shrink(file);
      if (small !== file) setFiles(input, small);
      preview.src = URL.createObjectURL(small);
      preview.alt = "Preview of " + file.name;
      filename.textContent = file.name;
      preview.hidden = filename.hidden = false;
      prompt.hidden = true;
      if (reset) reset.hidden = false;
      submit.textContent = readyLabel;
    };

    input.addEventListener("change", () => show(input.files[0]));

    if (reset) {
      reset.addEventListener("click", () => {
        input.value = "";
        show(null);
        input.focus();
      });
    }

    const toggleDrag = (on) => (e) => {
      e.preventDefault();
      dropzone.classList.toggle("is-dragover", on);
    };
    dropzone.addEventListener("dragenter", toggleDrag(true));
    dropzone.addEventListener("dragover", toggleDrag(true));
    dropzone.addEventListener("dragleave", toggleDrag(false));
    dropzone.addEventListener("drop", (e) => {
      toggleDrag(false)(e);
      const file = e.dataTransfer.files[0];
      if (file && setFiles(input, file)) show(file);
    });

    submit.addEventListener("click", (e) => {
      if (input.files.length) return;
      e.preventDefault();
      input.click();
    });

    form.addEventListener("submit", () => {
      submit.setAttribute("aria-busy", "true");
      if (submit.dataset.busyLabel) submit.textContent = submit.dataset.busyLabel;
    });
  };

  document.querySelectorAll("form[data-upload]").forEach(initUpload);
})();
