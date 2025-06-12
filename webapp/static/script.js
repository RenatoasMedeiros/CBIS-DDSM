document.getElementById("upload-form").addEventListener("submit", function (e) {
  e.preventDefault();

  const fileInput = document.getElementById("file-input");
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  // Show preview of uploaded image
  const reader = new FileReader();
  reader.onload = function (e) {
    const img = document.getElementById("uploaded-image");
    img.src = e.target.result;
    img.style.display = "block";
  };
  reader.readAsDataURL(file);

  // Clear previous results
  const output = document.getElementById("result-text");
  output.textContent = "";
  const annotatedImage = document.getElementById("annotated-image");
  annotatedImage.style.display = "none";
  annotatedImage.src = "";

  // Send file to backend
  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        output.textContent = `Error: ${data.error}`;
        return;
      }

      let resultText = `Malignant Probability: ${(data.malignant_probability * 100).toFixed(2)}%\n`;

      if (data.detection_triggered) {
        resultText += "Object detection was triggered.\n";
        resultText += `Detections:\n${JSON.stringify(data.detection_results, null, 2)}\n`;

        if (data.annotated_image_base64) {
          annotatedImage.src = "data:image/jpeg;base64," + data.annotated_image_base64;
          annotatedImage.style.display = "block";
        }
      } else {
        resultText += "No suspicious objects detected.";
      }
      output.textContent = resultText;
    })
    .catch((error) => {
      console.error("Error:", error);
      output.textContent = "Request failed: " + error.message;
    });
});
