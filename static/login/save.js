document.addEventListener("DOMContentLoaded", function () {
    const screenshotBtn = document.getElementById("screenshotBtn");
    const contentDiv = document.getElementById("content");

    screenshotBtn.addEventListener("click", function () {
      html2canvas(contentDiv).then(function (canvas) {
        // Create an "a" element to trigger the download
        const downloadLink = document.createElement("a");
        downloadLink.href = canvas.toDataURL("image/png");
        downloadLink.download = "screenshot.png";
        // Trigger the click event on the download link
        downloadLink.click();
      });
    });
  });