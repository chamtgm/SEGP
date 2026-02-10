
        const videoElement = document.getElementById('videoElement');
        const startButton = document.getElementById('startCamera');
        const stopButton = document.getElementById('stopCamera');
        const takeButton = document.getElementById('takePhotoBtn');
        const downloadLink = document.getElementById('downloadLink');
        const canvas = document.getElementById('hiddenCanvas');
        const photoImg = document.getElementById('photoTaken');
        let stream = null;

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { facingMode: 'environment' },
                    audio: false
                });
                videoElement.srcObject = stream;
                startButton.disabled = true;
                stopButton.disabled = false;
                takeButton.disabled = false;
                downloadLink.style.display = 'none';
            } catch (err) {
                console.error('Error accessing camera:', err);
                alert('Could not access the camera. Please make sure you have granted permission and that your site is served over HTTPS or localhost.');
            }
        }

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                videoElement.srcObject = null;
                stream = null;
                startButton.disabled = false;
                stopButton.disabled = true;
                takeButton.disabled = true;
            }
        }

        function takePhoto() {
            if (!stream) {
                alert('Camera is not started. Click "Start Camera" first.');
                return;
            }

            // Use video element's intrinsic size if available, otherwise fall back to 640x480
            const width = videoElement.videoWidth || 640;
            const height = videoElement.videoHeight || 480;

            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');


            ctx.drawImage(videoElement, 0, 0, width, height);

            // Create data URL and show preview
            const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
            photoImg.src = dataUrl;
            photoImg.alt = 'Captured photo';
            saveToHistory(dataUrl); 
            console.log("takePhoto called");

            // Create download link
            downloadLink.href = dataUrl;
            downloadLink.download = 'photo.jpg';
            downloadLink.style.display = 'inline-block';
            downloadLink.textContent = 'Download Photo';

            saveToHistory(dataUrl);
        }

        // Wire up buttons
        startButton.addEventListener('click', startCamera);
        stopButton.addEventListener('click', stopCamera);
        takeButton.addEventListener('click', takePhoto);

        // Initial button states
        stopButton.disabled = true;
        takeButton.disabled = true;
    
       // =====================

       function saveToHistory(imageData) {
        console.log("Saving to history...");
    console.log(imageData);
    let history = JSON.parse(localStorage.getItem("imageHistory")) || [];
    history.unshift(imageData); // newest first
    localStorage.setItem("imageHistory", JSON.stringify(history));
}
