<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        header {
            background-color: #4CAF50;
            padding: 20px;
            text-align: center;
            color: white;
        }
        main {
            margin: 20px;
            padding: 20px;
            background-color: white;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 600px;
            text-align: center;
        }
        h1 {
            font-size: 24px;
            color: #333;
        }
        #image-preview {
            width: 100%;
            max-width: 300px;
            height: 300px;
            border: 2px dashed #ccc;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px auto;
        }
        #image-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        input[type="file"] {
            display: none;
        }
        .custom-file-upload {
            border: 1px solid #ccc;
            display: inline-block;
            padding: 8px 12px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            margin-top: 10px;
        }
        button {
            padding: 10px 20px;
            margin-top: 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .prediction-result {
            margin-top: 20px;
            font-size: 18px;
            color: #333;
        }
    </style>
</head>
<body>

    <header>
        <h1>Face Recognition Prediction</h1>
    </header>

    <main>
        <h1>Upload Your Image</h1>
        
        <div id="image-preview">
            <p>No Image Selected</p>
        </div>

        <label for="imageUpload" class="custom-file-upload">
            Choose Image
        </label>
        <input type="file" id="imageUpload" accept="image/*" onchange="previewImage(event)" />
        
        <button onclick="predictFace()">Predict</button>

        <div class="prediction-result" id="predictionResult">
            <!-- Prediction result will be displayed here -->
        </div>
    </main>

    <script>
        function previewImage(event) {
            const imagePreview = document.getElementById('image-preview');
            imagePreview.innerHTML = '';
            const img = document.createElement('img');
            img.src = URL.createObjectURL(event.target.files[0]);
            imagePreview.appendChild(img);
        }

        function predictFace() {
            // This is where you would send the image to the machine learning model
            // For now, we'll just display a mock result.
            const predictionResult = document.getElementById('predictionResult');
            predictionResult.innerHTML = "Prediction: Face recognized as John Doe (Confidence: 95%)";
        }
    </script>
    
</body>
</html>
