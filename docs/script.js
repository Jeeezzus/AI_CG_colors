let modelSession;  // Global variable to store the ONNX model session
let drawingCanvas, drawingCtx;  // Canvas and context to store only the drawing
let backgroundImage;

document.addEventListener("DOMContentLoaded", async function() {
    const canvas = document.getElementById('inputCanvas');
    const ctx = canvas.getContext('2d');
    const outputImage = document.getElementById('outputImage');

    // Create an in-memory canvas for drawing
    drawingCanvas = document.createElement('canvas');
    drawingCanvas.width = 512;
    drawingCanvas.height = 512;
    drawingCtx = drawingCanvas.getContext('2d');  // Corrected scope

    // Load the background image
    backgroundImage = new Image();  // Adjusted scope
    backgroundImage.src = 'sample_images/robot.png';  // Replace with your image path
    backgroundImage.onload = () => {
        ctx.drawImage(backgroundImage, 0, 0, 512, 512);
    };

    // Setup color, line width, and line cap style
    const colorPicker = document.getElementById('colorPicker');
    const lineWidthSlider = document.getElementById('lineWidthSlider');

    ctx.strokeStyle = colorPicker.value;
    ctx.lineWidth = lineWidthSlider.value;
    ctx.lineCap = 'round';  // Set line cap style to round

    drawingCtx.strokeStyle = ctx.strokeStyle;
    drawingCtx.lineWidth = ctx.lineWidth;
    drawingCtx.lineCap = 'round';  // Set line cap style to round for drawing canvas

    colorPicker.addEventListener('input', function() {
        ctx.strokeStyle = colorPicker.value;
        drawingCtx.strokeStyle = colorPicker.value;
    });

    lineWidthSlider.addEventListener('input', function() {
        ctx.lineWidth = lineWidthSlider.value;
        drawingCtx.lineWidth = lineWidthSlider.value;
    });

    let mode = 'draw'; // Current mode: 'draw' or 'erase'

    // Setup buttons
    const drawButton = document.getElementById('drawButton');
    const eraseButton = document.getElementById('eraseButton');

    drawButton.addEventListener('click', function() {
        mode = 'draw';
        ctx.globalCompositeOperation = 'source-over'; // Default drawing mode
        drawingCtx.globalCompositeOperation = 'source-over'; // Same for the drawing canvas
    });

    eraseButton.addEventListener('click', function() {
        mode = 'erase';
        ctx.globalCompositeOperation = 'destination-out'; // Eraser mode
        drawingCtx.globalCompositeOperation = 'destination-out'; // Same for the drawing canvas
    });

    // Drawing functionality
    let isDrawing = false;
    let lastX, lastY; // Track the last X and Y positions

    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        if (mode === 'draw') {
            ctx.strokeStyle = colorPicker.value; // Use the selected color for drawing
            drawingCtx.strokeStyle = colorPicker.value;
        } else {
            ctx.strokeStyle = 'rgba(0,0,0,1)'; // Eraser mode: set color to transparent
            drawingCtx.strokeStyle = 'rgba(0,0,0,1)';
        }
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();

        // Mirror the drawing on the in-memory canvas
        drawingCtx.beginPath();
        drawingCtx.moveTo(lastX, lastY);
        drawingCtx.lineTo(e.offsetX, e.offsetY);
        drawingCtx.stroke();

        [lastX, lastY] = [e.offsetX, e.offsetY];
    });

    canvas.addEventListener('mouseup', () => isDrawing = false);
    canvas.addEventListener('mouseout', () => isDrawing = false);

    // Load the ONNX model
    modelSession = await ort.InferenceSession.create('onnx_models/auto1_float.onnx');
    console.log("Model successfully loaded");  // Log when model is loaded

    // Reset button functionality
    const resetButton = document.getElementById('resetButton');
    resetButton.addEventListener('click', function() {
        // Clear the visible canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(backgroundImage, 0, 0, 512, 512); // Redraw the background image

        // Clear the drawing canvas
        drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    });

    // Add event listener to show the drawing
    document.getElementById('test').addEventListener('click', function() {
        outputImage.src = drawingCanvas.toDataURL();
    });
});

// Function to preprocess the canvas data
function preprocessCanvasData(drawingCanvas, backgroundImageSrc) {
    // Create a new canvas to compose the 4-channel image
    const composedCanvas = document.createElement('canvas');
    composedCanvas.width = 512;
    composedCanvas.height = 512;
    const composedCtx = composedCanvas.getContext('2d');

    // Draw the drawing (RGB) on the composed canvas
    composedCtx.drawImage(drawingCanvas, 0, 0);

    // Create a temporary canvas to draw the background image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 512;
    tempCanvas.height = 512;
    const tempCtx = tempCanvas.getContext('2d');

    // Draw the background image onto the temporary canvas
    const tempImage = new Image();
    tempImage.src = backgroundImageSrc;
    tempCtx.drawImage(tempImage, 0, 0, 512, 512);

    // Get the image data from both canvases
    const drawingData = composedCtx.getImageData(0, 0, 512, 512);
    const backgroundData = tempCtx.getImageData(0, 0, 512, 512);

    // Combine the drawing and background data into the drawingData's alpha channel
    for (let i = 0; i < drawingData.data.length; i += 4) {
        // Assuming the background is grayscale, use any color channel for the alpha value
        drawingData.data[i + 3] = backgroundData.data[i]; // Set alpha channel to the grayscale value
    }

    composedCtx.putImageData(drawingData, 0, 0);

    const imageData = composedCtx.getImageData(0, 0, 512, 512).data;

    // Flatten the image data into a tensor-like structure
    let flattenedData = [];
    for (let i = 0; i < imageData.length; i += 4) {
         // Place the alpha (lineart) channel first
         flattenedData.push(imageData[i + 3]/255); // Alpha (from the background) as the first channel
         // Followed by the RGB channels
         flattenedData.push(imageData[i]/255);     // Red
         flattenedData.push(imageData[i + 1]/255); // Green
         flattenedData.push(imageData[i + 2]/255); // Blue
    }

    return flattenedData;
}



// Function to run the model
async function runModel() {
    // Preprocess the canvas data into the correct format (4 channels)
    const preprocessedData = preprocessCanvasData(drawingCanvas, 'sample_images/robot.png');
    const inputTensor = new ort.Tensor('float32', preprocessedData, [1, 4, 512, 512]);

    // Run the model
    const feeds = { 'input.1': inputTensor };
    const outputData = await modelSession.run(feeds);

    const outputTensor = outputData['16'];

    // Postprocess and display the output
    const outputCanvas = postprocessOutput(outputTensor);
    const outputImg = document.getElementById('outputImage');
    outputImg.src = outputCanvas.toDataURL();
}

// Function to postprocess the model output and display it
function postprocessOutput(outputTensor) {
    // Get the dimensions of the output tensor
    const [batch, channels, height, width] = outputTensor.dims;

    // Check that the output tensor has the expected shape
    if (batch !== 1 || channels !== 3 || height !== 512 || width !== 512) {
        console.error('Unexpected output dimensions:', outputTensor.dims);
        return;
    }
// Use tensorToImageData to convert the tensor data to an ImageData object
const outputCanvas = tensorToImageData(outputTensor.data, width, height);

// Return the canvas
return outputCanvas;
}

function tensorToImageData(tensor, width, height) {
    // Create a new canvas element
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Create a new ImageData object
    let imageData = ctx.createImageData(width, height);

    // Assuming tensor is a Float32Array with normalized values [0, 1]
    // The data is also assumed to be in [channels, height, width] order
    for (let c = 0; c < 3; c++) { // Loop over the channels: R, G, B
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // Calculate the indices in the tensor and in the ImageData
                let tensorIndex = c * (width * height) + y * width + x;
                let imageDataIndex = (y * width + x) * 4 + c;

                // Assign the pixel data
                imageData.data[imageDataIndex] = tensor[tensorIndex+1] * 255; // Scale to [0, 255]
            }
        }
    }

    // Set alpha channel to fully opaque
    for (let i = 3; i < imageData.data.length; i += 4) {
        imageData.data[i] = 255;
    }

    // Put the ImageData object onto the canvas
    ctx.putImageData(imageData, 0, 0);

    // Return the canvas
    return canvas;
}



document.getElementById('runModelButton').addEventListener('click', runModel);
