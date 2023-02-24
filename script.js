const input = document.getElementById('file-input');
const preview = document.getElementById('preview');
const uploadButton = document.getElementById('upload-button');
uploadButton.addEventListener('click', function () {
  const reader = new FileReader();
  reader.onload = function () {
    preview.src = reader.result;
    preview.classList.remove('hidden');
  };
  reader.readAsDataURL(input.files[0]);
});
document.getElementById('start-button').addEventListener('click', () => {
  const resultElement = document.getElementById('result');

  if (!window.EyeDropper) {
    resultElement.textContent = 'Your browser does not support the EyeDropper API';
    return;
  }

  const eyeDropper = new EyeDropper();

  eyeDropper.open().then((result) => {
    resultElement.textContent = result.sRGBHex;
    resultElement.style.backgroundColor = result.sRGBHex;
  }).catch((e) => {
    resultElement.textContent = e;
  });
});

/*
Eye dropper 
This code adds a click event listener to an element with the id "start-button". 
When the button is clicked, it first checks if the "EyeDropper" API is supported 
by the user's browser. If not, it updates the text content of an element with 
the id "result" with a message indicating that the API is not supported. 
If the API is supported, it creates an instance of the EyeDropper API and 
opens it. The promise returned by the open() method either resolves 
with the result of the eye dropper tool, which is an object containing the 
sRGBHex value of the selected color, or rejects with an error. 
If the promise resolves, the text content and background color of the "result" element 
are updated with the sRGBHex value. 
If the promise rejects, the error is displayed as the text content of the "result" element.
*/