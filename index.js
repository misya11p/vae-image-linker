const canvas = new fabric.Canvas("canvas");
var images = []
var imageLoaded = false;


document.getElementById("draw").addEventListener("click", function () {
  canvas.freeDrawingBrush = new fabric.PencilBrush(canvas);
  canvas.freeDrawingBrush.width=5;
  canvas.freeDrawingBrush.color="black";
  canvas.isDrawingMode = true;
});

canvas.backgroundColor="white";

document.getElementById("erase").addEventListener("click", function () {
  canvas.freeDrawingBrush = new fabric.PencilBrush(canvas);
  canvas.freeDrawingBrush.width=10;
  canvas.freeDrawingBrush.color="white";
  canvas.isDrawingMode = true;
});

function previewImage(obj)
{
	var fileReader = new FileReader();
	fileReader.onload = (function() {
		document.getElementById('loaded-image').src = fileReader.result;
	});
	fileReader.readAsDataURL(obj.files[0]);
}

// const API_URL = "https://muds.gdl.jp/s2122027/";
const API_URL = "http://127.0.0.1:5000/image-linker";
// const API_URL = "http://127.0.0.1:8000/";
// fetch(API_URL)
//   .then((data) => data.text())
//   .then((res) => console.log(res));


function getResult() {
  console.log("getResult");
  var image1 = document.getElementById("canvas");
  var image2 = document.getElementById("loaded-image");
  var data = {
    image1: image1.toDataURL("image/png"),
    image2: image2.src,
  };
  var jsonData = JSON.stringify(data);

  fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: jsonData,
  })
    .then((data) => data.text())
    .then((res) => {
      let data = JSON.parse(res);
      console.log(data.status);
      if (data.status == "success") {
        images = data.images;
        imageLoaded = true;
        updateImage();
      }
    });
}

function updateImage() {
  if (imageLoaded) {
    var slider = document.getElementById("image-slider");
    var idx = Number(slider.value);
    var image = document.getElementById("result-image");
    image.src = "data:image/png;base64," + images[idx - 1];
  }
}
let slider = document.getElementById("image-slider");
slider.addEventListener("input", updateImage);
