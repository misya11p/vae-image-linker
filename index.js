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

function previewImage(obj, idx)
{
	var fileReader = new FileReader();
	fileReader.onload = (function() {
  var idName = "loaded-image" + idx;
		document.getElementById(idName).src = fileReader.result;
	});
	fileReader.readAsDataURL(obj.files[0]);
}

// const API_URL = "https://muds.gdl.jp/s2122027/";

function getResult() {
  console.log("getResult");
  var image1 = document.getElementById("loaded-image1");
  var image2 = document.getElementById("loaded-image2");
  var data = {
    image1: image1.src,
    image2: image2.src,
  };
  var jsonData = JSON.stringify(data);
  fetch("http://127.0.0.1:5000/image-linker", {
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
        let result = document.getElementById("result");
        result.style.display = "block";
        window.scrollTo({
          top: 2000,
          behavior: "smooth",
        });
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
