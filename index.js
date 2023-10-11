const canvas = new fabric.Canvas("canvas");

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
		document.getElementById('loadedImage').src = fileReader.result;
	});
	fileReader.readAsDataURL(obj.files[0]);
}

const API_URL = "https://muds.gdl.jp/s2122027/";
fetch(API_URL)
  .then((data) => data.text())
  .then((res) => console.log(res));


function getResult() {
  var image1 = document.getElementById("canvas");
  var blob1 = Base64toBlob(image2.toDataURL());
  var image2 = document.getElementById("loadedImage");
  var blob2 = Base64toBlob(image1.src);
  var formData = new FormData();
  formData.append("image1", blob1);
  formData.append("image2", blob2);

  fetch(API_URL, {
    method: "POST",
    body: formData,
  })
    .then((data) => data.text())
    .then((res) => {
      console.log(res);
      document.getElementById("result").innerHTML = res;
    });
}
