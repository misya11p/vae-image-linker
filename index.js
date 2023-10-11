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
		document.getElementById('preview').src = fileReader.result;
	});
	fileReader.readAsDataURL(obj.files[0]);
}
