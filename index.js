var images = []
var imageLoaded = false;
var isUpload = [false, false]


function previewImage(obj, idx)
{
	var fileReader = new FileReader();
	fileReader.onload = (function() {
  var idName = "loaded-image" + idx;
		document.getElementById(idName).src = fileReader.result;
	});
	fileReader.readAsDataURL(obj.files[0]);
  isUpload[idx - 1] = true;
}

const API_URL = "https://muds.gdl.jp/s2122027/image-linker";
// const API_URL = "http://127.0.0.1:5000/image-linker";

function scroll() {
  window.scrollTo({
    top: 2000,
    behavior: "smooth",
  });
}

function getResult() {
  console.log("getResult");
  let message = document.getElementById("message");
  message.style.display = "block";
  var image1 = document.getElementById("loaded-image1");
  var image2 = document.getElementById("loaded-image2");
  if (!isUpload[0] || !isUpload[1]) {
    message.innerHTML = "Please upload images.";
    scroll();
    return;
  }
  var data = {
    image1: image1.src,
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
        message.style.display = "none";
        images = data.images;
        imageLoaded = true;
        let result = document.getElementById("result");
        result.style.display = "block";
        scroll();
        updateImage();
      } else {
        message.innerHTML = "Error";
      }
    })
    .catch((err) => {
      console.log(err);
      message.innerHTML = "Error";
      scroll();
    });
  message.innerHTML = "Generating... It may take a while the first time.";
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
