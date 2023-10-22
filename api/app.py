from flask import Flask, request
from flask_cors import CORS
from link_images import link_images

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


CONFIG = {
    "model_path": "./models/vae3.pth",
    "n_frames": 20,
    "device": "cpu",
    "image_size": 96,
    "z_dim": 512
}

@app.route("/image-linker", methods=["POST"])
def image_linker():
    try:
        data = request.get_json()
        image1 = data["image1"] # Images of base64
        image2 = data["image2"]
    except Exception as e:
        print(e)
        return {
            "status": "dataLoadError",
            "message": e
        }

    try:
        images = link_images(image1, image2, CONFIG) # List of base64 images
        return {
            "status": "success",
            "images": images
        }
    except Exception as e:
        print(e)
        return {
            "status": "linkImagesError",
            "message": e
        }


if __name__ == "__main__":
    app.run(debug=True)
