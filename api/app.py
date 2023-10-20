from flask import Flask, request
from flask_cors import CORS
from link_imaes import link_images

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


CONFIG = {
    "model_path": "models/vae1.pth",
    "n_frames": 10,
    "device": "cpu",
    "image_size": 96
}

@app.route("/image-linker", methods=["POST"])
def image_linker():
    try:
        data = request.get_json()
    except Exception as e:
        print(e)
        return {"result": "error"}

    # Images of base64
    image1 = data["image1"]
    image2 = data["image2"]

    images = link_images(image1, image2, CONFIG) # List of base64 images
    return {
        "status": "success",
        "images": images
    }


if __name__ == "__main__":
    app.run(debug=True)
