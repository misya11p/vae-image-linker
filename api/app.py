from flask import Flask, request
from flask_cors import CORS
from link_imaes import link_images

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


CONFIG = {
    "model_path": "models/model.pth",
    "n_frames": 10,
    "device": "cpu",
    "image_size": 128
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

    result = link_images(image1, image2, CONFIG)
    print(type(result))
    print(len(result))
    return {"result": "success"}


if __name__ == "__main__":
    app.run(debug=True)
