from flask import Flask, request
from .link_imaes import link_images

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/image-linker", methods=["POST"])
def image_linker():
    image1 = request.files["image1"]
    image2 = request.files["image2"]
    result = link_images(image1, image2)
    return result
