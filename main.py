from flask import Flask, request, jsonify
import werkzeug
import numpy as np
import cv2
import pickle


app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if(request.method == "POST"):
        file = request.files['files']
        filename = werkzeug.utils.secure_filename(file.filename)
        file.save("./uploadedimages/" + filename)

        p_model = pickle.load(open("SoftEngModel.pkl", "rb"))
        imgpath = "./uploadedimages/" + filename
        print(imgpath)
        img = cv2.imread(imgpath)
        resz = cv2.resize(img, (60,60))

        chck = []
        chck.append(resz / 255.0)
        chck = np.array(chck, dtype='float32')

        pprd = p_model.predict(chck)
        pprd = np.argmax(pprd, axis=1)

        if pprd[0] == 0:
            return jsonify({
                "message": "BIO"
            })
        else:
            return jsonify({
                "message": "NON-BIO"
            })
        
if __name__ == "__main__":
    app.run(debug=True, port=4000)