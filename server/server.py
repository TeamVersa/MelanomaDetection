from flask import Flask, jsonify, request
import time

app = Flask(__name__)

@app.route('/melanoma', methods=['POST'])
def get_tasks():
    i = request.files.get('image', '')
    print(f'received image {i}')
    i.save('image.jpg')
    # time.sleep(2.0)
    return jsonify({
        'prediction': 0.69,
        'weather': 'nice'
    })

if __name__ == '__main__':
    print('hello world!')
    app.run(debug=True, host='0.0.0.0')