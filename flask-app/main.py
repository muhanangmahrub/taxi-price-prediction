from flask import Flask, request, jsonify
from google.cloud import aiplatform

app = Flask(__name__)

def predict_instance(project_id, endpoint_id, instance):
    endpoint = aiplatform.Endpoint('projects/{}/locations/us-central1/endpoints/{}'.format(project_id, endpoint_id))
    instances_list = [instance]
    prediction = endpoint.predict(instances_list)
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True) 
    instance = data['instance']
    endpoint_id = 748064830585307136
    project_id = 141982122440
    prediction = predict_instance(project_id, endpoint_id, instance)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)

# curl -X POST -H "Content-Type: application/json" -d '{"instance": [25.61,0,0,3.0,1,2,3.79,1.56,0.29,79.89,0,0,0,0,0,0,0,0]}' https://taxiprice-online-predict-141982122440.us-central1.run.app/predict
