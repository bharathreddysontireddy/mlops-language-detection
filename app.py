from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running successfully'})

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        text_input = request.form.get('text')
        
        if not text_input or text_input.strip() == "":
            return render_template('index.html', prediction_text="Error: Please enter some text.")
            
        # Predict language
        predicted_lang = model.predict([text_input])[0]
        
        return render_template('index.html', 
                               prediction_text=f'Detected Language: {predicted_lang}',
                               user_input=text_input)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)