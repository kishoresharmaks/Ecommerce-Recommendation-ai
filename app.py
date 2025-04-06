from flask import Flask, render_template
import main  

app = Flask(__name__)

@app.route('/')
def home():
    main.generate_recommendations()  
    return render_template('recommendations.html')  

if __name__ == '__main__':
    app.run(debug=True)