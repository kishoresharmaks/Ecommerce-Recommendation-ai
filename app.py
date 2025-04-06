from flask import Flask, render_template
import main
import os

app = Flask(__name__)

@app.route('/')
def home():
    
    main.main()  
    
    with open('recommendations.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

if __name__ == '__main__':
    app.run(debug=True)
