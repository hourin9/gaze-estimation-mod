from flask import Flask, Response;

app = Flask(__name__);

@app.route("/")
def test():
    return "<p> con cho pbl4 </p>";

app.run(host="0.0.0.0", threaded=True);

