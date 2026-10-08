from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.schema import CompanyInput
from app.model import predict_company
from app.risk_engine import generate_risk_score, generate_risk_factors
import webbrowser
import threading

app = FastAPI(
    title="Corporate Tax Risk AI",
    description="Explainable Corporate Tax Risk Detection using Machine Learning",
    version="1.0.0"
)


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html>

<head>

<title>Corporate Tax Risk AI</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f5efe6;
    margin: 0;
    padding: 0;
}

.header {
    background: #5c4033;
    color: white;
    padding: 25px;
    text-align: center;
}

.container {
    width: 90%;
    max-width: 1100px;
    margin: 30px auto;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}

h1 {
    margin: 0;
}

h2 {
    color: #5c4033;
}

.form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
}

label {
    font-weight: bold;
    color: #4b3621;
}

input, select {
    width: 100%;
    padding: 10px;
    margin-top: 5px;
    border: 1px solid #ccc;
    border-radius: 8px;
    box-sizing: border-box;
}

button {
    margin-top: 25px;
    width: 100%;
    padding: 15px;
    background: #5c4033;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 18px;
    cursor: pointer;
}

button:hover {
    background: #3e2b22;
}

.result {
    display: none;
}

.risk-box {
    text-align: center;
    padding: 25px;
    border-radius: 12px;
    background: #f3eee8;
}

.risk-value {
    font-size: 35px;
    font-weight: bold;
}

.factor {
    padding: 10px;
    margin: 8px 0;
    background: #f5efe6;
    border-radius: 8px;
}

.probability {
    margin: 10px 0;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #777;
}

</style>

</head>


<body>


<div class="header">

<h1>Corporate Tax Risk AI</h1>

<p>Explainable Machine Learning for Corporate Tax Risk Detection</p>

</div>


<div class="container">


<div class="card">

<h2>Company Risk Assessment</h2>

<div class="form-grid">


<div>
<label>Ownership Type</label>

<select id="ownership_type">

<option>Private</option>
<option>Public</option>
<option>Government</option>
<option>Foreign Subsidiary</option>

</select>
</div>


<div>
<label>Market Region</label>

<select id="market_region">

<option>Asia</option>
<option>EU</option>
<option>North America</option>
<option>South America</option>
<option>Middle East</option>

</select>
</div>


<div>
<label>Internal Control Score</label>

<input id="internal_control_score" type="number" value="60">

</div>


<div>
<label>Effective Tax Rate (%)</label>

<input id="effective_tax_rate" type="number" step="0.01" value="15">

</div>


<div>
<label>Audit Likelihood</label>

<input id="audit_likelihood" type="number" step="0.01" value="0.5">

</div>


<div>
<label>Offshore Transactions</label>

<input id="offshore_transactions" type="number" value="2">

</div>


<div>
<label>Previous Fines</label>

<input id="previous_fines" type="number" value="100000">

</div>


<div>
<label>Board Independence</label>

<input id="board_independence" type="number" value="60">

</div>


<div>
<label>Revenue</label>

<input id="revenue" type="number" value="50000000">

</div>


<div>
<label>Profit Before Tax</label>

<input id="profit_before_tax" type="number" value="8000000">

</div>


<div>
<label>Leverage Ratio</label>

<input id="leverage_ratio" type="number" step="0.01" value="1.5">

</div>


<div>
<label>Governance Score</label>

<input id="governance_score" type="number" value="65">

</div>


<div>
<label>Restatement History</label>

<input id="restatement_history" type="number" value="0">

</div>


<div>
<label>Whistleblower Reports</label>

<input id="whistleblower_reports" type="number" value="1">

</div>


</div>


<button onclick="predictRisk()">

Analyze Corporate Tax Risk

</button>

</div>


<div class="card result" id="result">

<h2>AI Risk Assessment</h2>


<div class="risk-box">

<div>Predicted Risk</div>

<div class="risk-value" id="risk"></div>

<div>

Risk Score:

<strong id="risk_score"></strong>/100

</div>

</div>


<h3>Risk Probabilities</h3>

<div id="probabilities"></div>


<h3>Key Risk Factors</h3>

<div id="factors"></div>


</div>


</div>


<div class="footer">

Corporate Tax Risk AI • Machine Learning Prototype

</div>


<script>


async function predictRisk() {


const data = {

ownership_type:
document.getElementById("ownership_type").value,

internal_control_score:
parseFloat(document.getElementById("internal_control_score").value),

effective_tax_rate:
parseFloat(document.getElementById("effective_tax_rate").value),

audit_likelihood:
parseFloat(document.getElementById("audit_likelihood").value),

offshore_transactions:
parseInt(document.getElementById("offshore_transactions").value),

previous_fines:
parseFloat(document.getElementById("previous_fines").value),

board_independence:
parseFloat(document.getElementById("board_independence").value),

revenue:
parseFloat(document.getElementById("revenue").value),

profit_before_tax:
parseFloat(document.getElementById("profit_before_tax").value),

leverage_ratio:
parseFloat(document.getElementById("leverage_ratio").value),

market_region:
document.getElementById("market_region").value,

governance_score:
parseFloat(document.getElementById("governance_score").value),

restatement_history:
parseInt(document.getElementById("restatement_history").value),

whistleblower_reports:
parseInt(document.getElementById("whistleblower_reports").value)

};


try {


const response = await fetch("/predict", {

method: "POST",

headers: {

"Content-Type": "application/json"

},

body: JSON.stringify(data)

});


if (!response.ok) {

const error = await response.text();

alert(error);

return;

}


const result = await response.json();


document.getElementById("result").style.display = "block";


document.getElementById("risk").innerText =
result.risk;


document.getElementById("risk_score").innerText =
result.risk_score;


let probabilities = "";

for (const [key, value] of
Object.entries(result.probabilities)) {

probabilities +=

`<div class="probability">

<strong>${key}</strong> :
${(value * 100).toFixed(2)}%

</div>`;

}


document.getElementById("probabilities").innerHTML =
probabilities;


let factors = "";

result.key_risk_factors.forEach(function(factor) {

factors +=

`<div class="factor">

⚠️ ${factor}

</div>`;

});


document.getElementById("factors").innerHTML =
factors;


}

catch(error) {

alert("Prediction failed: " + error);

}

}


</script>


</body>

</html>
"""


# ---------------------------------------------------------
# PREDICTION API
# ---------------------------------------------------------

@app.post("/predict")
def predict(company: CompanyInput):

    # Convert Pydantic model to dictionary
    data = company.model_dump()

    # Pass dictionary directly.
    # model.py will create the DataFrame.
    prediction, probabilities = predict_company(data)

    # Generate risk score
    risk_score = generate_risk_score(probabilities)

    # Generate explainable risk factors
    factors = generate_risk_factors(data)

    return {
        "risk": str(prediction),
        "risk_score": risk_score,
        "probabilities": probabilities,
        "key_risk_factors": factors
    }

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8000/")

    threading.Timer(2, open_browser).start()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )