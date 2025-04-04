from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=["GET"])
def form():
    # Load dropdown options from Excel
    dropdown_df = pd.read_excel("data/UI_sheet.xlsx", sheet_name="Dropdown_Options")
    
    # Create a dictionary of options by field name
    dropdown_options = {}
    for field in dropdown_df["Field Name"].unique():
        dropdown_options[field] = dropdown_df[dropdown_df["Field Name"] == field]["Option Value"].dropna().tolist()
    
    # Pass the entire dictionary to the template
    return render_template(
        "form.html",
        dropdown_options=dropdown_options,
        subcategory_list=dropdown_options.get("subcategory", [])
    )

@app.route("/submit", methods=["POST"])
def submit():
    selected_subcategory = request.form.get("subcategory")
    selected_region = request.form.get("region")
    sourcing_stage = request.form.get("sourcing_stage")
    contract_due_date = request.form.get("contract_due_date")
    financial_impact = request.form.get("financial_impact")
    supply_risk = request.form.get("supply_risk")
    market_volatility = request.form.get("market_volatility")
    supplier_dependency = request.form.get("supplier_dependency")
    
    # Determine Kraljic Quadrant
    if financial_impact == "High" and supply_risk == "High":
        kraljic_quadrant = "Strategic"
    elif financial_impact == "High" and supply_risk == "Low":
        kraljic_quadrant = "Leverage"
    elif financial_impact == "Low" and supply_risk == "High":
        kraljic_quadrant = "Bottleneck"
    else:
        kraljic_quadrant = "Non-Critical"

    # Load Porter's Five Forces levels from Excel
    porter_df = pd.read_excel("data/UI_sheet.xlsx", sheet_name="Input - Porter Forces API", skiprows=1)
    porter_df.columns = ["Subcategory", "Region", "Porter Force", "Low", "Low-Med", "Med", "Med-High", "High"]

    # List of forces to check
    forces = ["Threat of New Entrants", "Bargaining power of suppliers", "Rivalry in the industry",
              "Bargaining power of buyers", "Threat of substitutes"]

    # Extract level for each force
    force_levels = {}
    for force in forces:
        match = porter_df[
            (porter_df["Subcategory"] == selected_subcategory) &
            (porter_df["Region"] == selected_region) &
            (porter_df["Porter Force"] == force)
        ]
        if not match.empty:
            row = match.iloc[0][["Low", "Low-Med", "Med", "Med-High", "High"]]
            level = row[row == 'X'].index[0] if 'X' in row.values else "Unknown"
            force_levels[force] = level

    # Load Market Dynamics advice from Excel
    market_df = pd.read_excel("data/UI_sheet.xlsx", sheet_name="Market Dynamics")
    market_advice = []

    for force, level in force_levels.items():
        match = market_df[(market_df["Porter Force"] == force)]
        if not match.empty:
            row = match.iloc[0]
            advice = row.get(level, "")
            market_advice.append({"force": force, "level": level, "advice": advice})

    # Load Category Strategy advice based on Kraljic Quadrant
    category_df = pd.read_excel("data/UI_sheet.xlsx", sheet_name="Category Strategy")
    strategy_advice = category_df[category_df["Quadrant"] == kraljic_quadrant]

    # Load Pricing Models advice based on subcategory
    pricing_df = pd.read_excel("data/UI_sheet.xlsx", sheet_name="Pricing Models")
    pricing_advice_df = pricing_df[pricing_df["Subcategory"] == selected_subcategory]
    pricing_advice = pricing_advice_df[["Consider this pricing model..", "Benefit/Use:"]].to_dict(orient="records")

    # Load Negotiation Levers advice based on subcategory
    levers_df = pd.read_excel("data/UI_sheet.xlsx", sheet_name="Negotiation Levers")
    levers_advice_df = levers_df[levers_df["Subcategory"] == selected_subcategory]
    levers_advice = levers_advice_df[["Subject", "Advice"]].to_dict(orient="records")

    # Load Contract Types advice based on subcategory
    contract_df = pd.read_excel("data/UI_sheet.xlsx", sheet_name="Contract Types")
    contract_advice_df = contract_df[contract_df["Subcategory"] == selected_subcategory]
    contract_advice = contract_advice_df[["Advice"]].to_dict(orient="records")

    return render_template(
        "results.html",
        subcategory=selected_subcategory,
        region=selected_region,
        sourcing_stage=sourcing_stage,
        contract_due_date=contract_due_date,
        financial_impact=financial_impact,
        supply_risk=supply_risk,
        market_volatility=market_volatility,
        supplier_dependency=supplier_dependency,
        force_levels=force_levels,
        kraljic_quadrant=kraljic_quadrant,
        strategy_advice=strategy_advice.to_dict(orient="records"),
        market_advice=market_advice,
        pricing_advice=pricing_advice,
        levers_advice=levers_advice,
        contract_advice=contract_advice
    )

@app.route("/health")
def health_check():
    return {"status": "healthy"}, 200

@app.errorhandler(Exception)
def handle_error(e):
    return render_template(
        "error.html",
        error=str(e)
    ), 500

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

if __name__ == "__main__":
    app.run(debug=True)
