from flask import Flask, render_template, request, jsonify
import pandas as pd
from rate_card_predictor import RateCardPredictor

app = Flask(__name__)
rate_predictor = RateCardPredictor()

# Load and train the model when the app starts
try:
    data = rate_predictor.load_data('data/Rate_Cards_20.22.24.V2.xlsx')
    model_score = rate_predictor.train_model()
    
    # Get unique values for dropdowns
    unique_suppliers = sorted(data['Supplier'].unique())
    unique_roles = sorted(data['Roles'].unique())
    unique_countries = sorted(data['Country'].unique())
except Exception as e:
    print(f"Error loading rate card model: {e}")
    unique_suppliers = []
    unique_roles = []
    unique_countries = []

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
    
    # Calculate Strategic Importance score (average of 5 questions)
    strategic_scores = [
        int(request.form.get(f'strategic_q{i}', 0)) 
        for i in range(1, 6)
    ]
    strategic_importance = sum(strategic_scores) / len(strategic_scores)

    # Calculate Supply Risk score (average of 5 questions)
    # Note: Questions 1, 2, 3, and 4 are reverse scored because they are negatively worded
    risk_scores = [
        8 - int(request.form.get('risk_q1', 0)),  # Reverse score
        8 - int(request.form.get('risk_q2', 0)),  # Reverse score
        8 - int(request.form.get('risk_q3', 0)),  # Reverse score
        8 - int(request.form.get('risk_q4', 0)),  # Reverse score
        int(request.form.get('risk_q5', 0))       # Normal score
    ]
    supply_risk = sum(risk_scores) / len(risk_scores)

    # Determine Kraljic Quadrant based on average scores
    # Scores 1-4 are considered "Low", 4-7 are considered "High"
    if strategic_importance >= 4:
        if supply_risk >= 4:
            kraljic_quadrant = "Strategic"
        else:
            kraljic_quadrant = "Leverage"
    else:
        if supply_risk >= 4:
            kraljic_quadrant = "Bottleneck"
        else:
            kraljic_quadrant = "Non-Critical"

    # Load Category Strategy advice based on Kraljic Quadrant
    category_df = pd.read_excel("data/UI_sheet.xlsx", sheet_name="Category Strategy")
    strategy_advice = category_df[category_df["Quadrant"] == kraljic_quadrant].to_dict(orient="records")

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

    # Load resource information from Mapping sheet
    mapping_df = pd.read_excel("data/UI_sheet.xlsx", sheet_name="Mapping")
    market_resource = mapping_df[mapping_df["Tool"] == "Market Dynamics"]["Resource"].iloc[0] if not mapping_df[mapping_df["Tool"] == "Market Dynamics"].empty else ""
    strategy_resource = mapping_df[mapping_df["Tool"] == "Category Strategy"]["Resource"].iloc[0] if not mapping_df[mapping_df["Tool"] == "Category Strategy"].empty else ""
    pricing_resource = mapping_df[mapping_df["Tool"] == "Pricing Models"]["Resource"].iloc[0] if not mapping_df[mapping_df["Tool"] == "Pricing Models"].empty else ""
    levers_resource = mapping_df[mapping_df["Tool"] == "Negotiation Levers"]["Resource"].iloc[0] if not mapping_df[mapping_df["Tool"] == "Negotiation Levers"].empty else ""
    contracts_resource = mapping_df[mapping_df["Tool"] == "Contract Types"]["Resource"].iloc[0] if not mapping_df[mapping_df["Tool"] == "Contract Types"].empty else ""

    # Generate summaries for each tab
    market_summary = f"For {selected_subcategory} in {selected_region}, market analysis shows {force_levels.get('Bargaining power of suppliers', 'varying')} supplier power and {force_levels.get('Rivalry in the industry', 'varying')} industry rivalry, suggesting a need for {market_advice[0]['advice'] if market_advice else 'strategic market approach'}."
    
    strategy_summary = f"As a {kraljic_quadrant.lower()} category with strategic importance of {strategic_importance:.1f}/7 and supply risk of {supply_risk:.1f}/7, focus on {strategy_advice[0]['Focus on..'] if strategy_advice else 'strategic relationship building'}."
    
    pricing_summary = f"For {selected_subcategory}, consider {pricing_advice[0]['Consider this pricing model..'] if pricing_advice else 'appropriate pricing models'} to {pricing_advice[0]['Benefit/Use:'] if pricing_advice else 'optimize value'}."
    
    levers_summary = f"Key negotiation focus areas for {selected_subcategory} include {levers_advice[0]['Subject'] if levers_advice else 'strategic levers'} to {levers_advice[0]['Advice'] if levers_advice else 'achieve optimal outcomes'}."
    
    contracts_summary = f"For {selected_subcategory}, {contract_advice[0]['Advice'] if contract_advice else 'implement appropriate contract structures'} to ensure successful engagement."

    # Calculate MAPE for the model
    try:
        mape = rate_predictor.calculate_mape()
    except Exception as e:
        print(f"Error calculating MAPE: {e}")
        mape = None

    # Get predictions for all positions in China
    try:
        china_predictions = rate_predictor.predict_all_china_positions()
    except Exception as e:
        print(f"Error getting China predictions: {e}")
        china_predictions = []

    return render_template(
        "results.html",
        subcategory=selected_subcategory,
        region=selected_region,
        sourcing_stage=sourcing_stage,
        strategic_importance=strategic_importance,
        supply_risk=supply_risk,
        kraljic_quadrant=kraljic_quadrant,
        strategy_advice=strategy_advice,
        market_advice=market_advice,
        force_levels=force_levels,
        pricing_advice=pricing_advice,
        levers_advice=levers_advice,
        contract_advice=contract_advice,
        market_resource=market_resource,
        strategy_resource=strategy_resource,
        pricing_resource=pricing_resource,
        levers_resource=levers_resource,
        contracts_resource=contracts_resource,
        strategic_scores=strategic_scores,
        risk_scores=risk_scores,
        market_summary=market_summary,
        strategy_summary=strategy_summary,
        pricing_summary=pricing_summary,
        levers_summary=levers_summary,
        contracts_summary=contracts_summary,
        unique_suppliers=unique_suppliers,
        unique_roles=unique_roles,
        unique_countries=unique_countries,
        current_year=rate_predictor.current_year,
        mape=mape,
        china_predictions=china_predictions
    )

@app.route("/predict_rate", methods=["POST"])
def predict_rate():
    try:
        # Get input data from request
        input_data = {
            'Supplier': request.form.get('supplier'),
            'Roles': request.form.get('role'),
            'Experience': request.form.get('experience'),
            'Currency': request.form.get('currency', 'USD'),
            'Country': request.form.get('country'),
            'Year': 2025,  # Set fixed year to 2025
            'Delta PPI': float(request.form.get('delta_ppi', 0)),
            'Delta GDP': float(request.form.get('delta_gdp', 0)),
            'Delta Electricity': float(request.form.get('delta_electricity', 0)),
            'Delta Labor': float(request.form.get('delta_labor', 0)),
            'Inflation': float(request.form.get('inflation', 0))
        }
        
        # Make prediction
        predicted_rate = rate_predictor.predict_rate(input_data)
        
        return jsonify({
            'success': True,
            'predicted_rate': round(float(predicted_rate), 2)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

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
    app.run(debug=True, port=5001)
