from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# load the model
MODEL_PATH = 'srilanka_electricity_predictor.pkl'
model = None

# tariff structure
tariff_structure = {
    "industrial_rate1_small": {
        "name": "Industrial Rate1 (VDMC ≤ 300 kWh/month)",
        "fixed_charge": 250.00,  
        "day_time_charge": 7.00, 
        "peak_time_charge": 7.00,  
        "off_peak_charge": 7.00, 
        "demand_charge": 0  
    },
    "industrial_rate1_large": {
        "name": "Industrial Rate1 (VDMC > 300 kWh/month)",
        "fixed_charge": 750.00,  
        "day_time_charge": 13.00,  
        "peak_time_charge": 13.00, 
        "off_peak_charge": 13.00,  
        "demand_charge": 0  
    },
    "industrial_rate2": {
        "name": "Industrial Rate 2",
        "fixed_charge": 5000.00,  
        "day_time_charge": 13.00,  
        "peak_time_charge": 23.00,  
        "off_peak_charge": 11.00,  
        "demand_charge": 1300.00  
    },
    "industrial_rate3": {
        "name": "Industrial Rate 3",
        "fixed_charge": 5000.00,  
        "day_time_charge": 12.00,  
        "peak_time_charge": 22.00,  
        "off_peak_charge": 10.00, 
        "demand_charge": 1250.00  
    }
}

def load_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    return model

def calculate_bill(tariff_id, total_day_consumption, total_peak_consumption, total_off_peak_consumption, max_apparent_power):
    """Calculate electricity bill using the tariff structure"""
    
# total monthly consumption
    total_monthly_consumption = total_day_consumption + total_peak_consumption + total_off_peak_consumption
    
# switch the tariff for Industrial Rate 1
    if tariff_id == "industrial_rate1_small" and total_monthly_consumption > 300:
        print("Consumption exceeds 300 kWh/month, switching to industrial_rate1_large")
        tariff_id = "industrial_rate1_large"
    
# tariff details
    tariff = tariff_structure[tariff_id]
    
# calculate bill components
    fixed_charge = tariff["fixed_charge"]
    day_charge = total_day_consumption * tariff["day_time_charge"]
    peak_charge = total_peak_consumption * tariff["peak_time_charge"]
    off_peak_charge = total_off_peak_consumption * tariff["off_peak_charge"]
    demand_charge = max_apparent_power * tariff["demand_charge"]
    
# calculate total bill
    total_bill = fixed_charge + day_charge + peak_charge + off_peak_charge + demand_charge
    
    return {
        "actual_tariff_used": tariff_id,
        "tariff_name": tariff["name"],
        "fixed_charge": fixed_charge,
        "day_charge": day_charge,
        "peak_charge": peak_charge,
        "off_peak_charge": off_peak_charge,
        "demand_charge": demand_charge,
        "total_bill": total_bill,
        "total_consumption_kwh": total_monthly_consumption
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict electricity bill using ML model"""
    try:
        if load_model() is None:
            return jsonify({"error": "Model not loaded. Please ensure the model file exists."}), 500
        
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_fields = ["company_id", "company_size", "tariff_category", "machines"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        machines = data["machines"]
        if not machines or not isinstance(machines, list):
            return jsonify({"error": "Invalid or empty machines list"}), 400
        
        total_kw = 0
        total_day_consumption = 0
        total_peak_consumption = 0
        total_off_peak_consumption = 0
        max_apparent_power = 0
        
        machine_counts = {}
        all_day_hours = []
        all_peak_hours = []
        all_off_peak_hours = []
        all_power_factors = []
        all_kw = []
        
        working_days = data.get("working_days", 22)  
        
        for machine in machines:
            machine_required_fields = ["name", "kw", "power_factor", "day_hours", "peak_hours", "off_peak_hours"]
            for field in machine_required_fields:
                if field not in machine:
                    return jsonify({"error": f"Missing required field in machine: {field}"}), 400
            
            machine_name = machine["name"]
            kw = float(machine["kw"])
            power_factor = float(machine["power_factor"])
            day_hours = float(machine["day_hours"])
            peak_hours = float(machine["peak_hours"])
            off_peak_hours = float(machine["off_peak_hours"])
            
            apparent_power = kw / power_factor
            
            max_apparent_power = max(max_apparent_power, apparent_power)
            
            day_consumption = kw * day_hours * working_days
            peak_consumption = kw * peak_hours * working_days
            off_peak_consumption = kw * off_peak_hours * working_days
            
            total_kw += kw
            total_day_consumption += day_consumption
            total_peak_consumption += peak_consumption
            total_off_peak_consumption += off_peak_consumption
            
            machine_counts[machine_name] = machine_counts.get(machine_name, 0) + 1
            
            all_day_hours.append(day_hours)
            all_peak_hours.append(peak_hours)
            all_off_peak_hours.append(off_peak_hours)
            all_power_factors.append(power_factor)
            all_kw.append(kw)
        
        avg_day_hours = sum(all_day_hours) / len(all_day_hours) if all_day_hours else 0
        avg_peak_hours = sum(all_peak_hours) / len(all_peak_hours) if all_peak_hours else 0
        avg_off_peak_hours = sum(all_off_peak_hours) / len(all_off_peak_hours) if all_off_peak_hours else 0
        avg_power_factor = sum(all_power_factors) / len(all_power_factors) if all_power_factors else 0
        avg_kw = sum(all_kw) / len(all_kw) if all_kw else 0
        max_kw = max(all_kw) if all_kw else 0
        
        total_monthly_consumption = total_day_consumption + total_peak_consumption + total_off_peak_consumption
        
        tariff_id = data["tariff_category"]
        if tariff_id == "industrial_rate1_small" and total_monthly_consumption > 300:
            actual_tariff = "industrial_rate1_large"
        else:
            actual_tariff = tariff_id
        
        bill_calculation = calculate_bill(
            tariff_id=tariff_id,
            total_day_consumption=total_day_consumption,
            total_peak_consumption=total_peak_consumption,
            total_off_peak_consumption=total_off_peak_consumption,
            max_apparent_power=max_apparent_power
        )
        
        features = {
            'company_id': data["company_id"],
            'company_size': data["company_size"],
            'tariff_category': tariff_id,
            'actual_tariff': actual_tariff,
            'working_days': working_days,
            'year': data.get("year", 2025),  
            'month_num': data.get("month", 1),  
            'num_machines': len(machines),
            'total_kw': total_kw,
            'avg_kw': avg_kw,
            'max_kw': max_kw,
            'avg_power_factor': avg_power_factor,
            'avg_day_hours': avg_day_hours,
            'avg_peak_hours': avg_peak_hours,
            'avg_off_peak_hours': avg_off_peak_hours,
            'total_day_kwh': total_day_consumption,
            'total_peak_kwh': total_peak_consumption,
            'total_off_peak_kwh': total_off_peak_consumption,
            'total_monthly_kwh': total_monthly_consumption,
            'max_demand_kva': max_apparent_power,
            'season': data.get("season", "non-season"),  
        }
        
        machine_types = [
            "Air Compressor", "CNC Machine", "Conveyor Belt", "Electric Furnace", 
            "Hydraulic Press", "Injection Molding Machine", "Industrial Mixer", 
            "Lathe Machine", "Milling Machine", "Packaging Machine", "Pump", "Welding Machine"
        ]
        
        for machine_type in machine_types:
            clean_name = machine_type.replace(" ", "_")
            features[f'count_{clean_name}'] = machine_counts.get(machine_type, 0)
        
        prediction_df = pd.DataFrame([features])
        
        predicted_bill = model.predict(prediction_df)[0]
        
        response = {
            "calculation": bill_calculation,
            "prediction": {
                "predicted_bill": predicted_bill,
                "difference": predicted_bill - bill_calculation["total_bill"],
                "difference_percent": ((predicted_bill - bill_calculation["total_bill"]) / bill_calculation["total_bill"]) * 100
            },
            "features": features
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/calculate', methods=['POST'])
def calculate():
    """Endpoint to calculate electricity bill based on tariff structure only (no ML)"""
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_fields = ["tariff_category", "day_consumption", "peak_consumption", "off_peak_consumption", "max_demand_kva"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        tariff_id = data["tariff_category"]
        day_consumption = float(data["day_consumption"])
        peak_consumption = float(data["peak_consumption"])
        off_peak_consumption = float(data["off_peak_consumption"])
        max_demand = float(data["max_demand_kva"])
        
        bill_result = calculate_bill(
            tariff_id=tariff_id,
            total_day_consumption=day_consumption,
            total_peak_consumption=peak_consumption,
            total_off_peak_consumption=off_peak_consumption,
            max_apparent_power=max_demand
        )
        
        return jsonify(bill_result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)



# Random Forest