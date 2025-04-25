import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# set random seed for reproducibility
np.random.seed(42)

# define  tariff structure
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

# industrial machines with their voltage and power ratings
machines = [
     {"name": "Bailing Machine", "voltage_range": [400], "kw_range": [8], "power_factor": [0.8]},
    {"name": "Bobst Gluing Machine", "voltage_range": [415], "kw_range": [12], "power_factor": [0.8]},
    {"name": "Bobst Nova Cut 1", "voltage_range": [415], "kw_range": [25], "power_factor": [0.9]},
    {"name": "Bobst Nova Cut 2", "voltage_range": [415], "kw_range": [25], "power_factor": [0.8]},
    {"name": "Box Making Machine", "voltage_range": [400], "kw_range": [5], "power_factor": [0.9]},
    {"name": "Carton Breaking Machine", "voltage_range": [415], "kw_range": [10], "power_factor": [0.8]},
    {"name": "CD 102 Heidelburge", "voltage_range": [415], "kw_range": [160], "power_factor": [0.8]},
    {"name": "Diana Eye", "voltage_range": [415], "kw_range": [12], "power_factor": [0.8]},
    {"name": "Guillotine Polar 115E", "voltage_range": [415], "kw_range": [20], "power_factor": [0.9]},
    {"name": "Guillotine Polar 115EM", "voltage_range": [415], "kw_range": [20], "power_factor": [0.9]},
    {"name": "Heidelburge Five Colour SPM", "voltage_range": [415], "kw_range": [72], "power_factor": [0.8]},
    {"name": "Heidelburge Six Colour", "voltage_range": [415], "kw_range": [160], "power_factor": [0.8]},
    {"name": "Heidelburge XL", "voltage_range": [415], "kw_range": [90], "power_factor": [0.9]},
    {"name": "Horizon Book Binding", "voltage_range": [400], "kw_range": [18], "power_factor": [0.9]},
    {"name": "Horizon Book Binding AFC", "voltage_range": [400], "kw_range": [12], "power_factor": [0.8]},
    {"name": "Horizon Book Binding AFC7", "voltage_range": [400], "kw_range": [15], "power_factor": [0.8]},
    {"name": "Komori LS", "voltage_range": [415], "kw_range": [180], "power_factor": [0.9]},
    {"name": "SBL Die Cutter", "voltage_range": [415], "kw_range": [30], "power_factor": [0.9]},
    {"name": "Screen UV", "voltage_range": [415], "kw_range": [25], "power_factor": [0.9]},
    {"name": "Sun Gluing", "voltage_range": [415], "kw_range": [15], "power_factor": [0.9]},
    {"name": "Thermal Laminating", "voltage_range": [415], "kw_range": [12], "power_factor": [0.8]},
    {"name": "UV Varnish", "voltage_range": [415], "kw_range": [26], "power_factor": [0.8]},
    {"name": "Verimatrix Die Cutter", "voltage_range": [415], "kw_range": [40], "power_factor": [0.8]},
    {"name": "Central A/C", "voltage_range": [400], "kw_range": [125], "power_factor": [0.7]},
    {"name": "Compressuer 1", "voltage_range": [400], "kw_range": [15], "power_factor": [0.8]},
    {"name": "Compressuer 2", "voltage_range": [400], "kw_range": [40], "power_factor": [0.8]}
]

# define typical usage patterns
usage_patterns = {
    "single_shift": {
        "description": "Single shift operation (8 hours)",
        "day_hours": 8,
        "peak_hours": 0,
        "off_peak_hours": 0
    },
    "double_shift": {
        "description": "Double shift operation (16 hours)",
        "day_hours": 13,
        "peak_hours": 3,
        "off_peak_hours": 0
    },
    "triple_shift": {
        "description": "Triple shift operation (24 hours)",
        "day_hours": 13,
        "peak_hours": 4,
        "off_peak_hours": 7
    },
    "mixed_shift": {
        "description": "Mixed shift with night operations",
        "day_hours": 10,
        "peak_hours": 2,
        "off_peak_hours": 4
    }
}

# number of companies 
num_companies = 20
start_date = datetime(2024, 1, 1)
months = 12


data = []

# generate data for each company
for company_id in range(1, num_companies + 1):
    sizes = ["small", "medium", "large"]
    company_size = sizes[company_id % 3]  
    
    if company_size == "small":
        tariff_category = "industrial_rate1_small"
    elif company_size == "medium":
        tariff_category = "industrial_rate1_large"
    else:  
        tariff_category = "industrial_rate3"
        
    
    if company_size == "small":
        usage_pattern = "single_shift"
    elif company_size == "medium":
        usage_pattern = "double_shift"
    else:  
        usage_pattern = "triple_shift"
    
    if company_size == "small":
        num_machines = 3  
    elif company_size == "medium":
        num_machines = 8  
    else: 
        num_machines = 15  
   
   
    company_machines = []
    for i in range(num_machines):
        machine_index = (company_id + i) % len(machines)
        company_machines.append(machines[machine_index])
    
    # monthly data
    for month in range(months):
        current_date = start_date + timedelta(days=30 * month)
        month_name = current_date.strftime("%Y-%m")
        
        working_days = 22  
        
        total_day_consumption = 0
        total_peak_consumption = 0
        total_off_peak_consumption = 0
        
        max_apparent_power = 0
        
        # add data for each machine
        for i, machine in enumerate(company_machines):
            machine_name = machine["name"]
            voltage = machine["voltage_range"][0]  
            kw = machine["kw_range"][0]  
            power_factor = 0.85  
            
            apparent_power = round(kw / power_factor, 2)
            
            max_apparent_power = max(max_apparent_power, apparent_power)
            
            day_hours = usage_patterns[usage_pattern]["day_hours"]
            peak_hours = usage_patterns[usage_pattern]["peak_hours"]
            off_peak_hours = usage_patterns[usage_pattern]["off_peak_hours"]
            
            day_consumption = round(kw * day_hours * working_days, 2)
            peak_consumption = round(kw * peak_hours * working_days, 2)
            off_peak_consumption = round(kw * off_peak_hours * working_days, 2)
            
            total_day_consumption += day_consumption
            total_peak_consumption += peak_consumption
            total_off_peak_consumption += off_peak_consumption
            
            # add machine data
            data.append({
                "company_id": f"Company_{company_id}",
                "month": month_name,
                "company_size": company_size,
                "tariff_category": tariff_category,
                "tariff_name": tariff_structure[tariff_category]["name"],
                "machine_name": machine_name,
                "machine_id": i + 1,
                "voltage": voltage,
                "kw": kw,
                "power_factor": power_factor,
                "apparent_power_kva": apparent_power,
                "day_hours": day_hours,
                "peak_hours": peak_hours,
                "off_peak_hours": off_peak_hours,
                "working_days": working_days,
                "day_consumption_kwh": day_consumption,
                "peak_consumption_kwh": peak_consumption,
                "off_peak_consumption_kwh": off_peak_consumption,
                "total_machine_consumption_kwh": day_consumption + peak_consumption + off_peak_consumption
            })
        
        # total monthly consumption
        total_monthly_consumption = total_day_consumption + total_peak_consumption + total_off_peak_consumption
        
       
        if tariff_category == "industrial_rate1_small" and total_monthly_consumption > 300:
            actual_tariff = "industrial_rate1_large"
        else:
            actual_tariff = tariff_category
        
        # Calculate bill
        fixed_charge = tariff_structure[actual_tariff]["fixed_charge"]
        
        day_charge = total_day_consumption * tariff_structure[actual_tariff]["day_time_charge"]
        peak_charge = total_peak_consumption * tariff_structure[actual_tariff]["peak_time_charge"]
        off_peak_charge = total_off_peak_consumption * tariff_structure[actual_tariff]["off_peak_charge"]
        
        demand_charge = max_apparent_power * tariff_structure[actual_tariff]["demand_charge"]
        
        # calculate total bill
        total_bill = fixed_charge + day_charge + peak_charge + off_peak_charge + demand_charge
        
        # update all entries
        for entry in data:
            if entry["company_id"] == f"Company_{company_id}" and entry["month"] == month_name:
                entry["total_company_day_kwh"] = round(total_day_consumption, 2)
                entry["total_company_peak_kwh"] = round(total_peak_consumption, 2)
                entry["total_company_off_peak_kwh"] = round(total_off_peak_consumption, 2)
                entry["total_monthly_consumption_kwh"] = round(total_monthly_consumption, 2)
                entry["max_demand_kva"] = round(max_apparent_power, 2)
                entry["actual_tariff_used"] = actual_tariff
                entry["fixed_charge"] = fixed_charge
                entry["day_energy_charge"] = round(day_charge, 2)
                entry["peak_energy_charge"] = round(peak_charge, 2)
                entry["off_peak_energy_charge"] = round(off_peak_charge, 2)
                entry["demand_charge"] = round(demand_charge, 2)
                entry["total_bill"] = round(total_bill, 2)

df = pd.DataFrame(data)

# save to CSV
df.to_csv("srilanka_electricity_dataset.csv", index=False)
print(f"Dataset created with {len(df)} entries.")


print("\nSample data:")
print(df.head())

# summary statistics
print("\nSummary statistics by tariff category:")
for tariff in tariff_structure:
    subset = df[df["actual_tariff_used"] == tariff]
    if len(subset) > 0:
        print(f"\n{tariff_structure[tariff]['name']}:")
        print(f"  Number of records: {len(subset)}")
        print(f"  Average monthly consumption: {subset['total_monthly_consumption_kwh'].mean():.2f} kWh")
        print(f"  Average bill: Rs. {subset['total_bill'].mean():.2f}") 