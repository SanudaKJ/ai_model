import unittest
import json
import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app
from app import app, calculate_bill

class TestElectricityApp(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'healthy')
    
    def test_calculate_bill(self):
        # Test the bill calculation function directly
        bill_data = calculate_bill(
            tariff_id="industrial_rate2",
            total_day_consumption=1000,
            total_peak_consumption=200,
            total_off_peak_consumption=300,
            max_apparent_power=50
        )
        
        self.assertEqual(bill_data['actual_tariff_used'], "industrial_rate2")
        self.assertEqual(bill_data['fixed_charge'], 5000.00)
        self.assertEqual(bill_data['day_charge'], 13000.00)  # 1000 * 13.00
        self.assertEqual(bill_data['peak_charge'], 4600.00)  # 200 * 23.00
        self.assertEqual(bill_data['off_peak_charge'], 3300.00)  # 300 * 11.00
        self.assertEqual(bill_data['demand_charge'], 65000.00)  # 50 * 1300.00
        self.assertEqual(bill_data['total_bill'], 90900.00)  # Sum of all charges
    
    def test_calculate_endpoint(self):
        # Test the /calculate endpoint
        request_data = {
            "tariff_category": "industrial_rate2",
            "day_consumption": 1000,
            "peak_consumption": 200,
            "off_peak_consumption": 300,
            "max_demand_kva": 50
        }
        
        response = self.app.post(
            '/calculate',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['total_bill'], 90900.00)

if __name__ == '__main__':
    unittest.main()