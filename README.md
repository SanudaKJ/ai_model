# Sri Lanka Electricity Bill Predictor

A machine learning-based API for predicting electricity bills for industrial customers in Sri Lanka based on their machine usage patterns.

## Features

- Calculate electricity bills based on Sri Lanka's industrial tariff structure
- Predict electricity bills using a machine learning model
- RESTful API for easy integration with other systems
- Automated CI/CD pipeline for smooth deployments

## System Architecture

The system consists of:

1. **Flask REST API**: Handles requests and responses for bill calculation and prediction
2. **Machine Learning Model**: Predicts electricity bills based on historical data
3. **CI/CD Pipeline**: Automated testing and deployment to AWS EC2

## API Endpoints

### Health Check
```
GET /health
```

### Calculate Bill (based on tariff structure)
```
POST /calculate
```
Request body:
```json
{
  "tariff_category": "industrial_rate2",
  "day_consumption": 1000,
  "peak_consumption": 200,
  "off_peak_consumption": 300,
  "max_demand_kva": 50
}
```

### Predict Bill (using ML model)
```
POST /predict
```
Request body:
```json
{
  "company_id": "Company_1",
  "company_size": "medium",
  "tariff_category": "industrial_rate2",
  "working_days": 22,
  "machines": [
    {
      "name": "Electric Furnace",
      "kw": 60,
      "power_factor": 0.85,
      "day_hours": 8,
      "peak_hours": 2,
      "off_peak_hours": 0
    },
    {
      "name": "Air Compressor",
      "kw": 15,
      "power_factor": 0.9,
      "day_hours": 8,
      "peak_hours": 2,
      "off_peak_hours": 0
    }
  ]
}
```

## Deployment

### Requirements

- Python 3.9+
- AWS EC2 instance (t2.micro or larger)
- GitHub account for CI/CD

### Manual Deployment

1. Clone the repository:
   ```
   git clone https://github.com/your-username/srilanka-electricity-predictor.git
   cd srilanka-electricity-predictor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Train the model (if not already trained):
   ```
   python train_model.py
   ```

4. Run the application:
   ```
   python app.py
   ```

### CI/CD Deployment

The project includes GitHub Actions workflow for automated CI/CD:

1. Push changes to the `main` branch
2. GitHub Actions will:
   - Run tests
   - Train the model if needed
   - Deploy to EC2
   - Restart the service

## Development

### Dataset Generation

To generate a new synthetic dataset:
```
python generate_dataset.py
```

### Training the Model

To train the model on the dataset:
```
python train_model.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
