import logging
import json
from datetime import datetime
import numpy as np

class ModelMonitor:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            filename='model_predictions.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
    def log_prediction(self, input_data, prediction, confidence):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'input_shape': input_data.shape if hasattr(input_data, 'shape') else None
        }
        logging.info(json.dumps(log_entry))