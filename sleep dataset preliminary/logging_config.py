import os
import logging
from datetime import datetime

def setup_logging():
    # Define the directory for the log files
    log_directory = "/Users/jeonsang-eon/sleep_data_processed"
    log_file_name = f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
    log_file_path = os.path.join(log_directory, log_file_name)
    
    # Ensure the directory exists
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    # Configure logging to write to the specified file
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
