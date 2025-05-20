import logging
from pathlib import Path

def get_logger(name: str = "quant") -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Name of the logger, defaults to "quant"
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "quant.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(name) 