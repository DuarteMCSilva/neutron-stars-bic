import time

def get_current_date_string():
    """
    Returns the current date and time as a string in the format "YYYY-MM-DD_HH-MM-SS".
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S")