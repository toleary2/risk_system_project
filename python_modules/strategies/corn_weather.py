def get_metadata():
    """
    Returns the parameters that the PM Toolkit needs for sizing and dashboards.
    """
    return {
        "StrategyName": "Corn_Weather_Macro",
        "Bucket": "Directional",
        "Conviction": "High",
        "RegimeFit": "Event"
    }

def calculate_signal(price_data):
    # This is where your complex research logic will eventually go
    return "Long"