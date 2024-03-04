import re

def text_splitter(license_plate_data: str) -> list:
    
    if license_plate_data.isalnum():
        data = re.finditer(pattern='^\s*(\d+)\s*(\w+)\s*$|^\s*(\D+)\s*(\d+)\s*$', string=license_plate_data, flags=0)
        for matches in data:
            lp_data = [lp for lp in matches.groups() if lp is not None]
        return lp_data


def error_detection(lp) -> list:
    """Detect errors in the recognized license plate data.

    Parameters:
    license_plate_data (list): The list of recognized license plate strings.

    Returns:
    error_detected_data (list): The list of license plate data with detected errors.
    """
    try:
        lp0 = lp[0]; lp1 = lp[1]; len_lp0 = len(lp[0]); len_lp1 = len(lp[1]); num_pattern = '^\s*(\d{2,4})\s*$'; letter_pattern = '^\s*(\D{2,4})\s*$'
        lp0_num = re.match(pattern=num_pattern, string=lp0)
        lp0_letter = re.match(pattern=letter_pattern, string=lp0)
        lp1_num = re.match(pattern=num_pattern, string=lp1)
        lp1_letter = re.match(pattern=letter_pattern, string=lp1)
        return True if ((lp0_num and lp1_letter) or (lp1_num and lp0_letter)) and (6 <= (len_lp0+len_lp1) <= 7) else False
    except:
        return False

def process_and_structure(license_plate_data) -> list:
    """Process and structure the recognized license plate data for further analysis.

    Parameters:
    license_plate_data (list): The list of recognized license plate strings.

    Returns:
    processed_data (list): The list of processed and structured license plate data.
    """
    lps = []
    for DUE in license_plate_data:
        if len(DUE) == 1:
            lp = text_splitter(DUE[0])
            error = error_detection(lp)
            lps.append({str(lp):error})
        elif len(DUE) == 2:
            error = error_detection(DUE)
            lps.append({str(DUE):error})
    return lps
    