import re


def text_splitter(license_plate_data: str) -> list:
    """Split the license plate data into groups based on regular expressions.

    Parameters:
    license_plate_data (str): The license plate data as a string.

    Returns:
    list: A list of groups that match the regular expressions.
    """
    try:
        if license_plate_data.isalnum():
            data1 = re.finditer(
                pattern=r"^\s*(\d+)\s*(\w+)\s*$", string=license_plate_data, flags=0
            )
            data2 = re.finditer(
                pattern=r"^\s*(\D+)\s*(\d+)\s*$", string=license_plate_data, flags=0
            )
            lp_data1 = lp_data2 = None
            for matches in data1:
                lp_data1 = [lp for lp in matches.groups() if lp is not None]
            for matches in data2:
                lp_data2 = [lp for lp in matches.groups() if lp is not None]
                lp_data2.reverse()
            return [
                lp
                for lp in [lp_data1, lp_data2]
                if ((lp_data1 is None) ^ (lp_data2 is None)) and (lp is not None)
            ][0]
    except:
        return None


def error_detection(lp: list) -> bool:
    """Detect errors in the recognized license plate data.

    Parameters:
    lp (list): The list of recognized license plate strings.

    Returns:
    bool: True if an error is detected, False otherwise.
    """
    try:
        if lp is not None:
            lp0 = lp[0]
            lp1 = lp[1]
            len_lp0 = len(lp0)
            len_lp1 = len(lp1)
            len_lp01 = len_lp0 + len_lp1
            num_pattern1 = r"^\s*(\d{4})\s*$"
            letter_pattern1 = r"^\s*(\D{2})\s*$"
            num_pattern2 = r"^\s*(\d{3,4})\s*$"
            letter_pattern2 = r"^\s*(\D{3})\s*$"
            lp0_num_pattern1 = re.search(pattern=num_pattern1, string=lp0)
            lp1_letter_pattern1 = re.search(pattern=letter_pattern1, string=lp1)
            lp0_num_pattern2 = re.search(pattern=num_pattern2, string=lp0)
            lp1_letter_pattern2 = re.search(pattern=letter_pattern2, string=lp1)

            if lp0_num_pattern1 is not None:
                lp0_num_pattern1 = lp0_num_pattern1[0]
            if lp1_letter_pattern1 is not None:
                lp1_letter_pattern1 = lp1_letter_pattern1[0]
            if lp0_num_pattern2 is not None:
                lp0_num_pattern2 = lp0_num_pattern2[0]
            if lp1_letter_pattern2 is not None:
                lp1_letter_pattern2 = lp1_letter_pattern2[0]

            return (
                True
                if (
                    bool(lp0_num_pattern1 and lp1_letter_pattern1)
                    != bool(lp0_num_pattern2 and lp1_letter_pattern2)
                )
                and (6 <= len_lp01 <= 7)
                else False
            )
        else:
            return False
    except TypeError:
        return False


def process_and_structure(license_plate_data: list) -> list:
    """Process and structure the recognized license plate data for further analysis.

    Parameters:
    license_plate_data (list): The list of recognized license plate strings.

    Returns:
    list: A list of dictionaries with license plate data as keys and error detection results as values.
    """
    lps = []
    for DUE in license_plate_data:
        if len(DUE) == 1:
            lp = text_splitter(DUE[0])
            if lp is not None:
                error = error_detection(lp)
                lps.append({str(lp): error})
        elif len(DUE) == 2:
            error = error_detection(DUE)
            lps.append({str(DUE): error})
    return lps
