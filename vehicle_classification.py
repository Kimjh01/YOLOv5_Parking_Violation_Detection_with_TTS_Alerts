import datetime

restriction_map = {
    0: [1, 6],
    1: [2, 7],
    2: [3, 8],
    3: [4, 9],
    4: [5, 0],
}

def classify_vehicle(plate_chars):
    digits = ''.join(filter(str.isdigit, plate_chars))
    if len(digits) < 2:
        return "알 수 없음"
    front_number = int(digits[:3])
    if 0 <= front_number <= 699:
        return "승용차"
    elif 700 <= front_number <= 799:
        return "승합차"
    elif 800 <= front_number <= 979:
        return "화물차"
    elif 980 <= front_number <= 997:
        return "특수차"
    elif front_number in [998, 999]:
        return "긴급차"
    else:
        return "알 수 없음"

def can_enter_public_office(plate_chars):
    digits = ''.join(filter(str.isdigit, plate_chars))
    if not digits:
        return "출입 가능 (숫자 아님)"
    last_digit = int(digits[-1])
    today = datetime.datetime.today().weekday()
    if today < 5:
        if last_digit in restriction_map[today]:
            return "출입 불가능"
        else:
            return "출입 가능"
    else:
        return "출입 가능 (주말)"
