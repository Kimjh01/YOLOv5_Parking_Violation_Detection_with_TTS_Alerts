import datetime

# 요일과 금지되는 차량 번호 마지막 숫자 매핑
restriction_map = {
    0: [1, 6],  # 월요일
    1: [2, 7],  # 화요일
    2: [3, 8],  # 수요일
    3: [4, 9],  # 목요일
    4: [5, 0],  # 금요일
}

def classify_vehicle(plate_chars):
    # 숫자와 문자 분리
    digits = ''.join(filter(str.isdigit, plate_chars))
    
    if len(digits) < 2:
        return "알 수 없음"
    
    # 차량 번호 앞 두 자리 추출
    front_number = int(digits[:3])
    
    # 차량 유형 분류
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
    # 숫자와 문자 분리
    digits = ''.join(filter(str.isdigit, plate_chars))
    
    if not digits:
        return "출입 가능 (숫자 아님)"
    
    # 차량 번호 마지막 숫자 추출
    last_digit = int(digits[-1])
    
    # 오늘의 요일 추출 (0: 월요일, 1: 화요일, ..., 6: 일요일)
    today = datetime.datetime.today().weekday()
    
    if today < 5:  # 월요일(0)부터 금요일(4)까지 확인
        if last_digit in restriction_map[today]:
            return "출입 불가능"
        else:
            return "출입 가능"
    else:
        return "출입 가능 (주말)"