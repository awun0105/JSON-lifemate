"""Test bộ đổi số đọc thành chữ số (`lifemate/backends/asr_numbers.py`, thử nghiệm L1 của benchmark ASR lần 4)."""

from __future__ import annotations

import pytest

from lifemate.backends.asr_numbers import digitize, parse_number


@pytest.mark.parametrize(
    ("words", "value"),
    [
        ("mười", 10),
        ("mười lăm", 15),
        ("hai mươi mốt", 21),
        ("hai mươi tư", 24),
        ("ba mươi", 30),
        ("một trăm", 100),
        ("một trăm linh năm", 105),
        ("một trăm hai mươi bảy", 127),
        ("hai trăm năm", 250),
        ("năm trăm nghìn", 500_000),
        ("một triệu hai", 1_200_000),
        ("hai nghìn không trăm hai mươi sáu", 2026),
        ("một tỷ", 1_000_000_000),
    ],
)
def test_parse_number(words: str, value: int) -> None:
    assert parse_number(words.split()) == value


@pytest.mark.parametrize("words", ["ba bốn", "mươi", "năm hai", "một mươi", "nghìn", "mười không"])
def test_parse_number_rejects_non_numbers(words: str) -> None:
    assert parse_number(words.split()) is None


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("tập báo thức năm giờ nhé à không năm rưỡi sáng mai đó", "tập báo thức 5 giờ nhé à không 5 rưỡi sáng mai đó"),
        ("để ba mươi phần trăm thôi", "để 30 phần trăm thôi"),
        ("bật điều hoà hai mươi sáu độ", "bật điều hoà 26 độ"),
        ("gọi số không chín không một hai ba bốn năm sáu bảy", "gọi số 0901234567"),
        ("chuyển năm trăm nghìn cho mẹ", "chuyển 500 nghìn cho mẹ"),
        ("còn một triệu hai thôi", "còn 1 triệu 2 thôi"),
        ("giá vàng một trăm hai mươi bảy triệu", "giá vàng 127 triệu"),
        ("bảy giờ kém mười lăm tối mai", "7 giờ kém 15 tối mai"),
        ("mười hai giờ ba mươi", "12 giờ 30"),
        ("kênh vtv sáu tám giờ tối", "kênh vtv 6 8 giờ tối"),
        ("mở kênh ba", "mở kênh 3"),
        ("bật điều hòa phòng ngủ hai sáu độ nhé", "bật điều hòa phòng ngủ 26 độ nhé"),
        ("chín h sáng mai", "9 h sáng mai"),
        ("à không giờ anh đổ e năm rồi", "à không giờ anh đổ e năm rồi"),
        ("hai ba giờ gì đó", "hai 3 giờ gì đó"),
        ("ba của con", "ba của con"),
        ("thứ tư tuần sau", "thứ tư tuần sau"),
        ("năm nay anh ba mươi tuổi", "năm nay anh 30 tuổi"),
        ("năm hai nghìn hai mươi sáu", "năm 2026"),
        ("đặt một tô ramen và hai chỗ", "đặt một tô ramen và hai chỗ"),
        ("ba bốn chỗ gì đó", "ba bốn chỗ gì đó"),
        ("một lít xăng", "một lít xăng"),
        ("", ""),
        ("xem trời bây giờ thế nào", "xem trời bây giờ thế nào"),
    ],
)
def test_digitize(text: str, want: str) -> None:
    assert digitize(text) == want
