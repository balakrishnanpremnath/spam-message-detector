from preprocessing import clean_text, message_length_features


def test_clean_text_lowercases():
    assert clean_text("HELLO") == "hello"


def test_clean_text_strips_punctuation():
    result = clean_text("Wow!!! Free $$$ prize???")
    assert "!" not in result
    assert "$" not in result
    assert "?" not in result


def test_clean_text_normalizes_urls():
    result = clean_text("Click http://scam.example.com now")
    assert "urltoken" in result
    assert "http" not in result


def test_clean_text_normalizes_long_numbers():
    result = clean_text("Call 08712345678 now")
    assert "numtoken" in result
    assert "08712345678" not in result


def test_clean_text_stems_words():
    result = clean_text("running runner runs")
    # Porter stemmer should collapse these toward a shared root
    assert "run" in result


def test_message_length_features_keys():
    feats = message_length_features("WOW!!! free prize")
    assert set(feats.keys()) == {"char_count", "exclaim_count", "upper_ratio", "digit_ratio"}
    assert feats["exclaim_count"] == 3
    assert feats["char_count"] == len("WOW!!! free prize")
    

def test_clean_text_removes_extra_whitespace():
    result = clean_text("Hello    world")
    assert "  " not in result
