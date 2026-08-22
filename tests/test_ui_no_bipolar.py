from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ui_hides_bipolar_toggle_and_bipolar_logic():
    html = (ROOT / 'ui_patient_data_observer.html').read_text()
    py = (ROOT / 'ui_patient_data_observer.py').read_text()

    assert 'toggleUnipolar' not in html
    assert 'egm_bi' not in py
    assert 'activation_bi' not in py
    assert 'blue: unipolar' in html
