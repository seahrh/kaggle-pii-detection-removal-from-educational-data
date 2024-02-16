__all__ = ["ID_TO_LABEL", "LABEL_TO_ID"]

ID_TO_LABEL = {
    0: "O",
    1: "B-NAME_STUDENT",
    2: "I-NAME_STUDENT",
    3: "B-URL_PERSONAL",
    4: "I-URL_PERSONAL",
    5: "B-ID_NUM",
    6: "I-ID_NUM",
    7: "B-EMAIL",
    8: "I-EMAIL",
    9: "B-USERNAME",
    10: "I-USERNAME",
    11: "B-PHONE_NUM",
    12: "I-PHONE_NUM",
    13: "B-STREET_ADDRESS",
    14: "I-STREET_ADDRESS",
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
