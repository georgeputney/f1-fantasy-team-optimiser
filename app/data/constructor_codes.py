"""Constructor code abbreviations, used for chips in the alternative-teams list."""

CONSTRUCTOR_CODES = {
    "red_bull":      "RBR",
    "ferrari":       "FER",
    "mercedes":      "MER",
    "mclaren":       "MCL",
    "aston_martin":  "AST",
    "alpine":        "ALP",
    "williams":      "WIL",
    "racing_bulls":  "RB",
    "haas":          "HAA",
    "audi":          "AUD",
    "cadillac":      "CAD",
}


def constructor_code(constructor_id):
    return CONSTRUCTOR_CODES.get(constructor_id, constructor_id[:3].upper())
