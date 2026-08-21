"""FIA three-letter driver codes, used for chips and the mobile ladder where full names don't fit."""

FIA_CODES = {
    "alexander_albon": "ALB",
    "andrea_kimi_antonelli": "ANT",
    "arvid_lindblad": "LIN",
    "carlos_sainz": "SAI",
    "charles_leclerc": "LEC",
    "esteban_ocon": "OCO",
    "fernando_alonso": "ALO",
    "franco_colapinto": "COL",
    "gabriel_bortoleto": "BOR",
    "george_russell": "RUS",
    "isack_hadjar": "HAD",
    "lance_stroll": "STR",
    "lando_norris": "NOR",
    "lewis_hamilton": "HAM",
    "liam_lawson": "LAW",
    "max_verstappen": "VER",
    "nico_hulkenberg": "HUL",
    "oliver_bearman": "BEA",
    "oscar_piastri": "PIA",
    "pierre_gasly": "GAS",
    "sergio_perez": "PER",
    "valtteri_bottas": "BOT",
    "yuki_tsunoda": "TSU",
}


# falls back to the first three letters of the surname (upper-cased) for a driver not yet in
# FIA_CODES, so a new/reserve driver doesn't break the ladder before the mapping is updated
def fia_code(driver_id):
    if driver_id in FIA_CODES:
        return FIA_CODES[driver_id]
    surname = driver_id.split("_")[-1]
    return surname[:3].upper()