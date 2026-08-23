"""Constructor colours - the same hex identifies a constructor everywhere in the UI (lineup, ladder,
alternative-team chips, price-move table), so this is the single source of truth for that mapping."""

TEAM_COLORS = {
    "red_bull":      "#3671C6",
    "ferrari":       "#FF2800",
    "mercedes":      "#27F4D2",
    "mclaren":       "#FF8000",
    "aston_martin":  "#229971",
    "alpine":        "#FF87BC",
    "williams":      "#64C4FF",
    "racing_bulls":  "#6692FF",
    "haas":          "#B6BABD",  # light warm silver
    "audi":          "#AA0000",
    "cadillac":      "#4A4A4A",  # dark charcoal - was #F0F0F0 (invisible on cream) then #8A8D8F
                                 # (too close to Haas' silver); this pairs distinctly with Haas
}