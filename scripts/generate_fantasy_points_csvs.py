"""One-off script to generate data/manual/fantasy_points CSVs from PDF data."""
import csv
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "manual", "fantasy_points")

# ---------------------------------------------------------------------------
# 2023  (22 races: R1-R5, R7-R23 -- R6 Imola cancelled)
# ---------------------------------------------------------------------------
DRIVERS_2023 = {
    # driver_id: [22 values, one per race file 01-22]
    "max_verstappen":  [35,61,36,37,64,35,46,35,56,46,46,68,46,38,25,47,62,53,42,45,44,47],
    "sergio_perez":    [28,36,52,61,28,10,27,28,51,26,39,7,30,30,16,-11,-3,32,-10,37,35,30],
    "lewis_hamilton":   [19,16,29,19,24,28,39,23,31,25,20,32,29,17,39,20,6,-4,41,13,26,14],
    "lando_norris":    [-1,7,19,0,1,6,-2,2,29,38,29,15,25,9,30,31,41,46,48,57,-14,22],
    "fernando_alonso":  [39,23,23,27,25,27,14,27,28,12,7,2,62,5,-1,16,25,-9,-16,39,12,16],
    "carlos_sainz":    [19,12,8,21,20,6,19,22,31,4,12,-8,31,38,45,18,-14,32,19,21,34,4],
    "george_russell":  [16,18,-7,25,28,16,31,-12,34,18,27,28,6,17,-2,14,37,25,18,1,14,32],
    "charles_leclerc": [-11,27,-15,29,16,16,14,19,32,6,12,26,-7,24,22,22,22,-10,24,-3,41,29],
    "oscar_piastri":   [-16,2,16,8,5,4,3,3,20,20,18,-7,18,15,24,35,56,-20,14,3,39,19],
    "lance_stroll":    [17,-14,21,18,12,-17,16,14,25,2,9,12,10,7,0,-14,17,10,8,28,39,11],
    "pierre_gasly":    [20,6,1,17,10,10,14,8,7,-3,-19,17,39,3,21,7,11,22,10,24,1,4],
    "yuki_tsunoda":    [8,9,7,-13,16,-4,11,8,4,3,5,7,16,-20,-17,2,11,30,20,22,14,21],
    "kevin_magnussen": [8,9,2,16,6,1,9,0,13,-19,5,9,14,1,8,4,23,12,-16,-21,5,0],
    "nico_hulkenberg": [-1,1,11,2,4,7,3,-1,-4,0,0,13,11,-2,0,10,-18,17,8,-1,2,-3],
    "valtteri_bottas": [13,-2,12,-1,2,10,2,6,5,-5,0,7,14,8,-18,-17,15,10,2,-19,-5,4],
    "logan_sargeant":  [11,2,5,0,1,0,1,-19,19,6,2,3,-18,3,6,-21,-34,16,6,12,-3,5],
    "guanyu_zhou":     [15,6,15,-10,1,11,16,4,10,6,-4,18,-9,3,11,8,19,2,2,-11,12,10],
    "alexander_albon": [13,-17,-15,2,-1,0,5,22,11,8,8,19,17,12,12,-15,23,14,13,-11,0,6],
    "esteban_ocon":    [-17,7,2,7,8,32,9,11,9,-18,-19,23,26,-18,-13,10,2,-10,10,17,40,2],
}

# part-timers sharing the alphatauri seat
# nyck_de_vries: files 01-10 (R1-R11, first 10 races)
# daniel_ricciardo: files 11-12 (R12-R13), then files 18-22 (R19-R23) = 7 races
# liam_lawson: files 13-17 (R14-R18) = 5 races
DRIVERS_2023_PARTTIME = {
    "nyck_de_vries":     {1:8, 2:7, 3:3, 4:-15, 5:0, 6:0, 7:6, 8:-1, 9:9, 10:3},
    "daniel_ricciardo":  {11:2, 12:7, 18:1, 19:12, 20:10, 21:6, 22:11},
    "liam_lawson":       {13:19, 14:5, 15:8, 16:5, 17:-16},
}

CONSTRUCTORS_2023 = {
    "red_bull":      [78,95,98,108,97,63,96,86,112,77,95,80,101,81,54,46,64,95,45,105,84,87],
    "mercedes":      [45,44,32,49,57,54,65,21,70,53,52,70,40,44,47,44,53,21,69,24,45,51],
    "ferrari":       [31,59,3,63,56,37,38,46,73,25,32,38,34,67,70,65,18,25,63,28,78,48],
    "mclaren":       [-16,14,36,23,8,15,11,15,54,68,62,18,56,44,64,66,110,36,57,65,34,59],
    "aston_martin":  [56,19,54,55,42,15,40,46,63,22,21,24,67,17,4,7,47,0,-7,77,56,32],
    "alpine":        [8,28,11,25,28,42,33,24,21,-16,-35,43,66,-16,13,20,23,27,21,44,46,11],
    "alphatauri":    [17,15,13,-23,17,1,20,6,20,5,8,18,40,-12,-4,15,-4,34,42,31,26,27],
    "alfa_romeo":    [31,7,26,-10,8,22,19,11,16,-5,6,26,4,12,-8,-10,39,15,14,-31,12,13],
    "haas":          [12,13,18,17,15,7,17,4,14,-18,10,23,26,0,18,15,6,30,-7,-19,12,2],
    "williams":      [25,-16,-5,5,1,1,5,-2,35,19,9,21,9,20,19,-35,-10,29,20,2,7,12],
}

NUM_RACES_2023 = 22

# ---------------------------------------------------------------------------
# 2024  (24 races: R1-R24)
# ---------------------------------------------------------------------------
DRIVERS_2024 = {
    "max_verstappen":  [45,36,-10,47,58,40,35,13,37,38,25,29,17,35,28,15,18,28,38,21,77,22,39,23],
    "lando_norris":    [16,8,23,19,36,25,37,19,40,51,13,26,28,17,56,36,38,35,30,29,24,29,25,35],
    "charles_leclerc": [22,37,38,31,34,33,24,45,-16,18,16,2,20,22,26,46,28,21,55,35,26,28,38,55],
    "oscar_piastri":   [10,23,21,10,12,22,24,27,17,15,42,23,46,30,20,29,47,26,30,26,16,22,36,10],
    "lewis_hamilton":   [12,6,-19,5,30,20,16,20,32,29,23,47,26,46,21,18,26,17,-10,24,15,46,12,33],
    "george_russell":  [20,15,-3,13,27,10,23,16,27,27,43,-9,27,-21,13,21,26,19,38,20,25,36,25,17],
    "sergio_perez":    [31,31,22,33,40,30,11,-20,-17,19,18,6,20,19,14,15,-2,9,19,10,26,18,-15,-18],
    "fernando_alonso": [7,16,9,14,15,9,2,5,14,1,10,9,4,9,6,6,14,9,-4,-19,1,11,18,7],
    "guanyu_zhou":     [11,-2,4,-17,13,20,5,2,6,4,2,-2,0,-18,-2,4,7,6,6,7,9,5,27,8],
    "pierre_gasly":    [6,-20,5,5,13,15,1,2,14,7,12,-20,-19,3,9,1,-4,4,-1,6,30,-12,21,14],
    "lance_stroll":    [8,-17,16,13,9,-19,12,2,14,6,16,12,9,7,1,4,-3,4,4,11,-18,4,-14,6],
    "nico_hulkenberg": [-3,9,11,10,4,13,3,-32,19,6,20,14,0,2,6,-3,6,6,11,9,-42,17,-12,12],
    "yuki_tsunoda":    [-1,-1,11,10,-6,23,4,7,0,4,10,7,4,6,-3,-17,-19,4,2,-20,17,9,10,8],
    "valtteri_bottas": [0,-1,2,8,-14,10,-2,8,8,1,8,4,1,3,1,6,1,3,1,2,-2,2,12,-16],
    "alexander_albon": [0,6,4,-20,11,10,-20,5,-12,4,12,6,5,2,-2,6,12,-16,11,-18,-15,-16,5,11],
}

# SAI missed R2 (file 02) -- BEA subbed at Ferrari
# MAG missed R17 (file 17) and R21 (file 21) -- BEA subbed at Haas
# RIC raced R1-R18 (files 01-18), LAW raced R19-R24 (files 19-24)
# SAR raced R1-R15 (files 01-15), COL raced R16-R24 (files 16-24)
# OCO raced R1-R23 (files 01-23), DOO raced R24 (file 24)
DRIVERS_2024_PARTTIME = {
    # carlos_sainz missed R2, raced all others
    "carlos_sainz": {
        1:36, 3:28, 4:30, 5:26, 6:17, 7:23, 8:-16, 9:16, 10:28, 11:28,
        12:17, 13:15, 14:20, 15:19, 16:-4, 17:18, 18:42, 19:47, 20:-12,
        21:31, 22:22, 23:27,
    },
    # bearman at ferrari R2
    "oliver_bearman": {2:26, 17:7, 21:14},
    # magnussen missed R17 and R21
    "kevin_magnussen": {
        1:7, 2:7, 3:8, 4:9, 5:17, 6:1, 7:13, 8:-33, 9:14, 10:4,
        11:16, 12:11, 13:7, 14:6, 15:7, 16:9, 18:-1, 19:9, 20:15,
        22:5, 23:21, 24:13,
    },
    # ricciardo R1-R18
    "daniel_ricciardo": {
        1:5, 2:0, 3:8, 4:-19, 5:-9, 6:16, 7:-2, 8:0, 9:9, 10:7,
        11:10, 12:6, 13:1, 14:8, 15:5, 16:1, 17:2, 18:23,
    },
    # lawson R19-R24
    "liam_lawson": {19:20, 20:4, 21:9, 22:2, 23:3, 24:-3},
    # sargeant R1-R15
    "logan_sargeant": {
        1:3, 2:7, 3:9, 4:8, 5:-7, 6:-2, 7:1, 8:-19, 9:1, 10:0,
        11:4, 12:1, 13:2, 14:0, 15:0,
    },
    # colapinto R16-R24
    "franco_colapinto": {16:11, 17:9, 18:6, 19:9, 20:8, 21:-10, 22:10, 23:-16, 24:-20},
    # ocon R1-R23
    "esteban_ocon": {
        1:7, 2:8, 3:0, 4:5, 5:12, 6:11, 7:1, 8:-19, 9:16, 10:4,
        11:4, 12:6, 13:7, 14:10, 15:3, 16:4, 17:7, 18:6, 19:10,
        20:10, 21:36, 22:-5, 23:-13,
    },
    # doohan R24
    "jack_doohan": {24:8},
}

CONSTRUCTORS_2024 = {
    "mclaren":       [36,41,54,39,48,50,64,56,57,66,55,59,74,67,81,83,85,81,80,73,50,66,76,65],
    "red_bull":      [89,90,32,108,126,85,61,8,25,82,56,55,47,69,52,40,26,47,72,41,96,55,44,15],
    "ferrari":       [73,58,92,59,74,79,51,68,-16,44,69,35,47,47,59,65,34,49,97,82,24,72,70,82],
    "mercedes":      [42,36,-17,28,62,40,54,46,74,66,76,38,58,25,39,49,65,49,33,54,45,82,47,55],
    "aston_martin":  [20,9,35,32,29,-7,15,13,38,10,27,31,23,21,17,11,16,18,5,-5,-4,14,9,18],
    "haas":          [9,19,20,20,26,19,21,-65,34,11,41,30,10,7,16,11,16,10,25,34,-29,27,14,30],
    "alpine":        [12,-13,6,11,28,29,5,-12,31,24,21,-12,-10,18,17,18,2,11,14,21,71,-12,9,27],
    "racing_bulls":  [7,4,24,-4,-14,44,12,12,19,10,23,16,25,18,15,-15,-14,22,28,-13,36,16,14,11],
    "kick_sauber":   [10,-4,7,-8,4,29,2,12,13,8,9,3,2,-14,-2,9,17,8,6,10,18,8,35,-3],
    "williams":      [4,14,5,-10,20,4,-21,11,-26,4,11,15,9,5,-2,22,31,-7,19,-5,-20,-5,-12,-10],
}

NUM_RACES_2024 = 24

# ---------------------------------------------------------------------------
# 2025  (24 races: R1-R24)
# ---------------------------------------------------------------------------
DRIVERS_2025 = {
    "max_verstappen":        [29,30,36,22,38,12,57,18,17,29,-15,21,30,11,29,46,45,27,48,26,63,56,55,46],
    "lando_norris":          [59,41,27,32,36,55,31,45,28,-2,36,37,39,36,-5,37,12,25,9,36,49,-10,25,29],
    "oscar_piastri":         [10,45,24,45,38,58,31,23,45,22,39,38,42,29,45,25,-16,19,-2,25,-1,-14,46,28],
    "george_russell":        [25,35,18,31,16,30,15,3,21,47,22,8,19,37,19,19,30,35,24,23,31,30,22,20],
    "charles_leclerc":       [12,-12,20,20,25,-8,24,37,27,17,25,2,30,20,-13,21,6,16,48,29,-2,23,5,35],
    "lewis_hamilton":         [4,-1,11,38,14,32,25,19,14,15,22,24,40,4,-16,26,12,17,32,14,-11,18,10,28],
    "andrea_kimi_antonelli": [32,29,23,14,15,11,-19,-2,-12,35,-18,-16,21,10,1,6,22,17,22,20,40,32,23,5],
    "oliver_bearman":        [2,20,4,22,5,-4,4,8,0,7,14,14,7,-18,27,3,6,6,14,36,30,10,-17,6],
    "alexander_albon":       [17,11,8,15,10,19,18,4,-18,-16,-19,12,14,12,27,15,13,6,20,13,19,-17,5,10],
    "esteban_ocon":          [8,24,1,15,6,7,-18,10,3,10,20,4,3,6,15,1,6,2,-8,9,17,9,6,21],
    "nico_hulkenberg":       [20,2,1,-20,3,9,9,-2,32,15,20,45,7,14,8,-20,4,-8,2,-16,5,11,-18,24],
    "lance_stroll":          [16,17,2,7,1,25,-2,5,-20,2,9,22,9,13,21,-1,-2,10,-3,14,-1,-20,2,20],
    "carlos_sainz":          [-19,8,4,-9,12,0,12,2,7,13,-20,-1,6,6,1,3,33,8,-3,6,15,16,27,3],
    "isack_hadjar":          [-20,11,8,5,8,0,6,12,16,1,6,-20,-7,4,33,12,4,3,3,-2,13,16,1,-2],
    "pierre_gasly":          [1,-8,0,13,-17,18,3,-20,10,6,2,15,-14,0,0,5,5,2,3,9,21,-2,3,3],
    "fernando_alonso":       [-20,-16,4,5,7,-13,4,-16,14,22,12,4,7,17,13,-15,-3,30,-16,-15,2,2,12,17],
    "gabriel_bortoleto":     [-18,7,-2,4,3,-11,-2,3,4,5,25,-20,7,24,0,13,7,-1,17,12,-24,-20,14,6],
}

# tsunoda: R1-R2 at racing_bulls (0, 6), R3-R24 at red_bull (22 values)
# lawson: R1-R2 at red_bull (-17, 22), R3-R24 at racing_bulls (22 values)
# doohan: R1-R6 at alpine (6 values)
# colapinto: R7-R24 at alpine (18 values)
DRIVERS_2025_PARTTIME = {
    "yuki_tsunoda": {
        1:0, 2:6,
        3:13, 4:9, 5:-17, 6:33, 7:9, 8:-5, 9:11, 10:12, 11:6, 12:0,
        13:0, 14:3, 15:11, 16:0, 17:15, 18:4, 19:37, 20:3, 21:11, 22:9, 23:12, 24:1,
    },
    "liam_lawson": {
        1:-17, 2:22,
        3:-4, 4:11, 5:5, 6:-14, 7:3, 8:7, 9:9, 10:-20, 11:13, 12:-18,
        13:9, 14:10, 15:0, 16:6, 17:17, 18:3, 19:15, 20:-18, 21:14, 22:-1, 23:11, 24:1,
    },
    "jack_doohan": {1:-20, 2:11, 3:6, 4:2, 5:1, 6:-14},
    "franco_colapinto": {
        7:0, 8:5, 9:6, 10:2, 11:5, 12:-20, 13:-2, 14:-4, 15:11, 16:0,
        17:1, 18:3, 19:7, 20:4, 21:-16, 22:0, 23:6, 24:0,
    },
}

CONSTRUCTORS_2025 = {
    "mclaren":       [71,101,71,92,86,118,82,83,88,40,100,90,93,100,52,97,6,64,22,86,63,-34,96,77],
    "red_bull":      [19,67,46,46,21,60,71,23,28,51,1,41,55,39,47,56,85,51,92,36,88,85,67,52],
    "mercedes":      [67,69,56,60,46,51,6,9,24,87,19,7,50,54,30,37,67,67,61,58,91,77,57,35],
    "ferrari":       [36,-23,56,73,64,34,67,71,61,57,67,41,70,39,-4,67,28,48,85,58,-3,48,22,83],
    "williams":      [10,26,19,13,27,31,42,16,-8,4,-36,16,27,24,35,23,46,5,27,29,42,9,39,24],
    "haas":          [14,47,10,40,12,8,-13,25,6,22,37,25,15,-9,43,7,13,15,13,45,54,24,-5,34],
    "racing_bulls":  [-10,42,14,22,18,-6,24,31,35,-4,34,-35,17,29,38,27,41,16,34,-10,47,30,22,6],
    "aston_martin":  [1,4,9,18,11,13,14,-4,1,31,24,33,20,45,41,-11,-2,37,-12,2,9,-8,21,47],
    "kick_sauber":   [3,12,0,-22,10,9,13,7,52,26,45,14,24,35,14,-2,17,-4,25,2,-9,-8,4,40],
    "alpine":        [-14,-6,9,22,-6,10,10,-11,31,11,17,5,-13,2,14,6,7,6,18,14,12,3,14,4],
}

NUM_RACES_2025 = 24

# ---------------------------------------------------------------------------
# 2026  (5 races so far: R1-R5)
# ---------------------------------------------------------------------------
DRIVERS_2026 = {
    "andrea_kimi_antonelli": [32,68,50,42,62],
    "charles_leclerc":       [29,51,31,27,27],
    "george_russell":        [39,45,27,42,4],
    "max_verstappen":        [50,14,13,52,26],
    "lewis_hamilton":         [25,48,19,20,42],
    "lando_norris":          [21,-10,24,54,6],
    "oscar_piastri":         [-14,-7,43,41,12],
    "carlos_sainz":          [9,28,4,19,11],
    "franco_colapinto":      [6,18,4,13,25],
    "esteban_ocon":          [9,24,9,14,8],
    "liam_lawson":           [5,35,10,-12,23],
    "oliver_bearman":        [20,34,-14,6,10],
    "pierre_gasly":          [11,20,14,-8,16],
    "sergio_perez":          [4,20,4,19,-11],
    "gabriel_bortoleto":     [13,-14,3,7,5],
    "alexander_albon":       [8,-7,-1,20,-8],
    "isack_hadjar":          [-8,19,5,-16,6],
    "valtteri_bottas":       [-16,3,2,4,12],
    "arvid_lindblad":        [15,7,1,-4,-16],
    "fernando_alonso":       [-14,-7,4,20,-26],
    "lance_stroll":          [-23,-14,-17,17,11],
    "nico_hulkenberg":       [-20,7,10,-29,1],
}

CONSTRUCTORS_2026 = {
    "mercedes":      [96,115,92,104,76],
    "ferrari":       [69,119,75,57,69],
    "mclaren":       [19,-7,72,110,33],
    "red_bull":      [42,45,25,31,44],
    "alpine":        [22,45,25,17,46],
    "haas":          [34,65,-4,23,19],
    "racing_bulls":  [35,50,18,2,22],
    "williams":      [20,22,9,42,6],
    "cadillac":      [-13,22,5,24,2],
    "audi":          [0,-4,23,-27,11],
    "aston_martin":  [-38,-20,-12,38,-14],
}

NUM_RACES_2026 = 5


def write_race_csv(season, race_num, drivers, constructors):
    """Write a single race CSV file."""
    race_id = f"{season}_{race_num:02d}"
    filepath = os.path.join(OUTPUT_DIR, f"{race_id}.csv")
    rows = []
    for driver_id, points in sorted(drivers.items()):
        rows.append([race_id, driver_id, "driver", points])
    for constructor_id, points in sorted(constructors.items()):
        rows.append([race_id, constructor_id, "constructor", points])
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["race_id", "asset_id", "asset_type", "fantasy_points"])
        writer.writerows(rows)


def generate_season(season, full_drivers, parttime_drivers, constructors_data, num_races):
    """Generate all CSVs for a season."""
    for race_idx in range(num_races):
        file_num = race_idx + 1  # 1-indexed
        race_drivers = {}
        # full-season drivers
        for driver_id, values in full_drivers.items():
            race_drivers[driver_id] = values[race_idx]
        # part-time drivers
        for driver_id, race_map in parttime_drivers.items():
            if file_num in race_map:
                race_drivers[driver_id] = race_map[file_num]
        # constructors
        race_constructors = {}
        for constructor_id, values in constructors_data.items():
            race_constructors[constructor_id] = values[race_idx]
        write_race_csv(season, file_num, race_drivers, race_constructors)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_season(2023, DRIVERS_2023, DRIVERS_2023_PARTTIME, CONSTRUCTORS_2023, NUM_RACES_2023)
    generate_season(2024, DRIVERS_2024, DRIVERS_2024_PARTTIME, CONSTRUCTORS_2024, NUM_RACES_2024)
    generate_season(2025, DRIVERS_2025, DRIVERS_2025_PARTTIME, CONSTRUCTORS_2025, NUM_RACES_2025)
    generate_season(2026, DRIVERS_2026, {}, CONSTRUCTORS_2026, NUM_RACES_2026)

    # count files
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")]
    print(f"Generated {len(files)} CSV files in {OUTPUT_DIR}")

    # quick sanity checks
    def check_avg(name, values, expected):
        avg = sum(values) / len(values)
        ok = abs(avg - expected) < 0.15
        if not ok:
            print(f"  MISMATCH {name}: got {avg:.2f}, expected {expected}")

    check_avg("2023 VER", DRIVERS_2023["max_verstappen"], 46.1)
    check_avg("2023 SAR", DRIVERS_2023["logan_sargeant"], 0.1)
    check_avg("2023 RED", CONSTRUCTORS_2023["red_bull"], 84)
    check_avg("2024 VER", DRIVERS_2024["max_verstappen"], 31.4)
    check_avg("2024 MCL", CONSTRUCTORS_2024["mclaren"], 62.8)
    check_avg("2025 VER", DRIVERS_2025["max_verstappen"], 32.3)
    check_avg("2025 MCL", CONSTRUCTORS_2025["mclaren"], 72.7)
    check_avg("2026 ANT", DRIVERS_2026["andrea_kimi_antonelli"], 50.8)
    check_avg("2026 MER", CONSTRUCTORS_2026["mercedes"], 96.6)

    # check part-timer race counts
    ric_23 = DRIVERS_2023_PARTTIME["daniel_ricciardo"]
    law_23 = DRIVERS_2023_PARTTIME["liam_lawson"]
    dev_23 = DRIVERS_2023_PARTTIME["nyck_de_vries"]
    print(f"  2023 AlphaTauri seat: DEV({len(dev_23)}) + RIC({len(ric_23)}) + LAW({len(law_23)}) = {len(dev_23)+len(ric_23)+len(law_23)} races")

    print("Done.")
