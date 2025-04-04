label_map = {
    **{i: chr(65 + i) for i in range(9)},           # A–I (0–8)
    **{i: chr(65 + i + 1) for i in range(10, 25)},  # K–Y (10–24), skipping J
    28: "7",
    29: "6",
    30: "9",
    31: "8",
    32: "4",
    33: "3",
    34: "2",
    35: "5",
}
