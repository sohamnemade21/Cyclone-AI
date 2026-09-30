import json

with open("CycloneGuard_Final.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("nb_dump.txt", "w", encoding="utf-8") as out:
    out.write(f"Total cells: {len(nb.get('cells', []))}\n")
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        out.write(f"\n========================================\n")
        out.write(f"CELL {i} [{cell_type.upper()}]\n")
        out.write(f"========================================\n")
        out.write(source + "\n")

print("Dumped notebook to nb_dump.txt")
