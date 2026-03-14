import json

with open('d:/EMI prediction ml project/EMIPredict AI.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('d:/EMI prediction ml project/extracted_code.py', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            out.write(f"# Cell {i}\n")
            out.write(source)
            out.write("\n\n")
