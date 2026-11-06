from pathlib import Path
import yaml

def load_style(path='configs/style_guide.yaml'):
    style = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if style['max_items'] + 1 > style['max_sentences']:
        raise ValueError('Sentence budget cannot fit the item count and opening')
    return style
