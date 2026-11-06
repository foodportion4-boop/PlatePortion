"""Deterministic templates. No language model or inferred dish names."""
from plateclock.language.style_guide import load_style

def describe(items, actions=None, mode='full', style=None):
    style = style or load_style()
    actions = actions if actions is not None else ['keep'] * len(items)
    if len(items) != len(actions):
        raise ValueError('One action is required per placement')
    pairs = [(p, a) for p, a in zip(items, actions) if a != 'omit']
    if any(a not in {'keep', 'hedge'} for _, a in pairs):
        raise ValueError('Invalid action')
    if not pairs:
        return style['empty']
    if mode == 'summary':
        p, a = max(pairs, key=lambda pair: pair[0].share)
        share = (style['share_large'] if p.share >= style['large_at'] else
                 style['share_small'] if p.share < style['small_below'] else style['share_medium'])
        prefix = style['hedge'] + ' ' if a == 'hedge' else ''
        return style['summary_template'].format(prefix=prefix, share=share.lower() if prefix else share, label=p.label)
    if mode != 'full':
        raise ValueError('Mode must be full or summary')
    if len(pairs) > style['max_items']:
        raise ValueError('Too many spoken items')
    pairs.sort(key=lambda pair: (pair[0].hour % 12, pair[0].angle_deg))
    sentences = [style['count_template'].format(count=len(pairs), noun='item' if len(pairs)==1 else 'items')]
    for p, a in pairs:
        prefix = style['hedge'] + ' ' if a == 'hedge' else ''
        sentences.append(style['item_template'].format(prefix=prefix, label=p.label, hour=p.hour, zone=p.zone))
    return ' '.join(sentences)
